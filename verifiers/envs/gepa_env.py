"""
GEPA Environment: Integrates GEPA prompt optimization into GRPO training.

This environment extends the base Environment to incorporate GEPA's prompt optimization
process during reinforcement learning. It maintains a Pareto front of system prompts
and dynamically selects/optimizes them based on performance.

Key Features:
- Separates system prompts from input prompts for dynamic optimization
- Maintains GEPA Pareto front of system prompt candidates
- Collects LLM-as-a-judge feedback for reflective optimization
- Manipulates rewards when optimized prompts perform well (to minimize model weight updates)
- Integrates seamlessly with GRPO trainer's advantage computation

Architecture:
1. Dataset is loaded with system_prompt kept separate from inputs
2. During generation, sample a system prompt from GEPA Pareto set
3. Collect completions, scores, and LLM feedback
4. Sample diverse examples for GEPA reflection
5. Optimize prompt using GEPA's reflective mutation
6. Test new prompt on validation examples
7. If new prompt performs well, manipulate rewards to reduce weight updates
"""

import asyncio
import json
import logging
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.envs.environment import Environment
from verifiers.types import (
    ChatMessage,
    GenerateInputs,
    GenerateOutputs,
    Info,
    Messages,
    SamplingArgs,
    State,
)


@dataclass
class GEPAState:
    """Tracks GEPA optimization state throughout training."""

    # System prompt candidates and their scores
    system_prompt_candidates: List[str]  # All system prompts discovered
    pareto_front_indices: List[int]  # Indices of prompts on Pareto front
    prompt_scores: List[float]  # Average score per prompt on valset
    prompt_per_example_scores: List[List[float]]  # Scores per example for each prompt

    # Current batch tracking
    current_system_prompt_idx: Optional[int] = None  # Which prompt was used this batch
    current_batch_feedback: List[Dict[str, Any]] = None  # LLM feedback for current batch

    # Optimization metadata
    num_evals: int = 0  # Total evaluations performed
    num_optimizations: int = 0  # Number of times GEPA optimization ran
    optimization_history: List[Dict[str, Any]] = None  # Track optimization events

    def __post_init__(self):
        if self.current_batch_feedback is None:
            self.current_batch_feedback = []
        if self.optimization_history is None:
            self.optimization_history = []


class GEPAEnvironment(Environment):
    """
    Environment that integrates GEPA prompt optimization with GRPO training.

    This environment extends the base Environment to enable dynamic system prompt
    optimization using GEPA while training the model with GRPO.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        seed_system_prompt: str | None = None,
        few_shot: List[ChatMessage] | None = None,
        # GEPA-specific parameters
        enable_gepa: bool = True,
        gepa_reflection_lm: str | None = None,  # Model for reflecting/proposing new prompts
        gepa_judge_lm: str | None = None,  # Model for LLM-as-a-judge feedback
        gepa_minibatch_size: int = 5,  # How many examples to use for GEPA reflection
        gepa_test_size: int = 10,  # How many examples to test new prompts on
        gepa_optimization_frequency: int = 100,  # Run GEPA every N batches
        gepa_reward_manipulation_threshold: float = 0.1,  # If new prompt improves by this much, manipulate rewards
        gepa_reward_manipulation_strength: float = 0.8,  # How much to flatten rewards (0=no change, 1=all equal)
        # Judge configuration
        judge_sampling_args: Optional[SamplingArgs] = None,
        judge_prompt_template: Optional[str] = None,
        # Diversity sampling
        use_input_diversity: bool = True,
        use_reward_diversity: bool = True,
        # Other parameters
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize GEPA Environment.

        Args:
            dataset: Training dataset (should have 'input' and 'answer' columns, NOT 'prompt')
            eval_dataset: Validation dataset for testing prompts
            seed_system_prompt: Initial system prompt to start with
            enable_gepa: Whether to enable GEPA optimization
            gepa_reflection_lm: LLM for reflecting on failures and proposing new prompts
            gepa_judge_lm: LLM for providing natural language feedback on completions
            gepa_minibatch_size: Number of examples to sample for GEPA reflection
            gepa_test_size: Number of examples to test new prompts on
            gepa_optimization_frequency: How often to run GEPA optimization (in batches)
            gepa_reward_manipulation_threshold: Performance improvement threshold for reward manipulation
            gepa_reward_manipulation_strength: How much to flatten rewards (0-1)
            judge_sampling_args: Sampling args for judge LLM
            judge_prompt_template: Template for judge prompts
            use_input_diversity: Sample diverse inputs for GEPA reflection
            use_reward_diversity: Sample diverse reward distributions for GEPA reflection
            seed: Random seed for reproducibility
        """
        # Don't pass system_prompt to parent - we'll manage it ourselves
        super().__init__(
            dataset=None,  # We'll set this after formatting
            eval_dataset=None,
            system_prompt=None,  # Important: we manage system prompts ourselves
            few_shot=few_shot,
            message_type="chat",  # GEPA works with chat format
            **kwargs
        )

        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")

        # GEPA configuration
        self.enable_gepa = enable_gepa
        self.gepa_reflection_lm = gepa_reflection_lm
        self.gepa_judge_lm = gepa_judge_lm
        self.gepa_minibatch_size = gepa_minibatch_size
        self.gepa_test_size = gepa_test_size
        self.gepa_optimization_frequency = gepa_optimization_frequency
        self.gepa_reward_manipulation_threshold = gepa_reward_manipulation_threshold
        self.gepa_reward_manipulation_strength = gepa_reward_manipulation_strength

        # Judge configuration
        self.judge_sampling_args = judge_sampling_args or {"temperature": 0.0, "max_tokens": 500}
        self.judge_prompt_template = judge_prompt_template or self._default_judge_template()

        # Diversity configuration
        self.use_input_diversity = use_input_diversity
        self.use_reward_diversity = use_reward_diversity

        # Random seed
        self.seed = seed
        self.rng = random.Random(seed)

        # Format datasets - keep system prompt separate
        if dataset is not None:
            self.raw_dataset = dataset
            self.dataset = self._format_dataset_gepa(dataset, few_shot)
        else:
            self.raw_dataset = None

        if eval_dataset is not None:
            self.raw_eval_dataset = eval_dataset
            self.eval_dataset = self._format_dataset_gepa(eval_dataset, few_shot)
        else:
            self.raw_eval_dataset = None

        # Initialize GEPA state
        if seed_system_prompt is None:
            seed_system_prompt = "You are a helpful assistant."

        self.gepa_state = GEPAState(
            system_prompt_candidates=[seed_system_prompt],
            pareto_front_indices=[0],
            prompt_scores=[0.0],  # Will be initialized on first eval
            prompt_per_example_scores=[[]],
        )

        # Track batch counter for optimization frequency
        self.batch_counter = 0

        # Store the few_shot for later use
        self.few_shot = few_shot

        self.logger.info(f"GEPAEnvironment initialized with seed prompt: {seed_system_prompt[:100]}...")
        if self.enable_gepa:
            self.logger.info(f"GEPA optimization enabled (frequency: every {self.gepa_optimization_frequency} batches)")
        else:
            self.logger.info("GEPA optimization disabled")

    def _default_judge_template(self) -> str:
        """Default template for LLM-as-a-judge prompts."""
        return """You are evaluating an AI assistant's response to a question.

Question: {input}
Ground Truth Answer: {answer}

AI Response:
{completion}

Reward Score: {reward}

Please provide detailed feedback on this response:
1. What did the AI do well?
2. What mistakes or issues are present?
3. How could the system prompt be improved to generate better responses?

Provide your feedback in 2-3 sentences, focusing on actionable insights for improving the system prompt."""

    def _format_dataset_gepa(
        self,
        dataset: Dataset,
        few_shot: Optional[List[ChatMessage]] = None,
    ) -> Dataset:
        """
        Format dataset for GEPA: keep system prompt SEPARATE from inputs.

        Instead of creating full 'prompt' with system message, we keep just the
        user input as 'input' column. System prompt will be dynamically added
        during generation.

        Args:
            dataset: Raw dataset with 'question'/'input' and 'answer' columns
            few_shot: Few-shot examples (will be added after system prompt)

        Returns:
            Formatted dataset with 'input', 'answer', 'info' columns
        """
        # Determine question column name
        question_key = "input" if "input" in dataset.column_names else "question"
        answer_key = "answer"

        # Add ID if not present
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))

        def format_row(row):
            # Build just the user message (no system prompt)
            messages = []
            if few_shot:
                messages.extend(few_shot)
            messages.append({"role": "user", "content": row[question_key]})

            # Store the input and metadata
            return {
                "input": row[question_key],  # Raw input text
                "prompt": messages,  # User messages (no system prompt)
                "answer": row.get(answer_key, ""),
                "info": row.get("info", {}),
            }

        dataset = dataset.map(format_row, num_proc=1)

        # Ensure all required columns exist
        assert "input" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "answer" in dataset.column_names

        return dataset

    def _sample_system_prompt(self) -> Tuple[str, int]:
        """
        Sample a system prompt from the Pareto front.

        Returns:
            (system_prompt, prompt_idx)
        """
        if not self.enable_gepa or len(self.gepa_state.pareto_front_indices) == 0:
            return self.gepa_state.system_prompt_candidates[0], 0

        # Sample from Pareto front
        idx = self.rng.choice(self.gepa_state.pareto_front_indices)
        prompt = self.gepa_state.system_prompt_candidates[idx]

        self.logger.debug(f"Sampled system prompt {idx} from Pareto front")
        return prompt, idx

    def _attach_system_prompt(
        self,
        messages: List[ChatMessage],
        system_prompt: str
    ) -> List[ChatMessage]:
        """
        Attach system prompt to the beginning of messages.

        Args:
            messages: User messages (from dataset)
            system_prompt: System prompt to prepend

        Returns:
            Full messages with system prompt
        """
        return [{"role": "system", "content": system_prompt}] + messages

    async def _get_judge_feedback(
        self,
        client: AsyncOpenAI,
        input_text: str,
        answer: str,
        completion: Messages,
        reward: float,
        info: Info,
    ) -> str:
        """
        Get natural language feedback from LLM-as-a-judge.

        Args:
            client: OpenAI client
            input_text: Original input question
            answer: Ground truth answer
            completion: Model's completion
            reward: Reward score for this completion
            info: Additional info

        Returns:
            Natural language feedback string
        """
        if not self.gepa_judge_lm:
            return f"Reward: {reward:.2f}"

        # Extract completion text
        if isinstance(completion, list):
            completion_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in completion
                if msg['role'] == 'assistant'
            ])
        else:
            completion_text = str(completion)

        # Build judge prompt
        judge_prompt = self.judge_prompt_template.format(
            input=input_text,
            answer=answer,
            completion=completion_text,
            reward=reward
        )

        try:
            response = await client.chat.completions.create(
                model=self.gepa_judge_lm,
                messages=[{"role": "user", "content": judge_prompt}],
                **self.judge_sampling_args
            )
            feedback = response.choices[0].message.content or ""
            return feedback.strip()
        except Exception as e:
            self.logger.warning(f"Failed to get judge feedback: {e}")
            return f"Reward: {reward:.2f}"

    async def _collect_batch_feedback(
        self,
        client: AsyncOpenAI,
        results: GenerateOutputs,
    ) -> List[Dict[str, Any]]:
        """
        Collect LLM-as-a-judge feedback for all examples in batch.

        Args:
            client: OpenAI client
            results: Generation results with prompts, completions, rewards

        Returns:
            List of feedback dictionaries with structure:
            {
                'input': str,
                'answer': str,
                'completion': Messages,
                'reward': float,
                'feedback': str (from judge LLM),
                'info': Info
            }
        """
        if not self.gepa_judge_lm:
            # No judge configured, return minimal feedback
            return [
                {
                    'input': self._extract_input_from_prompt(results.prompt[i]),
                    'answer': results.answer[i],
                    'completion': results.completion[i],
                    'reward': results.reward[i],
                    'feedback': f"Reward: {results.reward[i]:.2f}",
                    'info': results.info[i]
                }
                for i in range(len(results.prompt))
            ]

        # Collect feedback in parallel
        feedback_tasks = []
        for i in range(len(results.prompt)):
            input_text = self._extract_input_from_prompt(results.prompt[i])
            task = self._get_judge_feedback(
                client=client,
                input_text=input_text,
                answer=results.answer[i],
                completion=results.completion[i],
                reward=results.reward[i],
                info=results.info[i]
            )
            feedback_tasks.append(task)

        feedbacks = await asyncio.gather(*feedback_tasks)

        # Package results
        batch_feedback = []
        for i in range(len(results.prompt)):
            batch_feedback.append({
                'input': self._extract_input_from_prompt(results.prompt[i]),
                'answer': results.answer[i],
                'completion': results.completion[i],
                'reward': results.reward[i],
                'feedback': feedbacks[i],
                'info': results.info[i],
                'state': results.state[i] if results.state else None
            })

        return batch_feedback

    def _extract_input_from_prompt(self, prompt: Messages) -> str:
        """Extract the user input text from a prompt."""
        if isinstance(prompt, list):
            # Find the last user message
            for msg in reversed(prompt):
                if msg['role'] == 'user':
                    return msg['content']
        return str(prompt)

    def _sample_diverse_examples(
        self,
        feedback_batch: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Sample k diverse examples from feedback batch for GEPA reflection.

        Diversity criteria:
        1. Input diversity: Try to get different types of inputs
        2. Reward diversity: Sample across reward distribution

        Args:
            feedback_batch: All feedback from current batch
            k: Number of examples to sample

        Returns:
            k sampled examples (or fewer if batch is smaller)
        """
        k = min(k, len(feedback_batch))

        if k >= len(feedback_batch):
            return feedback_batch

        if not (self.use_input_diversity or self.use_reward_diversity):
            # Simple random sampling
            return self.rng.sample(feedback_batch, k)

        # Stratified sampling based on reward quantiles
        if self.use_reward_diversity and len(feedback_batch) >= k:
            # Sort by reward
            sorted_feedback = sorted(feedback_batch, key=lambda x: x['reward'])

            # Sample from different quantiles
            sampled = []
            n_bins = min(k, len(sorted_feedback))
            bin_size = len(sorted_feedback) / n_bins

            for i in range(n_bins):
                start_idx = int(i * bin_size)
                end_idx = int((i + 1) * bin_size)
                bin_examples = sorted_feedback[start_idx:end_idx]
                if bin_examples:
                    sampled.append(self.rng.choice(bin_examples))

            return sampled[:k]

        # Fallback to random sampling
        return self.rng.sample(feedback_batch, k)

    def _build_reflective_dataset(
        self,
        sampled_feedback: List[Dict[str, Any]],
        current_system_prompt: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build reflective dataset for GEPA from sampled feedback.

        Args:
            sampled_feedback: Sampled examples with feedback
            current_system_prompt: The system prompt that was used

        Returns:
            Dictionary mapping component name to list of examples:
            {
                "system_prompt": [
                    {
                        "Inputs": str,
                        "Generated Outputs": str,
                        "Feedback": str
                    },
                    ...
                ]
            }
        """
        examples = []

        for item in sampled_feedback:
            # Extract completion text
            if isinstance(item['completion'], list):
                completion_text = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in item['completion']
                    if msg['role'] == 'assistant'
                ])
            else:
                completion_text = str(item['completion'])

            examples.append({
                "Inputs": f"Question: {item['input']}\nGround Truth: {item['answer']}",
                "Generated Outputs": completion_text,
                "Feedback": f"{item['feedback']} (Reward: {item['reward']:.3f})"
            })

        return {"system_prompt": examples}

    async def _propose_new_prompt(
        self,
        client: AsyncOpenAI,
        current_prompt: str,
        reflective_dataset: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Use reflection LLM to propose a new system prompt.

        Args:
            client: OpenAI client
            current_prompt: Current system prompt
            reflective_dataset: Reflective dataset with feedback

        Returns:
            New proposed system prompt
        """
        if not self.gepa_reflection_lm:
            self.logger.warning("No reflection LLM configured, cannot propose new prompt")
            return current_prompt

        # Build reflection prompt
        examples_text = ""
        for example in reflective_dataset["system_prompt"]:
            examples_text += f"\n---\nInputs:\n{example['Inputs']}\n\n"
            examples_text += f"Generated Output:\n{example['Generated Outputs']}\n\n"
            examples_text += f"Feedback:\n{example['Feedback']}\n"

        reflection_prompt = f"""You are an expert at optimizing AI system prompts. You are given:
1. The current system prompt
2. Several examples of inputs, model outputs, and feedback

Your task is to propose an improved system prompt that addresses the issues in the feedback.

Current System Prompt:
```
{current_prompt}
```

Examples with Feedback:
{examples_text}

Based on the feedback, propose an improved system prompt that will help the model generate better responses. Focus on:
- Addressing specific issues mentioned in the feedback
- Maintaining clarity and conciseness
- Adding specific instructions that target the weaknesses
- Keeping the overall structure and tone appropriate

Output ONLY the new system prompt, nothing else."""

        try:
            response = await client.chat.completions.create(
                model=self.gepa_reflection_lm,
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            new_prompt = response.choices[0].message.content or current_prompt
            new_prompt = new_prompt.strip()

            # Remove markdown code blocks if present
            if new_prompt.startswith("```"):
                lines = new_prompt.split("\n")
                new_prompt = "\n".join(lines[1:-1] if len(lines) > 2 else lines)

            self.logger.info(f"Proposed new system prompt: {new_prompt[:100]}...")
            return new_prompt
        except Exception as e:
            self.logger.error(f"Failed to propose new prompt: {e}")
            return current_prompt

    async def _test_prompt(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        test_examples: Dataset,
        sampling_args: Optional[SamplingArgs] = None
    ) -> float:
        """
        Test a system prompt on validation examples.

        Args:
            client: OpenAI client
            model: Model name
            system_prompt: System prompt to test
            test_examples: Dataset of examples to test on
            sampling_args: Sampling arguments

        Returns:
            Average reward on test examples
        """
        # Attach system prompt to all examples
        test_inputs = []
        for example in test_examples:
            full_prompt = self._attach_system_prompt(example['prompt'], system_prompt)
            test_inputs.append({
                'prompt': full_prompt,
                'answer': example['answer'],
                'info': example.get('info', {}),
                'task': example.get('task', 'default')
            })

        # Generate and score
        results = await self.a_generate(
            inputs=test_inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            score_rollouts=True
        )

        # Return average reward
        avg_reward = sum(results.reward) / len(results.reward) if results.reward else 0.0
        return avg_reward

    def _manipulate_rewards(
        self,
        rewards: List[float],
        strength: float = 0.5
    ) -> List[float]:
        """
        Manipulate rewards to reduce gradient magnitude.

        For GRPO, advantages are computed as: advantage = reward - mean(rewards_in_group)
        To reduce weight updates, we want to make advantages smaller.

        Strategy: Move all rewards closer to their mean, which reduces variance
        and thus advantage magnitudes.

        Args:
            rewards: Original rewards
            strength: How much to flatten (0=no change, 1=all equal to mean)

        Returns:
            Manipulated rewards
        """
        if not rewards or strength <= 0:
            return rewards

        mean_reward = sum(rewards) / len(rewards)

        # Move rewards towards mean
        manipulated = [
            r * (1 - strength) + mean_reward * strength
            for r in rewards
        ]

        return manipulated

    def _update_gepa_state(
        self,
        new_prompt: str,
        test_score: float,
        parent_idx: int
    ):
        """
        Update GEPA state with new prompt if it's good.

        Args:
            new_prompt: Newly proposed prompt
            test_score: Score on test set
            parent_idx: Index of parent prompt
        """
        # Add to candidates
        new_idx = len(self.gepa_state.system_prompt_candidates)
        self.gepa_state.system_prompt_candidates.append(new_prompt)
        self.gepa_state.prompt_scores.append(test_score)
        self.gepa_state.prompt_per_example_scores.append([])  # Will be filled later

        # Update Pareto front (simplified: just track best score)
        # In full GEPA, this would be per-example Pareto front
        parent_score = self.gepa_state.prompt_scores[parent_idx]

        if test_score > parent_score:
            self.logger.info(f"New prompt improved: {parent_score:.3f} -> {test_score:.3f}")
            # Add to Pareto front
            self.gepa_state.pareto_front_indices.append(new_idx)

            # Remove dominated prompts (simplified)
            self.gepa_state.pareto_front_indices = [
                idx for idx in self.gepa_state.pareto_front_indices
                if self.gepa_state.prompt_scores[idx] >= test_score * 0.95  # Keep if within 5%
            ]
        else:
            self.logger.info(f"New prompt did not improve: {parent_score:.3f} vs {test_score:.3f}")

        # Log optimization event
        self.gepa_state.optimization_history.append({
            'iteration': self.gepa_state.num_optimizations,
            'parent_idx': parent_idx,
            'new_idx': new_idx,
            'parent_score': parent_score,
            'new_score': test_score,
            'improved': test_score > parent_score
        })

        self.gepa_state.num_optimizations += 1

    async def a_generate(
        self,
        inputs: GenerateInputs | Dataset | dict,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        **kwargs,
    ) -> GenerateOutputs:
        """
        Generate completions with GEPA integration.

        This overrides the base Environment's a_generate to:
        1. Sample a system prompt from GEPA Pareto front
        2. Attach it to all inputs in the batch
        3. Generate completions
        4. Collect LLM-as-a-judge feedback
        5. Periodically run GEPA optimization
        6. Manipulate rewards if optimized prompt performed well

        Returns:
            GenerateOutputs with potentially manipulated rewards
        """
        # Sample system prompt for this batch
        system_prompt, prompt_idx = self._sample_system_prompt()
        self.gepa_state.current_system_prompt_idx = prompt_idx

        # Attach system prompt to all inputs
        if isinstance(inputs, Dataset):
            inputs_list = []
            for i in range(len(inputs)):
                example = inputs[i]
                full_prompt = self._attach_system_prompt(example['prompt'], system_prompt)
                inputs_list.append({
                    'prompt': full_prompt,
                    'answer': example.get('answer', ''),
                    'task': example.get('task', 'default'),
                    'info': example.get('info', {})
                })
            inputs = inputs_list
        elif isinstance(inputs, dict):
            inputs_list = []
            for i in range(len(inputs['prompt'])):
                full_prompt = self._attach_system_prompt(inputs['prompt'][i], system_prompt)
                inputs_list.append({
                    'prompt': full_prompt,
                    'answer': inputs['answer'][i] if 'answer' in inputs else '',
                    'task': inputs['task'][i] if 'task' in inputs else 'default',
                    'info': inputs['info'][i] if 'info' in inputs else {}
                })
            inputs = inputs_list

        # Call parent's a_generate
        results = await super().a_generate(
            inputs=inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            **kwargs
        )

        # Store system prompt in results for GRPO trainer
        # The trainer needs the full prompt to recompute logprobs
        if not hasattr(results, 'metadata'):
            results.metadata = {}
        results.metadata['system_prompt'] = system_prompt
        results.metadata['system_prompt_idx'] = prompt_idx
        results.metadata['gepa_enabled'] = self.enable_gepa

        # Collect LLM-as-a-judge feedback
        if self.enable_gepa and score_rollouts:
            batch_feedback = await self._collect_batch_feedback(client, results)
            self.gepa_state.current_batch_feedback = batch_feedback

            # Increment batch counter
            self.batch_counter += 1

            # Check if we should run GEPA optimization
            should_optimize = (
                self.batch_counter % self.gepa_optimization_frequency == 0
            )

            if should_optimize:
                self.logger.info(f"Running GEPA optimization (batch {self.batch_counter})")

                # Sample diverse examples for reflection
                sampled_feedback = self._sample_diverse_examples(
                    batch_feedback,
                    self.gepa_minibatch_size
                )

                # Build reflective dataset
                reflective_dataset = self._build_reflective_dataset(
                    sampled_feedback,
                    system_prompt
                )

                # Propose new prompt
                new_prompt = await self._propose_new_prompt(
                    client,
                    system_prompt,
                    reflective_dataset
                )

                # Test new prompt on validation set
                if self.eval_dataset is not None and len(self.eval_dataset) > 0:
                    test_size = min(self.gepa_test_size, len(self.eval_dataset))
                    test_examples = self.eval_dataset.shuffle(seed=self.seed).select(range(test_size))

                    test_score = await self._test_prompt(
                        client,
                        model,
                        new_prompt,
                        test_examples,
                        sampling_args
                    )

                    # Update GEPA state
                    self._update_gepa_state(new_prompt, test_score, prompt_idx)

                    # Check if we should manipulate rewards
                    parent_score = self.gepa_state.prompt_scores[prompt_idx]
                    improvement = test_score - parent_score

                    if improvement >= self.gepa_reward_manipulation_threshold:
                        self.logger.info(
                            f"New prompt improved significantly ({improvement:.3f}), "
                            f"manipulating rewards with strength {self.gepa_reward_manipulation_strength}"
                        )
                        results.reward = self._manipulate_rewards(
                            results.reward,
                            self.gepa_reward_manipulation_strength
                        )
                        results.metadata['rewards_manipulated'] = True
                    else:
                        results.metadata['rewards_manipulated'] = False
                else:
                    self.logger.warning("No eval dataset available for testing new prompts")

        return results