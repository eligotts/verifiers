"""GEPA-integrated environment for single-turn tasks."""

from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI

from gepa.strategies.instruction_proposal import InstructionProposalSignature

from verifiers.envs.gepa_manager import (
    GEPAUpdateResult,
    GepaPromptManager,
    PromptEvalRecord,
    PromptEvaluation,
)
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.types import (
    GenerateInputs,
    GenerateOutputs,
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.utils.message_utils import cleanup_messages

DEFAULT_FEEDBACK_PROMPT = """You are assessing how well a system prompt prepares an assistant to answer questions.
You will see:
- The current system prompt the assistant used
- The task input from the user
- The assistant's response
- The expected answer (when available)
- A numeric reward between 0 and 1 indicating correctness

Write concise, constructive feedback that explains:
1. What the assistant did well or poorly relative to the expected answer.
2. Concrete guidance to adjust the system prompt so future answers improve.
Keep the feedback under 8 sentences and avoid repeating the numeric reward.
"""


class GepaEnvironment(SingleTurnEnv):
    """Single-turn environment that maintains a GEPA prompt set during training."""

    def __init__(
        self,
        *,
        seed_system_prompt: str,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        gepa_val_dataset: Dataset | None = None,
        gepa_sample_size: int = 5,
        gepa_rng_seed: int | None = None,
        prompt_optimizer_model: str,
        prompt_optimizer_sampling_args: dict[str, Any] | None = None,
        feedback_model: str | None = None,
        feedback_prompt: str | None = None,
        reward_shrink_factor: float = 0.2,
        improvement_margin: float = 0.0,
        val_sampling_args: SamplingArgs | None = None,
        few_shot: list[Messages] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=None,
            few_shot=None,
            **kwargs,
        )
        if gepa_val_dataset is None:
            gepa_val_dataset = eval_dataset if eval_dataset is not None else dataset
        if gepa_val_dataset is None:
            raise ValueError("gepa_val_dataset must be provided when dataset and eval_dataset are None")

        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.seed_system_prompt = seed_system_prompt
        self._few_shot = few_shot or []
        self.gepa_sample_size = max(1, int(gepa_sample_size))
        self.reward_shrink_factor = max(0.0, min(1.0, reward_shrink_factor))
        self.improvement_margin = improvement_margin
        self.prompt_optimizer_model = prompt_optimizer_model
        self.prompt_optimizer_sampling_args = prompt_optimizer_sampling_args or {}
        self.feedback_model = feedback_model or prompt_optimizer_model
        self.feedback_prompt = feedback_prompt or DEFAULT_FEEDBACK_PROMPT
        self.gepa_val_dataset = gepa_val_dataset
        self._gepa_rng = random.Random(gepa_rng_seed)
        self._manager = GepaPromptManager(
            seed_prompt=seed_system_prompt,
            rng_seed=gepa_rng_seed,
        )
        self._gepa_initialized = False
        self._cached_val_inputs: dict[str, list[Any]] | None = None
        self._val_sampling_args = self._build_eval_sampling_args(val_sampling_args)

    # ------------------------------------------------------------------
    # Utility builders
    # ------------------------------------------------------------------
    def _build_eval_sampling_args(
        self, override: SamplingArgs | None
    ) -> SamplingArgs:
        base = deepcopy(self.sampling_args)
        if override:
            extra = override.get("extra_body")
            if extra:
                base_extra = base.setdefault("extra_body", {})
                base_extra.update(extra)
            for key, value in override.items():
                if key == "extra_body":
                    continue
                base[key] = value
        return base

    def _normalize_inputs(
        self, inputs: GenerateInputs | Dataset | dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        if isinstance(inputs, GenerateInputs):
            payload = inputs.model_dump()
        elif isinstance(inputs, Dataset):
            payload = {}
            for col in inputs.column_names:
                if col == "info":
                    col_vals = inputs[col]
                    if col_vals and isinstance(col_vals[0], str):
                        payload[col] = [json.loads(item) for item in col_vals]
                    else:
                        payload[col] = [dict(item) for item in inputs[col]]
                else:
                    payload[col] = deepcopy(inputs[col])
        else:
            payload = {col: deepcopy(vals) for col, vals in inputs.items()}
        if "prompt" not in payload:
            raise ValueError("prompt column not found in inputs")
        n = len(payload["prompt"])
        if not payload.get("answer"):
            payload["answer"] = [""] * n
        if not payload.get("task"):
            payload["task"] = ["default"] * n
        if not payload.get("info"):
            payload["info"] = [{} for _ in range(n)]
        payload["prompt"] = [cleanup_messages(p) for p in payload["prompt"]]
        return payload

    def _compose_prompt(self, system_prompt: str, user_prompt: Messages) -> Messages:
        if self.message_type == "chat":
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if self._few_shot:
                messages.extend(deepcopy(self._few_shot))
            if isinstance(user_prompt, list):
                messages.extend(deepcopy(user_prompt))
            else:
                messages.append({"role": "user", "content": str(user_prompt)})
            return cleanup_messages(messages)
        if not system_prompt:
            return user_prompt
        return f"{system_prompt}\n\n{user_prompt}"

    def _get_val_inputs(self) -> dict[str, list[Any]]:
        if self._cached_val_inputs is None:
            self._cached_val_inputs = self._normalize_inputs(self.gepa_val_dataset)
        return deepcopy(self._cached_val_inputs)

    async def _ensure_gepa_initialized(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs,
    ) -> None:
        if self._gepa_initialized:
            return
        val_inputs = self._get_val_inputs()
        evaluation = await self._evaluate_prompt_on_dataset(
            client=client,
            model=model,
            system_prompt=self.seed_system_prompt,
            inputs=val_inputs,
            sampling_args=self._val_sampling_args,
        )
        self._manager.initialize(evaluation)
        self._gepa_initialized = True

    # ------------------------------------------------------------------
    # Core generation override
    # ------------------------------------------------------------------
    async def a_generate(
        self,
        inputs: GenerateInputs | Dataset | dict[str, list[Any]],
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        **kwargs: Any,
    ) -> GenerateOutputs:
        await self._ensure_gepa_initialized(
            client=client,
            model=model,
            sampling_args=self.sampling_args,
        )
        sampling_args = sampling_args or {}
        base_inputs = self._normalize_inputs(inputs)
        selected_idx = self._manager.choose_candidate()
        system_prompt = self._manager.get_prompt_text(selected_idx)
        full_prompts = [
            self._compose_prompt(system_prompt, p) for p in base_inputs["prompt"]
        ]
        gen_inputs = GenerateInputs(
            prompt=full_prompts,
            answer=base_inputs.get("answer"),
            info=base_inputs.get("info"),
            task=base_inputs.get("task"),
        )
        results = await super().a_generate(
            inputs=gen_inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            **kwargs,
        )

        feedback = await self._generate_feedback_batch(
            client=client,
            system_prompt=system_prompt,
            results=results,
            rewards=results.reward,
        )
        for idx, state in enumerate(results.state):
            state.setdefault("gepa", {})
            state["gepa"].update(
                {
                    "system_prompt": system_prompt,
                    "prompt_idx": selected_idx,
                    "feedback": feedback[idx],
                    "reward": results.reward[idx],
                }
            )
            results.info[idx].setdefault("gepa", {})
            results.info[idx]["gepa"].update(
                {
                    "system_prompt": system_prompt,
                    "prompt_idx": selected_idx,
                }
            )

        batch_records = self._build_prompt_eval_records(
            results=results,
            feedback=feedback,
        )
        reflective_indices = self._select_gepa_examples(batch_records)
        reflective_dataset = [
            self._build_reflective_example(batch_records[i]) for i in reflective_indices
        ]

        new_prompt = await self._propose_new_prompt(
            client=client,
            current_prompt=system_prompt,
            reflective_dataset=reflective_dataset,
        )
        if new_prompt:
            val_inputs = self._get_val_inputs()
            evaluation = await self._evaluate_prompt_on_dataset(
                client=client,
                model=model,
                system_prompt=new_prompt,
                inputs=val_inputs,
                sampling_args=self._val_sampling_args,
            )
            update_result = self._manager.maybe_add_candidate(
                parent_idx=selected_idx,
                prompt_text=new_prompt,
                evaluation=evaluation,
            )
            self._apply_post_update_logic(
                update_result=update_result,
                current_avg=self._manager.state.program_full_scores_val_set[selected_idx],
                results=results,
            )
        return results

    # ------------------------------------------------------------------
    # Supporting helpers
    # ------------------------------------------------------------------
    async def _generate_feedback_batch(
        self,
        *,
        client: AsyncOpenAI,
        system_prompt: str,
        results: GenerateOutputs,
        rewards: list[float],
    ) -> list[str]:
        feedback: list[str] = []
        for prompt, completion, answer, reward in zip(
            results.prompt,
            results.completion,
            results.answer,
            rewards,
            strict=False,
        ):
            user_text = self._extract_user_text(prompt)
            completion_text = self._messages_to_text(completion)
            formatted = self.feedback_prompt.format(
                system_prompt=system_prompt,
                user_input=user_text,
                model_completion=completion_text,
                expected_answer=answer or "",
                reward=f"{reward:.4f}",
            )
            response = await self.get_model_response(
                client=client,
                model=self.feedback_model,
                prompt=[{"role": "user", "content": formatted}],
                sampling_args=dict(self.prompt_optimizer_sampling_args),
                message_type="chat",
            )
            assert response is not None
            feedback_text = (
                response.choices[0].message.content if response.choices else ""
            )
            feedback_text = (feedback_text or "").strip()
            feedback.append(feedback_text)
        return feedback

    def _extract_user_text(self, prompt: Messages) -> str:
        if isinstance(prompt, list):
            for message in reversed(prompt):
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        return "\n".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    return str(content)
        return str(prompt)

    def _messages_to_text(self, completion: Messages) -> str:
        if isinstance(completion, list):
            texts: list[str] = []
            for message in completion:
                if message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    texts.append(
                        "\n".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    )
                else:
                    texts.append(str(content))
            return "\n".join(texts)
        return str(completion)

    def _build_prompt_eval_records(
        self,
        *,
        results: GenerateOutputs,
        feedback: list[str],
    ) -> list[PromptEvalRecord]:
        metrics = results.metrics
        num_examples = len(results.prompt)
        per_example_metrics: list[dict[str, float]] = [
            {name: metric_vals[i] for name, metric_vals in metrics.items()}
            for i in range(num_examples)
        ]
        records: list[PromptEvalRecord] = []
        for i in range(num_examples):
            records.append(
                PromptEvalRecord(
                    prompt=results.prompt[i],
                    completion=results.completion[i],
                    reward=results.reward[i],
                    metrics=per_example_metrics[i],
                    state=results.state[i],
                    feedback=feedback[i],
                    answer=results.answer[i],
                    info=results.info[i],
                )
            )
        return records

    def _select_gepa_examples(self, records: list[PromptEvalRecord]) -> list[int]:
        n = len(records)
        k = min(self.gepa_sample_size, n)
        if k >= n:
            return list(range(n))
        rewards = [rec.reward for rec in records]
        global_mean = sum(rewards) / len(rewards)
        sorted_indices = sorted(range(n), key=lambda i: rewards[i])
        selected: set[int] = set()
        selected.add(sorted_indices[0])
        if k > 1:
            selected.add(sorted_indices[-1])
        groups: dict[str, list[int]] = {}
        for idx, rec in enumerate(records):
            key = self._input_signature(rec)
            groups.setdefault(key, []).append(idx)
        group_keys = list(groups.keys())
        self._gepa_rng.shuffle(group_keys)
        for key in group_keys:
            if len(selected) >= k:
                break
            candidates = sorted(
                groups[key],
                key=lambda i: abs(records[i].reward - global_mean),
                reverse=True,
            )
            for idx in candidates:
                if idx not in selected:
                    selected.add(idx)
                    break
        while len(selected) < k:
            candidate = self._gepa_rng.choice(sorted_indices)
            selected.add(candidate)
        return sorted(selected)

    def _input_signature(self, record: PromptEvalRecord) -> str:
        info = record.info or {}
        if "id" in info:
            return str(info["id"])
        return self._extract_user_text(record.prompt)

    def _build_reflective_example(self, record: PromptEvalRecord) -> dict[str, Any]:
        return {
            "Inputs": {
                "user_message": self._extract_user_text(record.prompt),
                "expected_answer": record.answer or "",
            },
            "Generated Outputs": self._messages_to_text(record.completion),
            "Feedback": record.feedback or "",
            "Reward": record.reward,
        }

    async def _propose_new_prompt(
        self,
        *,
        client: AsyncOpenAI,
        current_prompt: str,
        reflective_dataset: list[dict[str, Any]],
    ) -> str | None:
        if not reflective_dataset:
            return None
        prompt_text = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current_prompt,
                "dataset_with_feedback": reflective_dataset,
            }
        )
        response = await self.get_model_response(
            client=client,
            model=self.prompt_optimizer_model,
            prompt=[{"role": "user", "content": prompt_text}],
            sampling_args=dict(self.prompt_optimizer_sampling_args),
            message_type="chat",
        )
        assert response is not None
        lm_output = (
            response.choices[0].message.content if response.choices else ""
        )
        lm_output = (lm_output or "").strip()
        extracted = InstructionProposalSignature.output_extractor(lm_output)
        new_prompt = extracted.get("new_instruction", "").strip()
        if not new_prompt or new_prompt == current_prompt:
            return None
        return new_prompt

    async def _evaluate_prompt_on_dataset(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        inputs: dict[str, list[Any]],
        sampling_args: SamplingArgs,
    ) -> PromptEvaluation:
        prompts = [self._compose_prompt(system_prompt, p) for p in inputs["prompt"]]
        rollouts = await self.run_rollouts(
            client=client,
            model=model,
            prompts=prompts,
            answers=inputs["answer"],
            tasks=inputs["task"],
            infos=inputs["info"],
            sampling_args=sampling_args,
        )
        completions = [r[0] for r in rollouts]
        states = [r[1] for r in rollouts]
        scores = await self.rubric.score_rollouts(
            prompts=prompts,
            completions=completions,
            answers=inputs["answer"],
            states=states,
            tasks=inputs["task"],
            infos=inputs["info"],
        )
        metrics_by_name = scores.metrics
        outputs: list[PromptEvalRecord] = []
        for idx, prompt in enumerate(prompts):
            metrics = {name: metrics_by_name[name][idx] for name in metrics_by_name}
            outputs.append(
                PromptEvalRecord(
                    prompt=prompt,
                    completion=completions[idx],
                    reward=scores.reward[idx],
                    metrics=metrics,
                    state=states[idx],
                    answer=inputs["answer"][idx],
                    info=inputs["info"][idx],
                )
            )
        return PromptEvaluation(outputs=outputs, scores=scores.reward)

    def _apply_post_update_logic(
        self,
        *,
        update_result: GEPAUpdateResult,
        current_avg: float,
        results: GenerateOutputs,
    ) -> None:
        results.metrics.setdefault("gepa_accepted", [0.0] * len(results.prompt))
        if not update_result.accepted:
            return
        new_avg = update_result.evaluation.average()
        for i in range(len(results.prompt)):
            results.metrics["gepa_accepted"][i] = 1.0
            results.state[i]["gepa"]["val_score"] = new_avg
            results.state[i]["gepa"]["improvements"] = update_result.improvements
            results.info[i]["gepa"]["val_score"] = new_avg
        if new_avg >= current_avg + self.improvement_margin:
            shrink = self.reward_shrink_factor
            if shrink < 1.0 and shrink > 0.0:
                adjusted = self._shrink_rewards(results.reward, shrink)
                results.reward[:] = adjusted

    def _shrink_rewards(self, rewards: list[float], factor: float) -> list[float]:
        n = len(rewards)
        if n == 0:
            return rewards
        generations = int(self.sampling_args.get("n", 1))
        adjusted = rewards.copy()
        if generations <= 1:
            mean = sum(rewards) / n
            for idx, value in enumerate(rewards):
                adjusted[idx] = mean + factor * (value - mean)
            return adjusted
        for start in range(0, n, generations):
            chunk = rewards[start : start + generations]
            if not chunk:
                continue
            mean = sum(chunk) / len(chunk)
            for i, value in enumerate(chunk):
                adjusted[start + i] = mean + factor * (value - mean)
        return adjusted


__all__ = ["GepaEnvironment"]
