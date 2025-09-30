# GEPA Environment Documentation

## Overview

The `GEPAEnvironment` integrates GEPA (Genetic-Pareto) prompt optimization into the GRPO (Group Relative Policy Optimization) training loop. This enables co-evolution of both model weights and system prompts, leading to better performance with less manual prompt engineering.

## Table of Contents

1. [Background](#background)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Usage Guide](#usage-guide)
5. [Configuration Parameters](#configuration-parameters)
6. [How It Works](#how-it-works)
7. [Best Practices](#best-practices)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

---

## Background

### The Problem

Traditional RL training assumes a fixed system prompt, but:
- Different prompts work better for different types of examples
- Finding the optimal prompt requires extensive manual tuning
- Model weights and prompts should ideally be optimized together

### The Solution: GEPA Environment

GEPA Environment combines:
- **GRPO**: Optimizes model weights via reinforcement learning
- **GEPA**: Optimizes system prompts via reflective evolution
- **Reward Manipulation**: Reduces weight updates when prompt improvements are found

This creates a virtuous cycle where better prompts guide learning, and the model learns to work well with evolving prompts.

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Batch                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Sample System Prompt from GEPA Pareto Front            │
│     - Selects from high-performing prompt candidates        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Attach System Prompt + Generate Completions            │
│     - Full prompt: [System] + [Few-shot] + [User Input]    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Score Completions + Collect LLM-as-Judge Feedback      │
│     - Reward functions evaluate quality                     │
│     - Judge LLM provides natural language feedback          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Every N batches: Run GEPA Optimization                 │
│     a. Sample diverse examples from batch                   │
│     b. Build reflective dataset with feedback               │
│     c. Reflection LLM proposes new system prompt            │
│     d. Test new prompt on validation set                    │
│     e. Add to Pareto front if better                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. If New Prompt Improves Significantly:                  │
│     - Manipulate rewards to reduce gradient magnitude       │
│     - This minimizes model weight changes                   │
│     - Rationale: Good prompt found, don't overfit model     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  6. GRPO Computes Advantages and Updates Weights           │
│     - Advantages computed per GRPO: reward - mean(group)    │
│     - If rewards were manipulated, gradients are smaller    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **GEPA State**
Tracks prompt optimization across training:
```python
@dataclass
class GEPAState:
    system_prompt_candidates: List[str]      # All discovered prompts
    pareto_front_indices: List[int]          # Best prompts
    prompt_scores: List[float]               # Performance scores
    current_system_prompt_idx: Optional[int] # Current prompt
    optimization_history: List[Dict]         # Optimization log
```

#### 2. **Prompt Sampling**
Samples from Pareto front using weighted random selection based on performance.

#### 3. **LLM-as-a-Judge**
Provides structured feedback:
- What worked well
- What mistakes were made
- How to improve the system prompt

#### 4. **Reflective Mutation**
Uses a strong LLM (e.g., GPT-4) to:
- Analyze failures
- Identify patterns
- Propose improved prompts

#### 5. **Reward Manipulation**
When new prompt improves performance:
```python
# Move rewards toward mean to reduce variance
manipulated_reward = (1 - strength) * original + strength * mean
```

---

## Key Features

### 1. Dynamic System Prompt Optimization
- Maintains multiple system prompt candidates
- Selects best prompts via Pareto dominance
- Continuously evolves prompts based on performance

### 2. LLM-as-a-Judge Feedback
- Provides detailed natural language feedback
- Identifies specific issues in completions
- Guides prompt optimization with actionable insights

### 3. Reward Manipulation Strategy
**Problem**: GRPO computes advantages as `reward - mean(rewards_in_group)`. If we find a better prompt, we want the model to use it, not overfit weights.

**Solution**: Flatten reward distribution when good prompt found:
- Reduces advantage magnitudes
- Leads to smaller gradients
- Prevents overfitting to current examples

### 4. Diverse Example Sampling
Samples examples for GEPA reflection using:
- **Input diversity**: Different question types
- **Reward diversity**: Both successes and failures

This ensures the optimized prompt is robust.

### 5. Seamless GRPO Integration
- No changes needed to GRPO trainer
- Works with existing reward functions
- Compatible with all GRPO features

---

## Usage Guide

### Basic Setup

```python
import verifiers as vf
from datasets import load_dataset

# 1. Prepare dataset (no system prompt in dataset!)
dataset = load_dataset("your_dataset", split="train")
eval_dataset = load_dataset("your_dataset", split="validation")

# 2. Define initial system prompt
seed_prompt = "You are a helpful AI assistant."

# 3. Create GEPA environment
env = vf.GEPAEnvironment(
    dataset=dataset,
    eval_dataset=eval_dataset,
    seed_system_prompt=seed_prompt,
    enable_gepa=True,
    gepa_reflection_lm="openai/gpt-4o",
    gepa_judge_lm="openai/gpt-4o-mini",
)

# 4. Train with GRPO as usual
trainer = vf.GRPOTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    env=env,
)

trainer.train()
```

### Dataset Format

**Important**: Your dataset should have `input` and `answer` columns, NOT `prompt`:

```python
# ✓ Correct format
dataset = Dataset.from_dict({
    "input": ["What is 2+2?", "What is the capital of France?"],
    "answer": ["4", "Paris"]
})

# ✗ Wrong format (don't include system prompt)
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        ...
    ]
})
```

The environment will automatically format prompts with the GEPA-selected system prompt.

---

## Configuration Parameters

### Core GEPA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_gepa` | bool | True | Enable/disable GEPA optimization |
| `gepa_reflection_lm` | str | None | Model for proposing new prompts (e.g., "openai/gpt-4o") |
| `gepa_judge_lm` | str | None | Model for providing feedback (e.g., "openai/gpt-4o-mini") |
| `gepa_minibatch_size` | int | 5 | Examples to use for each GEPA reflection |
| `gepa_test_size` | int | 10 | Examples to test new prompts on |
| `gepa_optimization_frequency` | int | 100 | Run GEPA every N batches |

### Reward Manipulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gepa_reward_manipulation_threshold` | float | 0.1 | Min improvement to trigger manipulation |
| `gepa_reward_manipulation_strength` | float | 0.8 | How much to flatten rewards (0-1) |

**Example**: If `threshold=0.1` and `strength=0.8`:
- New prompt must improve by ≥10% to trigger manipulation
- Rewards will be flattened by 80% toward their mean

### Judge Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judge_sampling_args` | dict | `{"temperature": 0.0, "max_tokens": 500}` | Sampling args for judge |
| `judge_prompt_template` | str | (default template) | Template for judge prompts |

### Diversity Sampling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_input_diversity` | bool | True | Sample diverse input types |
| `use_reward_diversity` | bool | True | Sample across reward distribution |

---

## How It Works

### Step-by-Step Process

#### Phase 1: Generation (Every Batch)

1. **Sample System Prompt**
   ```python
   system_prompt, idx = env._sample_system_prompt()
   # Samples from Pareto front of high-performing prompts
   ```

2. **Attach to Inputs**
   ```python
   full_prompt = [
       {"role": "system", "content": system_prompt},
       *few_shot_examples,
       {"role": "user", "content": user_input}
   ]
   ```

3. **Generate Completions**
   - Uses vLLM for fast parallel generation
   - Stores logprobs for GRPO training

4. **Score & Collect Feedback**
   ```python
   # Score with reward functions
   reward = rubric.score_rollout(prompt, completion, answer)

   # Get LLM feedback
   feedback = await judge_lm(
       f"Input: {input}\nOutput: {completion}\nAnswer: {answer}"
   )
   ```

#### Phase 2: GEPA Optimization (Every N Batches)

1. **Sample Diverse Examples**
   ```python
   # Sample across reward quantiles for diversity
   sampled = sample_diverse_examples(batch_feedback, k=5)
   ```

2. **Build Reflective Dataset**
   ```python
   reflective_data = {
       "system_prompt": [
           {
               "Inputs": f"Question: {input}\nAnswer: {answer}",
               "Generated Outputs": completion,
               "Feedback": llm_feedback
           }
           for each sampled example
       ]
   }
   ```

3. **Propose New Prompt**
   ```python
   new_prompt = await reflection_lm(
       f"""Current prompt: {current_prompt}

       Examples with feedback:
       {reflective_data}

       Propose an improved system prompt."""
   )
   ```

4. **Test New Prompt**
   ```python
   # Test on held-out validation set
   test_score = await test_prompt(new_prompt, eval_dataset[:10])
   parent_score = gepa_state.prompt_scores[parent_idx]
   ```

5. **Update Pareto Front**
   ```python
   if test_score > parent_score:
       # Add to candidates
       gepa_state.system_prompt_candidates.append(new_prompt)
       gepa_state.prompt_scores.append(test_score)

       # Add to Pareto front
       gepa_state.pareto_front_indices.append(new_idx)
   ```

6. **Manipulate Rewards (If Improved)**
   ```python
   improvement = test_score - parent_score

   if improvement >= threshold:
       # Flatten rewards to reduce gradients
       mean = sum(rewards) / len(rewards)
       manipulated = [
           r * (1 - strength) + mean * strength
           for r in rewards
       ]
   ```

#### Phase 3: GRPO Training (Every Batch)

1. **Compute Advantages**
   ```python
   # GRPO advantage computation (unchanged)
   mean_grouped = rewards.view(-1, num_gens).mean(dim=1)
   advantages = rewards - mean_grouped.repeat_interleave(num_gens)
   ```

2. **Update Weights**
   - If rewards were manipulated: smaller advantages → smaller gradients
   - If not manipulated: normal GRPO update

---

## Best Practices

### 1. Choosing GEPA Models

**Reflection LLM** (proposes new prompts):
- Use a strong model: GPT-4, Claude 3.5 Sonnet
- Needs to understand subtle patterns in failures
- More expensive but run infrequently

**Judge LLM** (provides feedback):
- Can use smaller model: GPT-4o-mini, Claude Haiku
- Run frequently (every batch if scoring)
- Balance cost vs. feedback quality

### 2. Optimization Frequency

```python
# Too frequent (every batch)
gepa_optimization_frequency=1  # ✗ Too expensive, unstable

# Good balance
gepa_optimization_frequency=50-100  # ✓ Enough data, not too slow

# Too infrequent
gepa_optimization_frequency=1000  # ✗ Misses opportunities
```

**Rule of thumb**: Optimize after seeing 50-100 batches worth of data.

### 3. Reward Manipulation Settings

**Threshold** (when to manipulate):
- Too low (0.01): Manipulate too often, slow learning
- Too high (0.5): Never manipulate, miss opportunities
- **Recommended**: 0.05-0.15 (5-15% improvement)

**Strength** (how much to flatten):
- Too low (0.1): Little effect
- Too high (0.99): Nearly stops learning
- **Recommended**: 0.6-0.8 (60-80% flattening)

### 4. Dataset Preparation

```python
# ✓ Good: Diverse examples
dataset = load_dataset("mmlu")  # Multiple subjects

# ✗ Bad: Too narrow
dataset = load_dataset("math", subset="algebra")  # Only algebra

# ✓ Good: Mix of difficulties
dataset = filter_dataset(lambda x: x['difficulty'] in ['easy', 'medium', 'hard'])

# ✗ Bad: All easy examples
dataset = filter_dataset(lambda x: x['difficulty'] == 'easy')
```

### 5. Monitoring GEPA

```python
# Access GEPA state during/after training
gepa_state = env.gepa_state

# Track key metrics
print(f"Prompts discovered: {len(gepa_state.system_prompt_candidates)}")
print(f"Optimization runs: {gepa_state.num_optimizations}")
print(f"Current Pareto front size: {len(gepa_state.pareto_front_indices)}")

# Plot prompt scores over time
import matplotlib.pyplot as plt
plt.plot(gepa_state.prompt_scores)
plt.xlabel("Prompt Index")
plt.ylabel("Score")
plt.title("GEPA Prompt Evolution")
```

---

## Advanced Topics

### Custom Judge Prompts

You can customize how the judge provides feedback:

```python
custom_judge_template = """You are evaluating an AI response.

Input Question: {input}
Correct Answer: {answer}
AI Response: {completion}
Reward Score: {reward}

Provide feedback in this format:
1. Accuracy: [0-10 score]
2. Completeness: [0-10 score]
3. Issues: [List specific problems]
4. Prompt suggestions: [How to improve the system prompt]
"""

env = vf.GEPAEnvironment(
    ...,
    judge_prompt_template=custom_judge_template
)
```

### Integrating with Custom Reward Functions

```python
# Create custom rubric
rubric = vf.Rubric()

# Add custom reward
async def custom_reward(prompt, completion, answer, state, **kwargs):
    # Your custom logic
    score = compute_custom_score(completion, answer)
    return score

rubric.add_reward_func(custom_reward, weight=1.0, name="custom")

# Use with GEPA environment
env = vf.GEPAEnvironment(
    ...,
    rubric=rubric
)
```

### Multi-Component Prompts

GEPA can optimize multiple prompt components:

```python
# Not directly supported yet, but you can adapt:
# 1. Use concatenated prompts
# 2. Optimize different sections separately
# 3. Track which section each optimization targets

# Example: System prompt + Task instructions
seed_prompt = """[SYSTEM]
You are a helpful assistant.

[TASK INSTRUCTIONS]
Answer questions accurately and concisely."""

# GEPA will optimize the full prompt
# You can parse sections in your judge feedback
```

### Saving and Loading GEPA State

```python
# Save GEPA state
import json

with open("gepa_state.json", "w") as f:
    json.dump({
        "prompts": env.gepa_state.system_prompt_candidates,
        "scores": env.gepa_state.prompt_scores,
        "pareto_indices": env.gepa_state.pareto_front_indices,
        "optimization_history": env.gepa_state.optimization_history,
    }, f, indent=2)

# Load and resume training
with open("gepa_state.json", "r") as f:
    saved_state = json.load(f)

# Create environment with loaded state
env = vf.GEPAEnvironment(...)
env.gepa_state.system_prompt_candidates = saved_state["prompts"]
env.gepa_state.prompt_scores = saved_state["scores"]
env.gepa_state.pareto_front_indices = saved_state["pareto_indices"]
```

---

## Troubleshooting

### Issue: GEPA never runs optimization

**Symptoms**:
- `gepa_state.num_optimizations` stays at 0
- No new prompts discovered

**Solutions**:
1. Check `gepa_optimization_frequency` - might be too high
2. Ensure `enable_gepa=True`
3. Verify `gepa_reflection_lm` is set correctly
4. Check logs for errors

### Issue: All prompts score 0.0

**Symptoms**:
- `prompt_scores` are all 0.0
- No prompts added to Pareto front

**Solutions**:
1. Check reward functions are working
2. Verify eval dataset has correct format
3. Ensure model is generating valid completions
4. Check if scoring is enabled (`score_rollouts=True`)

### Issue: Rewards always manipulated / never manipulated

**Symptoms**:
- Every optimization triggers manipulation (or never does)

**Solutions**:
1. Adjust `gepa_reward_manipulation_threshold`
   - Lower if never triggering
   - Raise if always triggering
2. Check if new prompts are actually improving
3. Verify test set is representative

### Issue: Out of memory

**Symptoms**:
- CUDA OOM during GEPA optimization

**Solutions**:
1. Reduce `gepa_test_size`
2. Reduce `gepa_minibatch_size`
3. Use smaller judge/reflection models
4. Enable gradient checkpointing

### Issue: Training is slow

**Symptoms**:
- Much slower than regular GRPO training

**Solutions**:
1. Increase `gepa_optimization_frequency` (less frequent GEPA)
2. Use faster judge LLM (e.g., GPT-4o-mini instead of GPT-4)
3. Reduce `gepa_test_size`
4. Disable judge feedback if not needed (`gepa_judge_lm=None`)
5. Use vLLM for judge/reflection models too

---

## Examples

### Example 1: Math Problem Solving

```python
from datasets import load_dataset
import verifiers as vf

# Load MATH dataset
dataset = load_dataset("hendrycks/math", split="train")
eval_dataset = load_dataset("hendrycks/math", split="test[:200]")

# Math-specific seed prompt
seed_prompt = """You are an expert mathematician. Solve problems step-by-step,
showing your work clearly. Provide the final answer in \\boxed{} format."""

# Math rubric
parser = vf.XMLParser(answer_tag="answer")
rubric = vf.Rubric()
rubric.add_reward_func(
    vf.math_reward,  # Checks if answer is numerically correct
    weight=1.0
)

# Create GEPA environment
env = vf.GEPAEnvironment(
    dataset=dataset,
    eval_dataset=eval_dataset,
    seed_system_prompt=seed_prompt,
    parser=parser,
    rubric=rubric,
    gepa_reflection_lm="openai/gpt-4o",
    gepa_judge_lm="openai/gpt-4o-mini",
    gepa_optimization_frequency=50,
)

# Train
trainer = vf.GRPOTrainer(model=model, args=config, env=env, ...)
trainer.train()
```

### Example 2: Coding Tasks

```python
# HumanEval dataset
dataset = load_dataset("openai_humaneval", split="test")

# Coding seed prompt
seed_prompt = """You are an expert programmer. Write clean, correct,
and efficient Python code. Include type hints and docstrings."""

# Coding rubric (assumes you have execution-based rewards)
rubric = vf.Rubric()
rubric.add_reward_func(execution_reward, weight=1.0)

env = vf.GEPAEnvironment(
    dataset=dataset,
    seed_system_prompt=seed_prompt,
    rubric=rubric,
    gepa_reflection_lm="openai/gpt-4o",
    # Use code-aware judge
    judge_prompt_template="""Evaluate this code:

    Task: {input}
    Generated Code: {completion}
    Expected: {answer}
    Execution Result: {reward}

    Provide feedback on:
    1. Correctness
    2. Code quality
    3. How to improve the system prompt for better code generation
    """
)
```

---

## FAQ

**Q: Can I use GEPA without GRPO?**

A: Currently, GEPA Environment is designed for GRPO training. For standalone GEPA optimization, use the main [gepa package](https://github.com/gepa-ai/gepa).

**Q: Does this work with multi-turn environments?**

A: The current implementation is designed for single-turn tasks. Multi-turn support could be added by extending the feedback collection to include dialogue history.

**Q: How much does GEPA slow down training?**

A: With default settings (`gepa_optimization_frequency=100`), overhead is ~5-10%. Most time is spent in generation (vLLM) and scoring, which happen regardless of GEPA.

**Q: Can I use local models for judge/reflection?**

A: Yes! Use vLLM-hosted local models:
```python
gepa_reflection_lm="local/my-model",
gepa_judge_lm="local/my-model",
```

**Q: What if my task doesn't have ground truth answers?**

A: Use judge-based rewards:
```python
rubric = vf.JudgeRubric(
    judge_lm="openai/gpt-4o-mini",
    judge_prompt="Rate this response from 0-1..."
)
```

---

## Citation

If you use GEPA Environment in your research, please cite both GEPA and Verifiers:

```bibtex
@article{gepa2025,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and Tan, Shangyin and ...},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Discord**: [Join our Discord](https://discord.gg/...)

---

**Happy training with GEPA! 🚀**