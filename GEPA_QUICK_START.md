# GEPA Environment - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install verifiers gepa
```

### 2. Minimal Example

```python
import verifiers as vf
from datasets import load_dataset

# Load your dataset (must have 'input' and 'answer' columns)
dataset = load_dataset("your_dataset", split="train")
eval_dataset = load_dataset("your_dataset", split="validation")

# Create GEPA environment
env = vf.GEPAEnvironment(
    dataset=dataset,
    eval_dataset=eval_dataset,
    seed_system_prompt="You are a helpful assistant.",
    gepa_reflection_lm="openai/gpt-4o",      # LLM for optimizing prompts
    gepa_judge_lm="openai/gpt-4o-mini",      # LLM for feedback
)

# Train with GRPO (same as before!)
model, tokenizer = vf.get_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
config = vf.GRPOConfig(output_dir="./outputs", num_train_epochs=3, ...)
trainer = vf.GRPOTrainer(model=model, args=config, env=env, processing_class=tokenizer)
trainer.train()

# Check results
print(f"Prompts discovered: {len(env.gepa_state.system_prompt_candidates)}")
print(f"Best prompt: {env.gepa_state.system_prompt_candidates[0]}")
```

## 📋 Checklist

Before running:
- [ ] Dataset has `input` and `answer` columns (NOT `prompt`)
- [ ] Eval dataset is separate from train dataset
- [ ] OpenAI API key is set (for judge/reflection LLMs)
- [ ] vLLM server is running (for fast generation)
- [ ] GPU memory is sufficient

## 🎯 Key Concepts

### What GEPA Does
1. **Maintains** a set of system prompt candidates
2. **Samples** prompts from best performers (Pareto front)
3. **Collects** LLM feedback on completions
4. **Optimizes** prompts every N batches using reflection
5. **Tests** new prompts on validation set
6. **Manipulates** rewards if new prompt is significantly better

### When to Use GEPA
✅ You want to optimize prompts during training
✅ You have a decent eval set (~100+ examples)
✅ You can afford LLM calls for judge/reflection
✅ Your task benefits from good system prompts

❌ You already have the perfect prompt
❌ You have no eval set
❌ You need fastest possible training
❌ Your task doesn't use system prompts

## ⚙️ Essential Parameters

```python
env = vf.GEPAEnvironment(
    # Required
    dataset=...,                          # Training data
    eval_dataset=...,                     # Validation data
    seed_system_prompt="...",             # Starting prompt

    # Core GEPA
    enable_gepa=True,                     # Enable optimization
    gepa_reflection_lm="openai/gpt-4o",   # Prompt optimizer
    gepa_judge_lm="openai/gpt-4o-mini",   # Feedback provider

    # Tuning (defaults are good for most cases)
    gepa_optimization_frequency=100,      # Run GEPA every N batches
    gepa_minibatch_size=5,                # Examples for reflection
    gepa_test_size=10,                    # Examples to test new prompts

    # Reward manipulation (when to reduce gradients)
    gepa_reward_manipulation_threshold=0.1,   # Trigger if 10% improvement
    gepa_reward_manipulation_strength=0.8,    # Flatten by 80%
)
```

## 📊 Monitoring

```python
# During training, check GEPA state
state = env.gepa_state

print(f"Prompts discovered: {len(state.system_prompt_candidates)}")
print(f"Optimization runs: {state.num_optimizations}")
print(f"Pareto front size: {len(state.pareto_front_indices)}")

# View optimization history
for event in state.optimization_history[-5:]:
    print(f"Iteration {event['iteration']}: "
          f"{event['parent_score']:.3f} → {event['new_score']:.3f}")

# Get best prompt
best_idx = max(range(len(state.prompt_scores)), key=lambda i: state.prompt_scores[i])
best_prompt = state.system_prompt_candidates[best_idx]
print(f"\nBest prompt (score {state.prompt_scores[best_idx]:.3f}):")
print(best_prompt)
```

## 🔧 Common Issues

### Issue: "No attribute 'GEPAEnvironment'"
**Fix**: Update verifiers package
```bash
pip install --upgrade verifiers
```

### Issue: "Dataset must have 'input' column"
**Fix**: Your dataset has 'question' instead of 'input'
```python
dataset = dataset.rename_column("question", "input")
```

### Issue: GEPA never runs
**Fix**: Check optimization frequency
```python
# Batch counter / frequency determines when GEPA runs
# If training for 50 batches and frequency=100, GEPA won't trigger
gepa_optimization_frequency=10  # Lower value
```

### Issue: All prompts score 0.0
**Fix**: Check reward function
```python
# Make sure rubric is properly configured
rubric = vf.Rubric()
rubric.add_reward_func(your_reward_func, weight=1.0)
env = vf.GEPAEnvironment(..., rubric=rubric)
```

### Issue: Training very slow
**Fix**: Reduce GEPA calls
```python
gepa_optimization_frequency=200  # Less frequent
gepa_test_size=5                 # Smaller test set
gepa_judge_lm=None               # Disable judge feedback (optional)
```

## 💡 Pro Tips

### Tip 1: Start Simple
```python
# First run: Disable GEPA to establish baseline
env = vf.GEPAEnvironment(..., enable_gepa=False)

# Second run: Enable GEPA and compare
env = vf.GEPAEnvironment(..., enable_gepa=True)
```

### Tip 2: Monitor Costs
```python
# Reflection LLM: ~1 call per optimization
# Judge LLM: batch_size calls per batch (if enabled)

# Cost estimation:
# Reflection: (num_batches / frequency) * cost_per_call
# Judge: num_batches * batch_size * cost_per_call

# Save costs: Use smaller/local models
gepa_reflection_lm="openai/gpt-4o-mini"  # Cheaper
gepa_judge_lm=None                        # Disable if not critical
```

### Tip 3: Save GEPA Results
```python
# Save after training
import json

with open("gepa_results.json", "w") as f:
    json.dump({
        "prompts": env.gepa_state.system_prompt_candidates,
        "scores": env.gepa_state.prompt_scores,
        "history": env.gepa_state.optimization_history,
    }, f, indent=2)

# Use best prompt for inference
best_idx = max(range(len(env.gepa_state.prompt_scores)),
               key=lambda i: env.gepa_state.prompt_scores[i])
production_prompt = env.gepa_state.system_prompt_candidates[best_idx]
```

### Tip 4: Customize Judge Feedback
```python
# Default judge prompt is generic
# Customize for your task:

math_judge_template = """Evaluate this math solution:

Problem: {input}
Correct Answer: {answer}
Student Solution: {completion}
Score: {reward}

Provide feedback on:
1. Mathematical correctness
2. Explanation clarity
3. How to improve the system prompt for better solutions
"""

env = vf.GEPAEnvironment(..., judge_prompt_template=math_judge_template)
```

### Tip 5: Experiment with Reward Manipulation
```python
# Conservative (less manipulation)
gepa_reward_manipulation_threshold=0.2   # Only if 20%+ improvement
gepa_reward_manipulation_strength=0.5    # Flatten by 50%

# Aggressive (more manipulation)
gepa_reward_manipulation_threshold=0.05  # Trigger easily
gepa_reward_manipulation_strength=0.9    # Nearly stop learning
```

## 📚 Next Steps

1. **Read the full docs**: [`docs/GEPA_ENVIRONMENT.md`](docs/GEPA_ENVIRONMENT.md)
2. **Run the example**: [`examples/gepa_example.py`](examples/gepa_example.py)
3. **Check implementation**: [`verifiers/envs/gepa_env.py`](verifiers/envs/gepa_env.py)
4. **Join the community**: [Discord](https://discord.gg/...)

## 📖 Additional Resources

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [GEPA GitHub](https://github.com/gepa-ai/gepa)
- [Verifiers Documentation](https://github.com/your-repo/verifiers)
- [GRPO Trainer Guide](docs/GRPO_TRAINER.md)

## 🙋 Getting Help

**Q: How do I know if GEPA is working?**
```python
# Check optimization counter increases
print(f"Optimizations: {env.gepa_state.num_optimizations}")
# Should be > 0 after (optimization_frequency) batches
```

**Q: Should I always use GEPA?**
No. Use GEPA when:
- You're exploring prompt space
- You have eval data for validation
- You can afford extra LLM calls

Skip GEPA when:
- You have the perfect prompt already
- Speed is critical
- Very limited compute budget

**Q: Can I use local models?**
Yes! Host with vLLM:
```python
# Start vLLM server with your model
# vllm serve meta-llama/Llama-3.1-70B-Instruct --port 8001

# Use it for GEPA
gepa_reflection_lm="local/meta-llama/Llama-3.1-70B-Instruct"
gepa_judge_lm="local/meta-llama/Llama-3.1-70B-Instruct"
```

## ✅ Validation Checklist

Before running production training:
- [ ] Tested on small dataset (100 examples)
- [ ] Verified GEPA optimization runs
- [ ] Checked prompt quality improves
- [ ] Monitored LLM costs
- [ ] Saved best prompts
- [ ] Compared to baseline (no GEPA)
- [ ] Documented optimal hyperparameters

---

**Ready to start? Run the minimal example above! 🚀**

For detailed guidance, see the [full documentation](docs/GEPA_ENVIRONMENT.md).