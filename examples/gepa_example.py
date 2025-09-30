"""
Example: Training with GEPA Environment

This example demonstrates how to use the GEPAEnvironment to train a model
with GRPO while simultaneously optimizing the system prompt using GEPA.

The GEPA environment:
1. Maintains a Pareto front of system prompt candidates
2. Dynamically samples prompts during training
3. Collects LLM-as-a-judge feedback on completions
4. Periodically optimizes prompts using reflective mutation
5. Manipulates rewards when optimized prompts perform well

This leads to co-evolution of both the model weights and the system prompt.
"""

import verifiers as vf
from datasets import load_dataset

# =============================================================================
# 1. Load and Prepare Dataset
# =============================================================================

# Load a simple QA dataset (e.g., MMLU, ARC, etc.)
# NOTE: Your dataset should have 'question' and 'answer' columns
# DO NOT include system prompt in the dataset - GEPA manages that
dataset = load_dataset("your_dataset_name", split="train[:1000]")
eval_dataset = load_dataset("your_dataset_name", split="validation[:200]")

# =============================================================================
# 2. Create GEPA Environment
# =============================================================================

# Define initial system prompt - GEPA will optimize this
seed_system_prompt = """You are a helpful AI assistant. Answer questions accurately and concisely."""

# Create environment with GEPA enabled
env = vf.GEPAEnvironment(
    dataset=dataset,
    eval_dataset=eval_dataset,
    seed_system_prompt=seed_system_prompt,

    # GEPA configuration
    enable_gepa=True,
    gepa_reflection_lm="openai/gpt-4o",  # Strong model for reflecting on mistakes
    gepa_judge_lm="openai/gpt-4o-mini",  # Model for providing feedback
    gepa_minibatch_size=5,  # Number of examples to use for reflection
    gepa_test_size=20,  # Number of examples to test new prompts on
    gepa_optimization_frequency=50,  # Optimize every 50 batches

    # Reward manipulation settings
    gepa_reward_manipulation_threshold=0.1,  # If new prompt improves by 10%
    gepa_reward_manipulation_strength=0.7,  # Flatten rewards by 70%

    # Judge configuration
    judge_sampling_args={"temperature": 0.0, "max_tokens": 500},

    # Diversity sampling
    use_input_diversity=True,
    use_reward_diversity=True,

    # Standard environment parameters
    parser=vf.Parser(),
    rubric=vf.Rubric(),
    sampling_args={"temperature": 0.7, "max_tokens": 512},
    seed=42,
)

# =============================================================================
# 3. Create GRPO Trainer
# =============================================================================

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,  # Use 4-bit quantization for efficiency
)

# Configure GRPO training
config = vf.GRPOConfig(
    output_dir="./outputs/gepa_training",

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,

    # GRPO-specific
    num_generations=4,  # Number of rollouts per prompt
    num_iterations=1,  # Use generations once (can be >1 for reuse)

    # vLLM configuration (for fast generation)
    vllm_server_host="localhost",
    vllm_server_port=8000,

    # Logging
    logging_steps=10,
    save_steps=100,
    report_to=["wandb"],
    run_name="gepa_training_example",

    # Other settings
    max_concurrent=512,
    seed=42,
)

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    env=env,
)

# =============================================================================
# 4. Train!
# =============================================================================

print("Starting training with GEPA environment...")
print(f"Initial system prompt: {seed_system_prompt}")
print()

trainer.train()

# =============================================================================
# 5. Inspect GEPA Results
# =============================================================================

print("\n" + "="*80)
print("GEPA Optimization Results")
print("="*80)

# Access GEPA state
gepa_state = env.gepa_state

print(f"\nTotal system prompts discovered: {len(gepa_state.system_prompt_candidates)}")
print(f"Prompts on Pareto front: {len(gepa_state.pareto_front_indices)}")
print(f"Number of optimizations run: {gepa_state.num_optimizations}")

print("\n" + "-"*80)
print("Top System Prompts:")
print("-"*80)

# Sort prompts by score
sorted_indices = sorted(
    range(len(gepa_state.prompt_scores)),
    key=lambda i: gepa_state.prompt_scores[i],
    reverse=True
)

for rank, idx in enumerate(sorted_indices[:5], 1):
    score = gepa_state.prompt_scores[idx]
    prompt = gepa_state.system_prompt_candidates[idx]
    on_pareto = "✓" if idx in gepa_state.pareto_front_indices else " "

    print(f"\n{rank}. [Pareto: {on_pareto}] Score: {score:.4f}")
    print(f"   Prompt: {prompt[:100]}...")

# Print optimization history
print("\n" + "-"*80)
print("Optimization History:")
print("-"*80)

for event in gepa_state.optimization_history[-10:]:  # Last 10 events
    improved = "✓ IMPROVED" if event['improved'] else "✗ No improvement"
    print(f"Iteration {event['iteration']}: "
          f"Parent score: {event['parent_score']:.4f} → "
          f"New score: {event['new_score']:.4f} "
          f"{improved}")

# =============================================================================
# 6. Save Final Model and Prompts
# =============================================================================

# Save model
trainer.save_model("./outputs/gepa_training/final_model")

# Save GEPA prompts
import json
with open("./outputs/gepa_training/gepa_prompts.json", "w") as f:
    json.dump({
        "prompts": gepa_state.system_prompt_candidates,
        "scores": gepa_state.prompt_scores,
        "pareto_indices": gepa_state.pareto_front_indices,
        "optimization_history": gepa_state.optimization_history,
    }, f, indent=2)

print("\n✓ Training complete! Model and prompts saved.")


# =============================================================================
# 7. Using the Best Prompt for Inference
# =============================================================================

# Get the best prompt
best_idx = max(range(len(gepa_state.prompt_scores)), key=lambda i: gepa_state.prompt_scores[i])
best_prompt = gepa_state.system_prompt_candidates[best_idx]

print(f"\nBest system prompt (score: {gepa_state.prompt_scores[best_idx]:.4f}):")
print(best_prompt)

# Use it for inference
from openai import AsyncOpenAI
import asyncio

async def run_inference(question: str):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    messages = [
        {"role": "system", "content": best_prompt},
        {"role": "user", "content": question}
    ]

    response = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content

# Example inference
question = "What is the capital of France?"
answer = asyncio.run(run_inference(question))
print(f"\nQ: {question}")
print(f"A: {answer}")