# Wordle RLM Environment

A Wordle game where the model writes Python code to play, rather than directly outputting guesses.

## Overview

This environment combines:
- **TextArena's Wordle game** - Classic word-guessing with feedback
- **RLM's Python REPL** - Persistent code execution environment
- **Sub-LLM calls** - Optional reasoning assistance via `llm_batch()`

The key difference from the standard Wordle environment is that **game state lives inside the Python REPL as variables**, not in the chat context. This enables:
- Context-efficient gameplay (smaller prompts)
- Model-controlled state management
- Algorithmic approaches to solving

## How It Works

1. The environment initializes a Wordle game inside a sandboxed Python REPL
2. The model writes Python code to interact with the game
3. Variables persist across code executions
4. The model can implement any strategy using code

## Model Interface

The model has access to:

```python
# Game instance (pre-initialized)
game                      # TextArena Wordle game
game.step(action)         # Take an action (guess a word)
game.get_observation()    # Get current state (player_id, observation)
game.state.done           # Boolean: is game over?

# Answer submission
answer["content"] = "..."  # Set final answer
answer["ready"] = True     # Signal completion

# Optional: Sub-LLM calls for reasoning
llm_batch(prompts)         # Get help from sub-LLM
```

## Example Model Code

```python
# Turn 1: Initial guess
_, obs = game.get_observation()
print(obs)

game.step("CRANE")
_, feedback = game.get_observation()
print(f"CRANE: {feedback}")

# Track state for next turn
history = [("CRANE", feedback)]
```

```python
# Turn 2: Use feedback to narrow down
# (history variable persists from Turn 1!)
game.step("SLIMY")
_, feedback = game.get_observation()
history.append(("SLIMY", feedback))
print(f"SLIMY: {feedback}")

if game.state.done:
    answer["content"] = f"Solved! History: {history}"
    answer["ready"] = True
```

## Rewards

- **game_reward**: 1.0 for winning, 0.0 for losing
- **efficiency_bonus**: Bonus for solving in fewer turns (0.0-1.0)
- **game_won**: Boolean metric
- **game_turns**: Number of guesses made

## Usage

```python
from wordle_rlm import load_environment

# Create environment
env = load_environment(
    num_train_examples=100,
    num_eval_examples=10,
    max_game_turns=10,      # Max guesses per game
    max_iterations=30,       # Max REPL turns
)

# Run evaluation
results = await env.evaluate(client, model="gpt-4o")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_examples` | 1000 | Training game instances |
| `num_eval_examples` | 20 | Evaluation game instances |
| `seed` | 0 | Random seed |
| `max_game_turns` | 10 | Maximum guesses per game |
| `max_iterations` | 30 | Maximum REPL code executions |

## Requirements

- `verifiers>=0.1.9.post3`
- `textarena>=0.7.4`
- `nltk>=3.9.2`
- `prime-sandboxes>=0.1.0`

## Comparison to Standard Wordle

| Aspect | Standard Wordle | Wordle RLM |
|--------|----------------|------------|
| Model output | Direct guesses in XML | Python code |
| State management | Chat context | Python variables |
| Context growth | Linear with turns | Model-controlled |
| Strategy | Implicit reasoning | Explicit code |
| Sub-LLM calls | Not available | Available via `llm_batch()` |
