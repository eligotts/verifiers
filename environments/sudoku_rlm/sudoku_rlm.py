"""
Sudoku RLM Environment.

A Sudoku game where the model writes Python code to play, rather than
directly outputting moves. The game state lives inside a persistent
Python REPL, enabling context-efficient gameplay.

This environment combines:
- TextArena's Sudoku game (9x9 grid puzzle)
- RLM's Python REPL with persistent state
- Sub-LLM calls for reasoning assistance (optional)

Available difficulty levels:
- Sudoku-v0: Easy (60 clues)
- Sudoku-v0-medium: Medium (40 clues)
- Sudoku-v0-hard: Hard (20 clues)
"""

from typing import Literal

import verifiers as vf
from verifiers.envs.experimental.textarena_rlm_env import (
    TextArenaRLMEnv,
    TextArenaRLMRubric,
)
from verifiers.types import State


# Sudoku-specific system prompt
_SUDOKU_SYSTEM_PROMPT = """You are playing Sudoku in a Python REPL environment.

## IMPORTANT: Game is Already Set Up

The `game` object is **already initialized with a Sudoku puzzle**. Do NOT:
- Import textarena or call `ta.make()` / `textarena.make()`
- Call `game.reset()` - the game has already been reset for you
- Create a new game instance - just use the existing `game` variable

## Sudoku Rules

- Fill a 9x9 grid so that each row, column, and 3x3 box contains digits 1-9
- Empty cells are shown as '.' - you can ONLY place numbers in empty cells
- Pre-filled cells (showing digits 1-9) CANNOT be changed - don't try to overwrite them!
- You win when ALL empty cells are correctly filled

## Available Commands

You have these pre-initialized variables:
- `game` - The Sudoku game instance, ready to play
- `answer` - Dict for your final answer

Key methods:
- `game.step("[row col value]")` - Place a digit. Format: `[row col value]` where row/col are 1-9.
- `game.get_observation()` - Get current board state. Returns `(player_id, observation_text)` tuple.
- `game.state.done` - Boolean, True when ALL cells are filled correctly.

## How to Play

1. **See the board**: Call `_, obs = game.get_observation()` and `print(obs)`
2. **Find empty cells**: Look for '.' in the grid - these are the cells you need to fill
3. **Make moves**: Call `game.step("[row col value]")` to place a digit
4. **Keep going**: Continue making moves until `game.state.done` is True
5. **When done**: Set `answer["content"] = "Solved!"` and `answer["ready"] = True`

## CRITICAL: Keep Playing Until Done!

You MUST continue calling `game.step()` until all cells are filled and `game.state.done` is True.
Do NOT stop early. Check the board for remaining '.' cells and fill them all.

## Example

```python
# See the board
_, obs = game.get_observation()
print(obs)

# Make moves - only target empty cells (shown as '.')
game.step("[1 3 7]")  # Place 7 at row 1, column 3 (if that cell is '.')
game.step("[4 1 8]")  # Place 8 at row 4, column 1 (if that cell is '.')

# Check progress
_, obs = game.get_observation()
print(obs)

# Keep going until done!
if game.state.done:
    answer["content"] = "Solved!"
    answer["ready"] = True
```

## Tips

- ONLY fill cells marked with '.' - never try to overwrite existing numbers
- Check each row, column, and 3x3 box to find the missing digits
- You can make multiple `game.step()` calls in one code block
- Keep playing until there are no more '.' cells on the board

## Initial Game State

{initial_observation}
"""


class SudokuRLMRubric(TextArenaRLMRubric):
    """
    Rubric for Sudoku RLM with completion tracking.

    Rewards:
    - game_reward: 1.0 for win, 0.0 for loss
    - completion_bonus: Bonus for completing with fewer moves
    """

    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = 0.0,
        max_turns_for_bonus: int = 81,  # Max cells in a 9x9 grid
        **kwargs,
    ):
        super().__init__(win_reward=win_reward, lose_reward=lose_reward, **kwargs)
        self.max_turns_for_bonus = max_turns_for_bonus

        # Add efficiency metric
        self.add_metric(self.efficiency_bonus)

    async def efficiency_bonus(self, state: State) -> float:
        """Bonus reward for solving efficiently."""
        if not state.get("game_won", False):
            return 0.0

        turns = state.get("game_turns", self.max_turns_for_bonus)
        if turns <= 0:
            return 0.0

        # Bonus scales inversely with moves used
        # Fewer moves = higher bonus
        bonus = max(0.0, (self.max_turns_for_bonus - turns) / self.max_turns_for_bonus)
        return bonus


class SudokuRLMEnv(TextArenaRLMEnv):
    """
    Sudoku-specific RLM environment with customized system prompt.
    """

    def __init__(
        self,
        game: str = "Sudoku-v0",
        system_prompt: str | None = None,
        **kwargs,
    ):
        # Use Sudoku-specific system prompt if not provided
        if system_prompt is None:
            system_prompt = _SUDOKU_SYSTEM_PROMPT

        super().__init__(
            game=game,
            system_prompt=system_prompt,
            **kwargs,
        )


def load_environment(
    difficulty: Literal["easy", "medium", "hard"] = "easy",
    num_train_examples: int = 100,
    num_eval_examples: int = 10,
    seed: int = 0,
    max_game_turns: int = 100,
    max_iterations: int = 50,
    **kwargs,
) -> vf.Environment:
    """
    Load the Sudoku RLM environment.

    Args:
        difficulty: Puzzle difficulty - "easy" (60 clues), "medium" (40 clues), or "hard" (20 clues)
        num_train_examples: Number of training game instances
        num_eval_examples: Number of evaluation game instances
        seed: Random seed for reproducibility
        max_game_turns: Maximum number of moves allowed per game
        max_iterations: Maximum REPL turns (code executions) per game
        **kwargs: Additional arguments passed to SudokuRLMEnv

    Returns:
        A SudokuRLMEnv configured for Sudoku

    Example:
        ```python
        env = load_environment(difficulty="medium", num_train_examples=50)
        results = await env.evaluate(client, model="gpt-4o")
        ```

    The model will write Python code like:
        ```python
        # See the current board
        _, obs = game.get_observation()
        print(obs)

        # Make a move
        game.step("3 5 7")  # Place 7 at row 3, column 5
        _, feedback = game.get_observation()
        print(feedback)

        # Continue until solved
        if game.state.done:
            answer["content"] = "Solved!"
            answer["ready"] = True
        ```
    """
    # Map difficulty to game ID
    game_id_map = {
        "easy": "Sudoku-v0",
        "medium": "Sudoku-v0-medium",
        "hard": "Sudoku-v0-hard",
    }
    game_id = game_id_map.get(difficulty, "Sudoku-v0")

    rubric = SudokuRLMRubric(
        win_reward=1.0,
        lose_reward=0.0,
        max_turns_for_bonus=81,
    )

    return SudokuRLMEnv(
        game=game_id,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        max_game_turns=max_game_turns,
        max_iterations=max_iterations,
        rubric=rubric,
        **kwargs,
    )
