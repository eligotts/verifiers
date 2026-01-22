"""
Wordle RLM Environment.

A Wordle game where the model writes Python code to play, rather than
directly outputting guesses. The game state lives inside a persistent
Python REPL, enabling context-efficient gameplay.

This environment combines:
- TextArena's Wordle game
- RLM's Python REPL with persistent state
- Sub-LLM calls for reasoning assistance (optional)
"""

import verifiers as vf
from verifiers.envs.experimental.textarena_rlm_env import (
    TextArenaRLMEnv,
    TextArenaRLMRubric,
)
from verifiers.types import State


class WordleRLMRubric(TextArenaRLMRubric):
    """
    Extended rubric for Wordle RLM with efficiency bonus.

    Rewards:
    - game_reward: 1.0 for win, 0.0 for loss
    - efficiency_bonus: Bonus for winning in fewer turns
    """

    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = 0.0,
        max_turns_for_bonus: int = 6,
        **kwargs,
    ):
        super().__init__(win_reward=win_reward, lose_reward=lose_reward, **kwargs)
        self.max_turns_for_bonus = max_turns_for_bonus

        # Add efficiency metric
        self.add_metric(self.efficiency_bonus)

    async def efficiency_bonus(self, state: State) -> float:
        """Bonus reward for solving in fewer turns."""
        if not state.get("game_won", False):
            return 0.0

        turns = state.get("game_turns", self.max_turns_for_bonus)
        if turns <= 0:
            return 0.0

        # Bonus scales inversely with turns used
        # 1 turn = 1.0 bonus, 6 turns = 0.0 bonus
        bonus = max(0.0, (self.max_turns_for_bonus - turns) / self.max_turns_for_bonus)
        return bonus


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 20,
    seed: int = 0,
    max_game_turns: int = 10,
    max_iterations: int = 30,
    **kwargs,
) -> vf.Environment:
    """
    Load the Wordle RLM environment.

    Args:
        num_train_examples: Number of training game instances
        num_eval_examples: Number of evaluation game instances
        seed: Random seed for reproducibility
        max_game_turns: Maximum number of guesses allowed per game
        max_iterations: Maximum REPL turns (code executions) per game
        **kwargs: Additional arguments passed to TextArenaRLMEnv

    Returns:
        A TextArenaRLMEnv configured for Wordle

    Example:
        ```python
        env = load_environment(num_train_examples=100)
        results = await env.evaluate(client, model="gpt-4o")
        ```

    The model will write Python code like:
        ```python
        # Check initial state
        _, obs = game.get_observation()
        print(obs)

        # Make a guess
        game.step("CRANE")
        _, feedback = game.get_observation()
        print(feedback)

        # Continue until done
        if game.state.done:
            answer["content"] = "Won!"
            answer["ready"] = True
        ```
    """
    rubric = WordleRLMRubric(
        win_reward=1.0,
        lose_reward=0.0,
        max_turns_for_bonus=6,
    )

    return TextArenaRLMEnv(
        game="Wordle-v0",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        max_game_turns=max_game_turns,
        max_iterations=max_iterations,
        rubric=rubric,
        **kwargs,
    )
