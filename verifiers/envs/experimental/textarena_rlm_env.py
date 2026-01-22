"""
TextArena RLM Environment.

Combines TextArena games with the RLM (Recursive Language Model) environment,
allowing models to write Python code to play text-based games.

The game state lives inside the REPL as Python variables, not in the chat context.
This enables context-efficient gameplay where the model manages state through code.
"""

import logging
import random
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

try:
    import textarena as ta  # type: ignore
except ImportError as e:
    raise ImportError(
        "TextArenaRLMEnv requires textarena. Install with: uv add 'verifiers[ta]'"
    ) from e

try:
    import nltk  # type: ignore

    # Ensure nltk data is available
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except ImportError as e:
    raise ImportError(
        "TextArenaRLMEnv requires nltk. Install with: uv add 'verifiers[ta]'"
    ) from e

logger = logging.getLogger(__name__)


# System prompt template for TextArena games
_TEXTARENA_RLM_SYSTEM_PROMPT = """You are playing a text-based game in a Python REPL environment.

## IMPORTANT: Game is Already Set Up

The `game` object is **already initialized and ready to play**. Do NOT:
- Import textarena or call `ta.make()` / `textarena.make()`
- Call `game.reset()` - the game has already been reset for you
- Create a new game instance - just use the existing `game` variable

## Available Commands

You have these pre-initialized variables:
- `game` - The game instance, ready to play
- `answer` - Dict for your final answer

Key methods:
- `game.step("YOUR_GUESS")` - Submit a guess/action. Returns `(done, info)` tuple.
- `game.get_observation()` - Get current state. Returns `(player_id, observation_text)` tuple.
- `game.state.done` - Boolean, True when game has ended.

## How to Play

1. **Get the current state**: Call `_, obs = game.get_observation()` and `print(obs)` to see feedback
2. **Make a guess**: Call `done, info = game.step("YOUR_GUESS")` with your guess as a string
3. **Check the result**: Print the observation after each step to see if you won or need to guess again
4. **When finished**: Set `answer["content"] = "your result"` and `answer["ready"] = True`

## Example Game Loop

```python
# Make a guess
done, info = game.step("CRANE")

# See the feedback
_, obs = game.get_observation()
print(obs)

# If game is done, report the result
if game.state.done:
    answer["content"] = "Game completed"
    answer["ready"] = True
```

## Initial Game State

{initial_observation}

## Tips

- Variables persist between code executions - use them to track your strategy
- Always print() results so you can see the feedback
- The game automatically tracks your progress - just focus on making good guesses
"""


class TextArenaRLMRubric(vf.Rubric):
    """Rubric for TextArena RLM games."""

    def __init__(self, win_reward: float = 1.0, lose_reward: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.win_reward = win_reward
        self.lose_reward = lose_reward

        # Add reward function
        self.add_reward_func(self.game_reward)

        # Add metrics for tracking
        self.add_metric(self.game_won)
        self.add_metric(self.game_turns)

    async def game_reward(self, state: State) -> float:
        """Reward based on game outcome."""
        if state.get("game_won", False):
            return self.win_reward
        return self.lose_reward

    async def game_won(self, state: State) -> bool:
        """Whether the model won the game."""
        return state.get("game_won", False)

    async def game_turns(self, state: State) -> int:
        """Number of game actions taken."""
        return state.get("game_turns", 0)


class TextArenaRLMEnv(RLMEnv):
    """
    RLM environment where TextArena games run inside the Python REPL.

    The model writes Python code to play text-based games. Game state lives
    as Python variables in the REPL, keeping chat context small while the
    model manages arbitrarily complex state through code.

    Args:
        game: TextArena game ID (e.g., "Wordle-v0", "Sudoku-v0", "TowerOfHanoi-v0-easy")
        num_train_examples: Number of training examples to generate
        num_eval_examples: Number of evaluation examples to generate
        seed: Random seed for reproducibility
        max_game_turns: Maximum game actions before forcing termination
        rubric: Optional custom rubric (default: TextArenaRLMRubric)
        system_prompt: Optional custom system prompt template. Must contain {initial_observation}
            placeholder. If not provided, uses the default TextArena RLM prompt.
        **kwargs: Additional arguments passed to RLMEnv

    Example:
        ```python
        env = TextArenaRLMEnv(game="Wordle-v0", num_train_examples=100)
        ```

    The model will receive a REPL with `game` pre-initialized and can write
    code like:
        ```python
        _, obs = game.get_observation()
        print(obs)

        game.step("CRANE")
        _, feedback = game.get_observation()
        print(feedback)

        if game.state.done:
            answer["content"] = "Won!"
            answer["ready"] = True
        ```
    """

    def __init__(
        self,
        game: str = "Wordle-v0",
        num_train_examples: int = 1000,
        num_eval_examples: int = 0,
        seed: int = 0,
        max_game_turns: int = 100,
        rubric: vf.Rubric | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.game = game
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        self.max_game_turns = max_game_turns

        # Create a reference game to get word list and initial observation
        ta_env = ta.make(env_id=game)
        ta_env.reset(num_players=1)
        _, initial_obs = ta_env.get_observation()
        self.initial_observation = initial_obs

        logger.info(f"[TextArenaRLM] Creating environment for game: {game}")

        # Get word list if available (for games like Wordle, Hangman)
        self.word_list = getattr(ta_env, "word_list", None)
        self.is_word_game = self.word_list is not None

        if not self.is_word_game:
            # For non-word games (Tower of Hanoi, Sudoku, etc.), the answer
            # is determined by the game itself, not by us. We use a placeholder.
            self.word_list = [f"game_instance_{i}" for i in range(100)]
            logger.info(
                f"Game '{game}' is not a word-guessing game. "
                "Answers will be placeholders; game state is randomized by TextArena."
            )

        # Build dataset
        dataset, eval_dataset = self._build_datasets()

        # Build system prompt - use custom if provided, otherwise use default
        # The prompt should contain {initial_observation} placeholder
        prompt_template = system_prompt if system_prompt else _TEXTARENA_RLM_SYSTEM_PROMPT
        system_prompt = prompt_template.format(initial_observation=initial_obs)

        # Default rubric
        if rubric is None:
            rubric = TextArenaRLMRubric()

        max_iterations = kwargs.pop("max_iterations", 50)
        max_startup_wait = kwargs.pop("max_startup_wait_seconds", 300)
        code_timeout = kwargs.pop("code_execution_timeout", 180)

        logger.info(
            f"[TextArenaRLM] Config: max_iterations={max_iterations}, "
            f"max_startup_wait={max_startup_wait}s, code_timeout={code_timeout}s"
        )

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            pip_install_packages="textarena nltk",
            rubric=rubric,
            max_iterations=max_iterations,
            # TextArena + nltk installation can be slow
            max_startup_wait_seconds=max_startup_wait,
            # First import of textarena can be slow
            code_execution_timeout=code_timeout,
            # TextArena requires Python REPL (not bash)
            repl_language="python",
            **kwargs,
        )

    def _build_datasets(self) -> tuple[Dataset, Dataset | None]:
        """Build training and evaluation datasets."""
        random.seed(self.seed)

        rows = []
        for _ in range(self.num_train_examples + self.num_eval_examples):
            # For word games, pick a random word as the answer
            answer = random.choice(self.word_list)
            rows.append(
                {
                    "question": f"Play and win at {self.game}.",
                    "answer": answer,
                }
            )

        dataset = Dataset.from_list(rows[: self.num_train_examples])
        eval_dataset = (
            Dataset.from_list(rows[self.num_train_examples :])
            if self.num_eval_examples > 0
            else None
        )

        return dataset, eval_dataset

    def _get_game_init_code(self, secret: str) -> str:
        """
        Generate Python code that initializes the game in the REPL.

        This code runs once during setup, before the model's first turn.
        """
        # Escape the secret for safe string embedding
        escaped_secret = secret.replace("\\", "\\\\").replace("'", "\\'")

        return f"""
# Initialize TextArena game (run by environment, not model)
# Set up NLTK to use sandbox directory (default ~/nltk_data is outside sandbox)
import os
import nltk

# Configure NLTK data path inside sandbox
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir
nltk.data.path.insert(0, nltk_data_dir)

# Monkey-patch nltk.download to always use our directory
_original_nltk_download = nltk.download
def _patched_download(*args, **kwargs):
    kwargs.setdefault('download_dir', nltk_data_dir)
    kwargs.setdefault('quiet', True)
    return _original_nltk_download(*args, **kwargs)
nltk.download = _patched_download

# Pre-download required data
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')

import textarena as ta

game = ta.make(env_id='{self.game}')
game.reset(num_players=1)

# Set the secret answer for word-guessing games
# (This field exists for Wordle, Hangman, etc.)
if 'secret_word' in game.state.game_state:
    game.state.game_state['secret_word'] = '{escaped_secret}'

# Track game metrics
_game_turn_count = 0
_original_step = game.step

def _tracked_step(action):
    global _game_turn_count
    _game_turn_count += 1
    return _original_step(action)

game.step = _tracked_step

# Confirm initialization
print(f"Game initialized: {{type(game).__name__}}")
"""

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup the RLM environment and initialize the TextArena game."""
        logger.info(f"[TextArenaRLM] Starting setup for game: {self.game}")

        # First, run parent setup (creates sandbox, starts worker, etc.)
        logger.info("[TextArenaRLM] Creating sandbox and installing packages...")
        state = await super().setup_state(state, **kwargs)
        logger.info("[TextArenaRLM] Sandbox ready, worker started")

        # Initialize game metrics
        state["game_won"] = False
        state["game_turns"] = 0
        state["game_result"] = None

        # Get the secret answer for this rollout
        secret = state.get("answer", "")
        if not secret:
            raise vf.DataError() from ValueError("No answer/secret provided in state")

        # Execute initialization code to set up the game in the REPL
        logger.info("[TextArenaRLM] Initializing TextArena game in REPL...")
        init_code = self._get_game_init_code(secret)

        try:
            result = await self._execute_code(init_code, state)
            if result.get("status") == "error":
                logger.error(f"Game initialization failed: {result}")
                raise vf.SandboxError() from Exception(
                    f"Failed to initialize TextArena game: {result.get('result', 'Unknown error')}"
                )
            logger.info(f"[TextArenaRLM] Game initialized: {result.get('stdout', '').strip()}")
        except Exception as e:
            logger.error(f"Error during game initialization: {e}")
            raise

        return state

    async def _check_game_state(self, state: State) -> None:
        """Check the game state and update metrics after each REPL turn."""
        # Execute code to check game state - extract game_info fields explicitly
        check_code = """
import json
_game_info_data = None
if game.state.done and game.state.game_info:
    try:
        # game_info can be: dict keyed by player ID {0: {...}}, or list [{...}]
        gi = game.state.game_info
        if isinstance(gi, dict):
            # Dict keyed by player ID - get player 0's info
            info = gi.get(0, gi.get('0', next(iter(gi.values()), {})))
        elif isinstance(gi, list) and len(gi) > 0:
            info = gi[0]
        else:
            info = gi
        _game_info_data = {
            "reward": info.get("reward", 0) if isinstance(info, dict) else 0,
            "reason": str(info.get("reason", "")) if isinstance(info, dict) else str(info),
        }
    except Exception as e:
        _game_info_data = {"reward": 0, "reason": str(e)}

_game_state = {
    "done": game.state.done,
    "turn_count": _game_turn_count,
    "game_info": _game_info_data,
    "raw_game_info": str(game.state.game_info) if game.state.done else None,
}
print(json.dumps(_game_state))
"""
        try:
            result = await self._execute_code(check_code, state)
            if result.get("status") == "ok" and result.get("stdout"):
                import json

                game_state = json.loads(result["stdout"].strip().split("\n")[-1])
                state["game_turns"] = game_state.get("turn_count", 0)

                if game_state.get("done"):
                    game_info = game_state.get("game_info")
                    raw_info = game_state.get("raw_game_info")
                    logger.info(f"[TextArenaRLM] Game done. raw_game_info={raw_info}, parsed_info={game_info}")
                    if game_info:
                        state["game_result"] = game_info
                        # Check for win: positive reward OR win indicators in reason
                        reward = game_info.get("reward", 0)
                        reason = game_info.get("reason", "").lower()
                        # TextArena uses "congratulations" for wins, "game over" for losses
                        won_by_reason = any(w in reason for w in ["congratulations", "you win", "won", "correct"])
                        lost_by_reason = any(w in reason for w in ["game over", "you lose", "lost", "failed"])
                        state["game_won"] = reward > 0 or (won_by_reason and not lost_by_reason)
                        logger.info(f"[TextArenaRLM] reward={reward}, reason={reason}, won={state['game_won']}")
                    else:
                        # Fallback: game is done but no game_info - mark as completed
                        logger.warning(f"[TextArenaRLM] Game done but game_info is empty! raw={raw_info}")
                        state["game_result"] = {"done": True}
                        state["game_won"] = False
        except Exception as e:
            logger.debug(f"Error checking game state: {e}")

    async def call_python_repl(
        self, code: str, state: Any
    ) -> str:
        """Execute Python code and check game state after each turn."""
        # Call parent implementation
        output = await super().call_python_repl(code, state)

        # Check game state after execution
        await self._check_game_state(state)

        # Add game turn count to output if the game is still running
        if not state.get("game_result"):
            turn_count = state.get("game_turns", 0)
            if turn_count > 0:
                output += f"\n[Game turns: {turn_count}]"

            # Warn if approaching max turns
            if self.max_game_turns > 0 and turn_count >= self.max_game_turns - 2:
                output += f"\n[WARNING: Approaching max game turns ({self.max_game_turns})]"

        return output

    @vf.stop
    async def game_completed(self, state: State) -> bool:
        """Stop when the game is done."""
        return state.get("game_result") is not None

    @vf.stop
    async def max_game_turns_reached(self, state: State) -> bool:
        """Stop when max game turns exceeded."""
        if self.max_game_turns <= 0:
            return False
        return state.get("game_turns", 0) >= self.max_game_turns
