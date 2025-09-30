"""
Utilities for maintaining GEPA prompt candidates inside verifiers environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front

from verifiers.types import Messages, State


@dataclass
class PromptEvalRecord:
    """Container for per-example evaluation artefacts used by GEPA."""

    prompt: Messages
    completion: Messages
    reward: float
    metrics: dict[str, float]
    state: State
    feedback: str | None = None
    answer: str | None = None
    info: dict[str, Any] | None = None


@dataclass
class PromptEvaluation:
    """Aggregated evaluation of a system prompt on the GEPA validation set."""

    outputs: List[PromptEvalRecord]
    scores: List[float]

    def average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


@dataclass
class GEPAUpdateResult:
    """Returned after attempting to add a candidate to the GEPA state."""

    accepted: bool
    parent_idx: int
    candidate_idx: int | None
    evaluation: PromptEvaluation
    improvements: list[bool]


class GepaPromptManager:
    """Wrapper around :class:`GEPAState` for single-component prompt optimization."""

    def __init__(
        self,
        seed_prompt: str,
        rng_seed: int | None = None,
        track_best_outputs: bool = False,
    ) -> None:
        self._seed_candidate = {"system_prompt": seed_prompt}
        self._rng = None
        if rng_seed is not None:
            import random

            self._rng = random.Random(rng_seed)
        self._track_best_outputs = track_best_outputs
        self._state: GEPAState | None = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def state(self) -> GEPAState:
        if self._state is None:
            raise RuntimeError("GEPA prompt manager has not been initialised yet")
        return self._state

    def is_initialized(self) -> bool:
        return self._state is not None

    def initialize(self, evaluation: PromptEvaluation) -> None:
        """
        Initialise the underlying :class:`GEPAState` using the seed prompt and its
        validation evaluation.
        """
        if self._state is not None:
            raise RuntimeError("GEPA prompt manager already initialised")
        self._state = GEPAState(
            seed_candidate=self._seed_candidate,
            base_valset_eval_output=(evaluation.outputs, evaluation.scores),
            track_best_outputs=self._track_best_outputs,
        )
        self._state.num_full_ds_evals = 1
        self._state.total_num_evals = len(evaluation.scores)

    # ------------------------------------------------------------------
    # Candidate sampling
    # ------------------------------------------------------------------
    def choose_candidate(self, strategy: str = "pareto") -> int:
        """
        Select a prompt candidate index according to the provided strategy.
        """
        state = self.state
        if strategy == "best":
            return idxmax(state.program_full_scores_val_set)

        rng = self._rng
        if rng is None:
            import random

            rng = random

        return select_program_candidate_from_pareto_front(
            state.program_at_pareto_front_valset,
            state.per_program_tracked_scores,
            rng,
        )

    def get_prompt_text(self, candidate_idx: int) -> str:
        state = self.state
        candidate = state.program_candidates[candidate_idx]
        prompt = candidate.get("system_prompt")
        if prompt is None:
            raise KeyError(
                f"Candidate {candidate_idx} does not define a 'system_prompt' component"
            )
        return prompt

    # ------------------------------------------------------------------
    # Candidate updates
    # ------------------------------------------------------------------
    def maybe_add_candidate(
        self,
        parent_idx: int,
        prompt_text: str,
        evaluation: PromptEvaluation,
    ) -> GEPAUpdateResult:
        """
        Attempt to add a new prompt to the GEPA state.

        The prompt is only added when it improves the Pareto frontier on at least
        one validation example.
        """
        state = self.state
        total_before = state.total_num_evals
        state.total_num_evals += len(evaluation.scores)
        state.num_full_ds_evals += 1

        pareto_scores = state.pareto_front_valset
        improvements = [
            new_score > old_score
            for new_score, old_score in zip(evaluation.scores, pareto_scores, strict=False)
        ]
        if not any(improvements):
            return GEPAUpdateResult(
                accepted=False,
                parent_idx=parent_idx,
                candidate_idx=None,
                evaluation=evaluation,
                improvements=improvements,
            )

        new_candidate = {"system_prompt": prompt_text}
        new_idx, _ = state.update_state_with_new_program(
            parent_program_idx=[parent_idx],
            new_program=new_candidate,
            valset_score=evaluation.average(),
            valset_outputs=evaluation.outputs,
            valset_subscores=evaluation.scores,
            run_dir=None,
            num_metric_calls_by_discovery_of_new_program=total_before,
        )
        return GEPAUpdateResult(
            accepted=True,
            parent_idx=parent_idx,
            candidate_idx=new_idx,
            evaluation=evaluation,
            improvements=improvements,
        )


__all__ = [
    "PromptEvalRecord",
    "PromptEvaluation",
    "GEPAUpdateResult",
    "GepaPromptManager",
]
