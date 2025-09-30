"""Factory for the GEPA-aware environment."""

from __future__ import annotations

from typing import Any

from datasets import Dataset

import verifiers as vf


def load_environment(
    *,
    seed_system_prompt: str,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    gepa_val_dataset: Dataset | None = None,
    prompt_optimizer_model: str = "gpt-4.1-mini",
    prompt_optimizer_sampling_args: dict[str, Any] | None = None,
    feedback_model: str | None = None,
    feedback_prompt: str | None = None,
    reward_shrink_factor: float = 0.2,
    improvement_margin: float = 0.0,
    gepa_sample_size: int = 5,
    gepa_rng_seed: int | None = None,
    val_sampling_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.GepaEnvironment:
    """
    Instantiate the GEPA-integrated single-turn environment.

    Parameters
    ----------
    seed_system_prompt:
        Initial system prompt used to seed the GEPA candidate pool.
    train_dataset / eval_dataset:
        Datasets following the usual verifiers schema. At least one must be provided.
    gepa_val_dataset:
        Optional dataset used to evaluate candidate prompts for Pareto updates.
        Defaults to ``eval_dataset`` if provided, otherwise ``train_dataset``.
    prompt_optimizer_model:
        Model used to synthesise improved system prompts.
    feedback_model:
        Model used to provide natural-language feedback for each rollout. Defaults
        to ``prompt_optimizer_model``.
    reward_shrink_factor:
        Factor in ``[0, 1]`` used to contract reward variance when a new prompt is
        accepted. ``0`` freezes updates and ``1`` leaves rewards unchanged.
    improvement_margin:
        Minimum average validation improvement required (in reward units) before
        rewards are down-weighted.
    gepa_sample_size:
        Number of rollouts sampled from each batch to build the reflective dataset
        for prompt refinement.
    gepa_rng_seed:
        Optional RNG seed governing sampling determinism.
    val_sampling_args:
        Optional sampling arguments applied when evaluating candidate prompts on the
        GEPA validation set.
    kwargs:
        Additional keyword arguments are forwarded to :class:`vf.GepaEnvironment`.
    """
    if train_dataset is None and eval_dataset is None:
        raise ValueError("At least one of train_dataset or eval_dataset must be provided")

    environment = vf.GepaEnvironment(
        seed_system_prompt=seed_system_prompt,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        gepa_val_dataset=gepa_val_dataset,
        gepa_sample_size=gepa_sample_size,
        gepa_rng_seed=gepa_rng_seed,
        prompt_optimizer_model=prompt_optimizer_model,
        prompt_optimizer_sampling_args=prompt_optimizer_sampling_args,
        feedback_model=feedback_model,
        feedback_prompt=feedback_prompt,
        reward_shrink_factor=reward_shrink_factor,
        improvement_margin=improvement_margin,
        val_sampling_args=val_sampling_args,
        **kwargs,
    )
    return environment


__all__ = ["load_environment"]
