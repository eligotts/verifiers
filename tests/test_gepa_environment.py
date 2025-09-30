import math

import pytest

from datasets import Dataset

from verifiers.envs.gepa_environment import GepaEnvironment
from verifiers.envs.gepa_manager import GepaPromptManager, PromptEvalRecord, PromptEvaluation


def _build_env(sample_size: int = 3, n_generations: int = 2) -> GepaEnvironment:
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "Question 1"}],
                [{"role": "user", "content": "Question 2"}],
                [{"role": "user", "content": "Question 3"}],
            ],
            "answer": ["A1", "A2", "A3"],
            "task": ["default", "default", "default"],
        }
    )
    env = GepaEnvironment(
        seed_system_prompt="You are helpful.",
        dataset=dataset,
        eval_dataset=dataset,
        gepa_val_dataset=dataset,
        gepa_sample_size=sample_size,
        prompt_optimizer_model="gpt-4.1-mini",
        prompt_optimizer_sampling_args={"max_completion_tokens": 32},
        reward_shrink_factor=0.25,
    )
    env.sampling_args["n"] = n_generations
    return env


def test_shrink_rewards_preserves_group_mean():
    env = _build_env()
    rewards = [0.0, 1.0, 0.25, 0.75]
    adjusted = env._shrink_rewards(rewards, factor=0.2)
    assert len(adjusted) == len(rewards)
    # Means per generation group should remain identical
    for start in range(0, len(rewards), env.sampling_args["n"]):
        chunk = rewards[start : start + env.sampling_args["n"]]
        new_chunk = adjusted[start : start + env.sampling_args["n"]]
        assert math.isclose(sum(chunk) / len(chunk), sum(new_chunk) / len(new_chunk))
        for original, new in zip(chunk, new_chunk, strict=False):
            mean = sum(chunk) / len(chunk)
            # Differences shrink by the factor (within tolerance for floating point)
            if original == mean:
                assert math.isclose(new, mean)
            else:
                shrink_ratio = abs(new - mean) / abs(original - mean)
                assert shrink_ratio == pytest.approx(0.2, rel=1e-4)


def test_select_gepa_examples_covers_unique_inputs():
    env = _build_env(sample_size=3)
    records = [
        PromptEvalRecord(
            prompt=[{"role": "user", "content": f"Question {i//2}"}],
            completion=[{"role": "assistant", "content": f"Answer {i}"}],
            reward=float(i) / 5.0,
            metrics={},
            state={"info": {"id": i}},
            feedback=f"Feedback {i}",
            answer=f"A{i}",
            info={"id": i // 2},
        )
        for i in range(6)
    ]
    selected = env._select_gepa_examples(records)
    assert len(selected) == 3
    selected_groups = {env._input_signature(records[i]) for i in selected}
    assert selected_groups == {"0", "1", "2"}


def test_gepa_manager_accepts_only_improving_candidates():
    manager = GepaPromptManager(seed_prompt="Seed")
    base_outputs = [
        PromptEvalRecord(
            prompt=[{"role": "user", "content": "q"}],
            completion=[{"role": "assistant", "content": "a"}],
            reward=0.2,
            metrics={},
            state={},
        )
        for _ in range(2)
    ]
    manager.initialize(PromptEvaluation(outputs=base_outputs, scores=[0.2, 0.1]))

    same_eval = PromptEvaluation(outputs=base_outputs, scores=[0.2, 0.1])
    update = manager.maybe_add_candidate(parent_idx=0, prompt_text="Seed", evaluation=same_eval)
    assert update.accepted is False

    improved_outputs = [
        PromptEvalRecord(
            prompt=[{"role": "user", "content": "q"}],
            completion=[{"role": "assistant", "content": "a"}],
            reward=score,
            metrics={},
            state={},
        )
        for score in (0.4, 0.15)
    ]
    improved_eval = PromptEvaluation(outputs=improved_outputs, scores=[0.4, 0.15])
    update = manager.maybe_add_candidate(parent_idx=0, prompt_text="Better", evaluation=improved_eval)
    assert update.accepted is True
    assert manager.state.program_candidates[update.candidate_idx]["system_prompt"] == "Better"
