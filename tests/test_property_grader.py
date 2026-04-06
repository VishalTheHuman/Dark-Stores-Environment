# Feature: dark-store-simulator, Property 15: Grader Normalization
"""
Property-based test for DarkStoreEnvironment grader normalization.

**Validates: Requirements 7.1, 7.2**

For any cumulative reward and max theoretical reward, grader score =
max(0.0, cumulative_reward / max_theoretical_reward) clamped to [0.0, 1.0].

Tests compute_score() directly after running episodes with different outcomes.
Tests that score is always in [0.0, 1.0].
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment, TASK_REGISTRY
from my_env.models import DarkStoreAction

TASK_NAMES = ["single_order", "concurrent_orders", "full_operations"]
task_name_st = st.sampled_from(TASK_NAMES)

# Actions that exercise different reward paths
action_st = st.one_of(
    st.just(DarkStoreAction(action="wait")),
    st.builds(
        DarkStoreAction,
        action=st.just("move_picker"),
        row=st.integers(min_value=0, max_value=9),
        col=st.integers(min_value=0, max_value=7),
    ),
    st.builds(
        DarkStoreAction,
        action=st.just("pick"),
        item_name=st.sampled_from(["milk", "chips", "coke", "juice"]),
        row=st.integers(min_value=0, max_value=9),
        col=st.integers(min_value=0, max_value=7),
    ),
    st.builds(
        DarkStoreAction,
        action=st.just("restock"),
        item_name=st.sampled_from(["coke", "juice"]),
        quantity=st.integers(min_value=1, max_value=5),
    ),
)


@given(
    task_name=task_name_st,
    actions=st.lists(action_st, min_size=0, max_size=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_grader_score_in_unit_range(task_name, actions):
    """
    **Validates: Requirements 7.1, 7.2**

    For any sequence of actions, compute_score() should return a value
    in [0.0, 1.0].
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    for action in actions:
        if obs.done:
            break
        obs = env.step(action)

    score = env.compute_score()

    assert 0.0 <= score <= 1.0, (
        f"Grader score should be in [0.0, 1.0], got {score} "
        f"(cumulative_reward={obs.cumulative_reward})"
    )


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_grader_score_zero_on_reset(task_name):
    """
    **Validates: Requirements 7.1**

    On reset (cumulative_reward=0.0), compute_score() should return 0.0.
    """
    env = DarkStoreEnvironment()
    env.reset(task=task_name)

    score = env.compute_score()
    assert score == 0.0, (
        f"Grader score on reset should be 0.0, got {score}"
    )


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_grader_score_matches_formula(task_name):
    """
    **Validates: Requirements 7.1, 7.2**

    After running some actions, compute_score() should equal
    max(0.0, min(1.0, cumulative_reward / max_theoretical_reward)).
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    # Run a few wait actions to potentially trigger expiry penalties
    for _ in range(5):
        if obs.done:
            break
        obs = env.step(DarkStoreAction(action="wait"))

    score = env.compute_score()
    max_reward = TASK_REGISTRY[task_name].max_theoretical_reward

    expected = max(0.0, min(1.0, obs.cumulative_reward / max_reward))

    assert abs(score - expected) < 1e-6, (
        f"Grader score ({score}) should match formula result ({expected}) "
        f"for cumulative_reward={obs.cumulative_reward}, max={max_reward}"
    )


def test_grader_clamps_negative_to_zero():
    """
    **Validates: Requirements 7.2**

    When cumulative reward is negative, compute_score() should return 0.0.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    # full_operations has bread expiring at tick 3 and coke/juice at stock=0
    # Wait for expiry to trigger negative rewards
    for _ in range(5):
        if obs.done:
            break
        obs = env.step(DarkStoreAction(action="wait"))

    # Also try stockout picks to accumulate negative reward
    obs_check = env.step(DarkStoreAction(
        action="move_picker", row=2, col=4
    ))
    if not obs_check.done:
        for _ in range(5):
            obs_check = env.step(DarkStoreAction(
                action="move_picker", row=2, col=4
            ))
            if obs_check.picker_position == (2, 4) or obs_check.done:
                break

    if not obs_check.done and obs_check.picker_position == (2, 4):
        # coke at (2,4) has stock=0 in full_operations
        obs_check = env.step(DarkStoreAction(
            action="pick", item_name="coke", row=2, col=4
        ))

    score = env.compute_score()

    if obs_check.cumulative_reward < 0:
        assert score == 0.0, (
            f"Grader should clamp negative reward to 0.0, got {score} "
            f"(cumulative_reward={obs_check.cumulative_reward})"
        )
    else:
        # Even if not negative, score should still be in [0, 1]
        assert 0.0 <= score <= 1.0
