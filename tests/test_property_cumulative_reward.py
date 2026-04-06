# Feature: dark-store-simulator, Property 4: Cumulative Reward Consistency
"""
Property-based test for DarkStoreEnvironment cumulative reward consistency.

**Validates: Requirements 2.10, 5.8**

For any sequence of actions, the cumulative_reward in the observation should
equal the sum of all individual step reward values returned throughout the
episode.
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

# Strategy for generating random actions that exercise different reward paths
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
        item_name=st.sampled_from(["milk", "bread", "chips", "eggs", "coke", "juice"]),
        row=st.integers(min_value=0, max_value=9),
        col=st.integers(min_value=0, max_value=7),
    ),
    st.builds(
        DarkStoreAction,
        action=st.just("restock"),
        item_name=st.sampled_from(["milk", "bread", "chips", "coke", "juice"]),
        quantity=st.integers(min_value=1, max_value=5),
    ),
)


@given(
    task_name=task_name_st,
    actions=st.lists(action_st, min_size=1, max_size=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_cumulative_reward_equals_sum_of_step_rewards(task_name, actions):
    """
    **Validates: Requirements 2.10, 5.8**

    For any sequence of actions, cumulative_reward should equal the sum of
    all individual step reward values.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    reward_sum = 0.0

    for action in actions:
        if obs.done:
            break

        obs = env.step(action)
        reward_sum += obs.reward

    # Compare with tolerance for floating point
    assert abs(obs.cumulative_reward - round(reward_sum, 4)) < 1e-3, (
        f"cumulative_reward ({obs.cumulative_reward}) should equal "
        f"sum of step rewards ({round(reward_sum, 4)})"
    )


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_cumulative_reward_starts_at_zero(task_name):
    """
    **Validates: Requirements 2.10**

    On reset, cumulative_reward should be 0.0.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    assert obs.cumulative_reward == 0.0, (
        f"cumulative_reward on reset should be 0.0, got {obs.cumulative_reward}"
    )


@given(
    task_name=task_name_st,
    num_waits=st.integers(min_value=1, max_value=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_cumulative_reward_consistent_with_wait_only(task_name, num_waits):
    """
    **Validates: Requirements 2.10, 5.8**

    Wait actions should produce 0.0 step reward (unless tick events like
    expiry fire). The cumulative should still match the sum.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    reward_sum = 0.0

    for _ in range(num_waits):
        if obs.done:
            break
        obs = env.step(DarkStoreAction(action="wait"))
        reward_sum += obs.reward

    assert abs(obs.cumulative_reward - round(reward_sum, 4)) < 1e-3, (
        f"cumulative_reward ({obs.cumulative_reward}) should equal "
        f"sum of step rewards ({round(reward_sum, 4)}) after wait-only sequence"
    )
