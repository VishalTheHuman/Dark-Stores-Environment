# Feature: dark-store-simulator, Property 2: Tick Invariant
"""
Property-based test for DarkStoreEnvironment tick invariant.

**Validates: Requirements 2.2, 4.1**

For any observation returned by step() or reset(), tick + ticks_remaining
should equal the task's tick budget, and each call to step() should
increment tick by exactly 1.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment, TASK_REGISTRY
from my_env.models import DarkStoreAction

# Task names and their tick budgets
TASK_NAMES = ["single_order", "concurrent_orders", "full_operations"]
TASK_BUDGETS = {
    "single_order": 20,
    "concurrent_orders": 30,
    "full_operations": 40,
}

task_name_st = st.sampled_from(TASK_NAMES)

# Valid action types the agent can send (simple ones that don't need complex setup)
SIMPLE_ACTIONS = [
    DarkStoreAction(action="wait"),
    DarkStoreAction(action="move_picker", row=0, col=0),
    DarkStoreAction(action="move_picker", row=1, col=1),
    DarkStoreAction(action="move_picker", row=2, col=2),
    DarkStoreAction(action="move_picker", row=3, col=3),
    DarkStoreAction(action="move_picker", row=0, col=7),
]

simple_action_st = st.sampled_from(SIMPLE_ACTIONS)


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_tick_invariant_on_reset(task_name: str):
    """
    **Validates: Requirements 2.2**

    On reset(), tick should be 0 and tick + ticks_remaining should equal
    the task's tick budget.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    budget = TASK_BUDGETS[task_name]

    assert obs.tick == 0, f"Reset tick should be 0, got {obs.tick}"
    assert obs.ticks_remaining == budget, (
        f"Reset ticks_remaining should be {budget}, got {obs.ticks_remaining}"
    )
    assert obs.tick + obs.ticks_remaining == budget, (
        f"tick + ticks_remaining should equal {budget}, "
        f"got {obs.tick} + {obs.ticks_remaining} = {obs.tick + obs.ticks_remaining}"
    )


@given(
    task_name=task_name_st,
    actions=st.lists(simple_action_st, min_size=1, max_size=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_tick_invariant_after_steps(task_name: str, actions):
    """
    **Validates: Requirements 2.2, 4.1**

    After each step(), tick should increment by exactly 1 and
    tick + ticks_remaining should equal the task's tick budget.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    budget = TASK_BUDGETS[task_name]

    # Verify invariant on reset
    assert obs.tick + obs.ticks_remaining == budget

    prev_tick = obs.tick

    for action in actions:
        if obs.done:
            break

        obs = env.step(action)

        # Tick should have incremented by exactly 1
        assert obs.tick == prev_tick + 1, (
            f"Tick should increment by 1: was {prev_tick}, now {obs.tick}"
        )

        # tick + ticks_remaining should always equal budget
        assert obs.tick + obs.ticks_remaining == budget, (
            f"tick + ticks_remaining should equal {budget}, "
            f"got {obs.tick} + {obs.ticks_remaining} = {obs.tick + obs.ticks_remaining}"
        )

        prev_tick = obs.tick


@given(
    task_name=task_name_st,
    actions=st.lists(simple_action_st, min_size=1, max_size=45),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_tick_reaches_budget(task_name: str, actions):
    """
    **Validates: Requirements 2.2, 4.1**

    When enough steps are taken, tick should reach the budget and
    ticks_remaining should be 0. The invariant holds at every step.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    budget = TASK_BUDGETS[task_name]

    for action in actions:
        if obs.done:
            break

        obs = env.step(action)

        # Invariant must hold at every step
        assert obs.tick + obs.ticks_remaining == budget, (
            f"tick + ticks_remaining should equal {budget}, "
            f"got {obs.tick} + {obs.ticks_remaining} = {obs.tick + obs.ticks_remaining}"
        )

    # If we ran enough steps to exhaust the budget, verify final state
    if obs.tick >= budget:
        assert obs.ticks_remaining == 0, (
            f"At budget, ticks_remaining should be 0, got {obs.ticks_remaining}"
        )
        assert obs.done, "Episode should be done when tick reaches budget"
