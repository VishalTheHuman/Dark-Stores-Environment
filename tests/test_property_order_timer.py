# Feature: dark-store-simulator, Property 12: Order Timer Decrement
"""
Property-based test for DarkStoreEnvironment order timer decrement.

**Validates: Requirements 4.2**

For any step, all pending orders' timer_ticks should decrease by exactly 1
compared to the previous observation. Orders that transition out of pending
status (e.g., become delivered) during a step won't appear in pending_orders
anymore, so only compare orders present in both observations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment
from my_env.models import DarkStoreAction

TASK_NAMES = ["single_order", "concurrent_orders", "full_operations"]

task_name_st = st.sampled_from(TASK_NAMES)

# Simple actions that don't require complex preconditions
SIMPLE_ACTIONS = [
    DarkStoreAction(action="wait"),
    DarkStoreAction(action="move_picker", row=0, col=0),
    DarkStoreAction(action="move_picker", row=1, col=1),
    DarkStoreAction(action="move_picker", row=2, col=2),
    DarkStoreAction(action="move_picker", row=3, col=3),
    DarkStoreAction(action="move_picker", row=0, col=7),
]

simple_action_st = st.sampled_from(SIMPLE_ACTIONS)


def _pending_timer_map(obs):
    """Build a dict of {order_id: timer_ticks} from pending_orders."""
    return {o.order_id: o.timer_ticks for o in obs.pending_orders}


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_order_timer_decrements_on_first_step(task_name: str):
    """
    **Validates: Requirements 4.2**

    After a single wait step from reset, all orders that remain pending
    should have their timer_ticks decreased by exactly 1.
    """
    env = DarkStoreEnvironment()
    obs_before = env.reset(task=task_name)

    if not obs_before.pending_orders:
        return  # No orders to check

    timers_before = _pending_timer_map(obs_before)

    obs_after = env.step(DarkStoreAction(action="wait"))

    timers_after = _pending_timer_map(obs_after)

    # Check orders present in both observations
    common_ids = set(timers_before.keys()) & set(timers_after.keys())
    for order_id in common_ids:
        expected = timers_before[order_id] - 1
        actual = timers_after[order_id]
        assert actual == expected, (
            f"Order {order_id} timer should be {expected} after 1 step, "
            f"got {actual} (was {timers_before[order_id]})"
        )


@given(
    task_name=task_name_st,
    actions=st.lists(simple_action_st, min_size=1, max_size=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_order_timer_decrements_every_step(task_name: str, actions):
    """
    **Validates: Requirements 4.2**

    At every step, all pending orders that remain pending should have
    their timer_ticks decreased by exactly 1 compared to the previous
    observation.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    for action in actions:
        if obs.done:
            break

        timers_before = _pending_timer_map(obs)

        obs = env.step(action)

        timers_after = _pending_timer_map(obs)

        # Only compare orders present in both snapshots
        common_ids = set(timers_before.keys()) & set(timers_after.keys())
        for order_id in common_ids:
            expected = timers_before[order_id] - 1
            actual = timers_after[order_id]
            assert actual == expected, (
                f"Order {order_id} timer should be {expected}, "
                f"got {actual} (was {timers_before[order_id]}) "
                f"at tick {obs.tick}"
            )


@given(task_name=st.sampled_from(["concurrent_orders", "full_operations"]))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_order_timer_decrements_across_multiple_steps(task_name: str):
    """
    **Validates: Requirements 4.2**

    Over N consecutive wait steps, an order that stays pending the entire
    time should have its timer decreased by exactly N.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    if not obs.pending_orders:
        return

    initial_timers = _pending_timer_map(obs)
    steps_taken = 0

    for _ in range(10):
        if obs.done:
            break
        obs = env.step(DarkStoreAction(action="wait"))
        steps_taken += 1

    final_timers = _pending_timer_map(obs)

    # Orders present from start to finish should have timer decreased by steps_taken
    common_ids = set(initial_timers.keys()) & set(final_timers.keys())
    for order_id in common_ids:
        expected = initial_timers[order_id] - steps_taken
        actual = final_timers[order_id]
        assert actual == expected, (
            f"Order {order_id} timer should be {expected} after {steps_taken} steps, "
            f"got {actual} (started at {initial_timers[order_id]})"
        )
