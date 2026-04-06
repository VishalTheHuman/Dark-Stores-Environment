# Feature: dark-store-simulator, Property 18: Text Urgency Markers
"""
Property-based test for DarkStoreEnvironment text urgency markers.

**Validates: Requirements 11.3**

For any observation where a pending order has timer < 5, text should contain
"URGENT". For any shelf with stock = 0, text should contain "STOCKOUT".
For any perishable shelf with expiry < 3, text should contain "EXPIRING".

Uses full_operations task which has stock=0 shelves (coke, juice) and bread
expiry at tick 3.
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

simple_action_st = st.one_of(
    st.just(DarkStoreAction(action="wait")),
    st.builds(
        DarkStoreAction,
        action=st.just("move_picker"),
        row=st.integers(min_value=0, max_value=9),
        col=st.integers(min_value=0, max_value=7),
    ),
)


def test_stockout_marker_on_reset():
    """
    **Validates: Requirements 11.3**

    full_operations has coke and juice at stock=0 on reset.
    The text should contain "STOCKOUT" immediately.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    # Verify there are shelves with stock=0
    zero_stock_shelves = [s for s in obs.shelves if s.stock == 0]
    assert len(zero_stock_shelves) > 0, (
        "full_operations should have shelves with stock=0"
    )

    assert "STOCKOUT" in obs.text, (
        f"Text should contain 'STOCKOUT' when shelves have stock=0. "
        f"Zero-stock shelves: {[(s.item_name, s.row, s.col) for s in zero_stock_shelves]}"
    )


def test_expiring_marker_near_expiry():
    """
    **Validates: Requirements 11.3**

    full_operations has bread expiry at tick 3. After waiting 1 tick,
    bread expiry should be 2 (< 3), so text should contain "EXPIRING".
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    # bread starts with expiry=3 in full_operations
    bread_shelves = [
        s for s in obs.shelves
        if s.item_name == "bread"
    ]
    assert len(bread_shelves) > 0, "Should have bread shelves"

    # Wait 1 tick — bread expiry goes from 3 to 2 (< 3 threshold)
    obs = env.step(DarkStoreAction(action="wait"))

    # Check bread expiry is now < 3
    bread_shelves_after = [
        s for s in obs.shelves
        if s.item_name == "bread" and s.expiry_ticks > 0 and s.expiry_ticks < 3
    ]

    if len(bread_shelves_after) > 0:
        assert "EXPIRING" in obs.text, (
            f"Text should contain 'EXPIRING' when perishable shelf has expiry < 3. "
            f"Bread expiry: {[(s.item_name, s.expiry_ticks) for s in bread_shelves_after]}"
        )


def test_urgent_marker_when_timer_low():
    """
    **Validates: Requirements 11.3**

    Wait enough ticks so that a pending order's timer drops below 5,
    then verify text contains "URGENT".
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="single_order")

    # single_order: Order-1 timer=20, budget=20
    # Wait 16 ticks so timer becomes 20 - 16 = 4 (< 5)
    for _ in range(16):
        if obs.done:
            break
        obs = env.step(DarkStoreAction(action="wait"))

    if obs.done:
        return

    # Check if any pending order has timer < 5
    urgent_orders = [
        o for o in obs.pending_orders
        if o.timer_ticks < 5
    ]

    if len(urgent_orders) > 0:
        assert "URGENT" in obs.text, (
            f"Text should contain 'URGENT' when pending order timer < 5. "
            f"Urgent orders: {[(o.order_id, o.timer_ticks) for o in urgent_orders]}"
        )


@given(
    actions=st.lists(simple_action_st, min_size=1, max_size=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_stockout_marker_consistency(actions):
    """
    **Validates: Requirements 11.3**

    For any action sequence on full_operations, whenever any shelf has
    stock=0, the text should contain "STOCKOUT".
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    # Check on reset
    has_zero_stock = any(s.stock == 0 for s in obs.shelves)
    if has_zero_stock:
        assert "STOCKOUT" in obs.text, (
            "Text should contain 'STOCKOUT' when any shelf has stock=0"
        )

    for action in actions:
        if obs.done:
            break
        obs = env.step(action)

        has_zero_stock = any(s.stock == 0 for s in obs.shelves)
        if has_zero_stock:
            assert "STOCKOUT" in obs.text, (
                f"Text should contain 'STOCKOUT' at tick {obs.tick} "
                f"when shelves have stock=0"
            )


@given(
    actions=st.lists(simple_action_st, min_size=1, max_size=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_urgent_marker_consistency(actions):
    """
    **Validates: Requirements 11.3**

    For any action sequence, whenever a pending order has timer < 5,
    the text should contain "URGENT".
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    for action in actions:
        if obs.done:
            break
        obs = env.step(action)

        has_urgent = any(o.timer_ticks < 5 for o in obs.pending_orders)
        if has_urgent:
            assert "URGENT" in obs.text, (
                f"Text should contain 'URGENT' at tick {obs.tick} "
                f"when pending orders have timer < 5: "
                f"{[(o.order_id, o.timer_ticks) for o in obs.pending_orders if o.timer_ticks < 5]}"
            )


@given(
    actions=st.lists(simple_action_st, min_size=1, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_expiring_marker_consistency(actions):
    """
    **Validates: Requirements 11.3**

    For any action sequence on full_operations, whenever a perishable shelf
    has expiry > 0 and expiry < 3, the text should contain "EXPIRING".
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    for action in actions:
        if obs.done:
            break
        obs = env.step(action)

        has_expiring = any(
            s.expiry_ticks > 0 and s.expiry_ticks < 3 and s.stock > 0
            for s in obs.shelves
        )
        if has_expiring:
            assert "EXPIRING" in obs.text, (
                f"Text should contain 'EXPIRING' at tick {obs.tick} "
                f"when perishable shelves have expiry < 3: "
                f"{[(s.item_name, s.expiry_ticks, s.stock) for s in obs.shelves if s.expiry_ticks > 0 and s.expiry_ticks < 3 and s.stock > 0]}"
            )
