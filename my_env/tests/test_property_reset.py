# Feature: dark-store-simulator, Property 1: Reset Determinism
"""
Property-based test for DarkStoreEnvironment reset determinism.

**Validates: Requirements 1.5, 6.5**

For any task identifier, calling reset(task) twice should produce
byte-identical observations — same grid layout, same shelf stocks,
same order schedules, same rider positions, same tick counter (0).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment

# All three valid task names from the task registry
TASK_NAMES = ["single_order", "concurrent_orders", "full_operations"]

task_name_st = st.sampled_from(TASK_NAMES)


@given(task_name=task_name_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_reset_determinism(task_name: str):
    """
    **Validates: Requirements 1.5, 6.5**

    For any task identifier, calling reset(task) twice should produce
    byte-identical observations — same grid layout, same shelf stocks,
    same order schedules, same rider positions, same tick counter (0).
    """
    env = DarkStoreEnvironment()

    obs1 = env.reset(task=task_name)
    obs2 = env.reset(task=task_name)

    # Tick counter must be 0 on both resets
    assert obs1.tick == 0, f"First reset tick should be 0, got {obs1.tick}"
    assert obs2.tick == 0, f"Second reset tick should be 0, got {obs2.tick}"

    # Field-by-field comparison
    assert obs1.tick == obs2.tick, "tick mismatch"
    assert obs1.ticks_remaining == obs2.ticks_remaining, "ticks_remaining mismatch"
    assert obs1.picker_position == obs2.picker_position, "picker_position mismatch"
    assert obs1.picker_holding == obs2.picker_holding, "picker_holding mismatch"
    assert obs1.done == obs2.done, "done mismatch"
    assert obs1.reward == obs2.reward, "reward mismatch"
    assert obs1.cumulative_reward == obs2.cumulative_reward, "cumulative_reward mismatch"
    assert obs1.error == obs2.error, "error mismatch"
    assert obs1.packed_orders == obs2.packed_orders, "packed_orders mismatch"

    # Shelves: same layout, same stocks, same expiry
    assert len(obs1.shelves) == len(obs2.shelves), "shelves count mismatch"
    for s1, s2 in zip(obs1.shelves, obs2.shelves):
        assert s1.item_name == s2.item_name, f"shelf item_name mismatch: {s1.item_name} vs {s2.item_name}"
        assert s1.row == s2.row, f"shelf row mismatch for {s1.item_name}"
        assert s1.col == s2.col, f"shelf col mismatch for {s1.item_name}"
        assert s1.stock == s2.stock, f"shelf stock mismatch for {s1.item_name}"
        assert s1.expiry_ticks == s2.expiry_ticks, f"shelf expiry mismatch for {s1.item_name}"

    # Pending orders: same schedule, same items, same positions
    assert len(obs1.pending_orders) == len(obs2.pending_orders), "pending_orders count mismatch"
    for o1, o2 in zip(obs1.pending_orders, obs2.pending_orders):
        assert o1.order_id == o2.order_id, f"order_id mismatch: {o1.order_id} vs {o2.order_id}"
        assert o1.items == o2.items, f"order items mismatch for {o1.order_id}"
        assert o1.picked_items == o2.picked_items, f"order picked_items mismatch for {o1.order_id}"
        assert o1.timer_ticks == o2.timer_ticks, f"order timer mismatch for {o1.order_id}"
        assert o1.customer_position == o2.customer_position, f"order customer_position mismatch for {o1.order_id}"

    # Riders: same positions, same statuses
    assert len(obs1.riders) == len(obs2.riders), "riders count mismatch"
    for r1, r2 in zip(obs1.riders, obs2.riders):
        assert r1.rider_id == r2.rider_id, f"rider_id mismatch: {r1.rider_id} vs {r2.rider_id}"
        assert r1.position == r2.position, f"rider position mismatch for {r1.rider_id}"
        assert r1.status == r2.status, f"rider status mismatch for {r1.rider_id}"
        assert r1.eta_ticks == r2.eta_ticks, f"rider eta mismatch for {r1.rider_id}"

    # Active deliveries and completed deliveries
    assert obs1.active_deliveries == obs2.active_deliveries, "active_deliveries mismatch"
    assert obs1.completed_deliveries == obs2.completed_deliveries, "completed_deliveries mismatch"

    # Full model equality via model_dump()
    dump1 = obs1.model_dump()
    dump2 = obs2.model_dump()
    assert dump1 == dump2, (
        f"model_dump() mismatch for task '{task_name}'. "
        f"Differences: {_diff_dicts(dump1, dump2)}"
    )


def _diff_dicts(d1: dict, d2: dict) -> dict:
    """Return keys where d1 and d2 differ (for debugging)."""
    diffs = {}
    all_keys = set(d1.keys()) | set(d2.keys())
    for k in all_keys:
        if d1.get(k) != d2.get(k):
            diffs[k] = {"obs1": d1.get(k), "obs2": d2.get(k)}
    return diffs
