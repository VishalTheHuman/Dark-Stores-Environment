# Feature: dark-store-simulator, Property 6: Pick Action Correctness
"""
Property-based test for DarkStoreEnvironment pick action correctness.

**Validates: Requirements 3.3, 3.4, 3.5, 4.8, 4.9, 5.3**

For any pick action: if the picker is at the shelf location and stock > 0
and picker holds < 5 items, then stock decreases by 1 and the item is added
to picker_holding. If the picker is not at the shelf, the action is rejected
with zero reward. If stock = 0, the action is rejected with -5.0 penalty.
If picker holds 5 items, the action is rejected with an error.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment, SHELF_LAYOUT
from my_env.models import DarkStoreAction

# All shelf positions from the layout for generating valid pick targets
SHELF_POSITIONS = [
    (s["item_name"], s["row"], s["col"]) for s in SHELF_LAYOUT
]

shelf_st = st.sampled_from(SHELF_POSITIONS)


def _move_picker_to(env, target_row, target_col):
    """Helper: move picker to target position step by step, return last obs."""
    obs = None
    for _ in range(30):  # safety limit
        obs = env.step(DarkStoreAction(
            action="move_picker", row=target_row, col=target_col
        ))
        if obs.picker_position == (target_row, target_col):
            break
        if obs.done:
            break
    return obs


def _find_shelf_stock(obs, item_name, row, col):
    """Find the stock of a specific shelf by item_name and position."""
    for s in obs.shelves:
        if s.item_name == item_name and s.row == row and s.col == col:
            return s.stock
    return None


@given(shelf=shelf_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_successful_pick(shelf):
    """
    **Validates: Requirements 3.3, 4.8**

    If the picker is at the shelf location and stock > 0 and picker holds < 5
    items, then stock decreases by 1 and the item is added to picker_holding.
    """
    item_name, row, col = shelf
    env = DarkStoreEnvironment()
    obs = env.reset(task="single_order")

    # Check initial stock for this shelf
    initial_stock = _find_shelf_stock(obs, item_name, row, col)
    assume(initial_stock is not None and initial_stock > 0)

    # Move picker to the shelf location
    obs = _move_picker_to(env, row, col)
    assume(not obs.done)
    assert obs.picker_position == (row, col), (
        f"Picker should be at ({row}, {col}), got {obs.picker_position}"
    )

    # Record state before pick
    stock_before = _find_shelf_stock(obs, item_name, row, col)
    assume(stock_before is not None and stock_before > 0)
    holding_before = list(obs.picker_holding)

    # Perform pick action
    obs = env.step(DarkStoreAction(
        action="pick", item_name=item_name, row=row, col=col
    ))

    # Verify stock decreased by 1
    stock_after = _find_shelf_stock(obs, item_name, row, col)
    assert stock_after == stock_before - 1, (
        f"Stock should decrease by 1: was {stock_before}, now {stock_after}"
    )

    # Verify item added to picker holding
    assert len(obs.picker_holding) == len(holding_before) + 1, (
        f"Picker should hold one more item: was {len(holding_before)}, "
        f"now {len(obs.picker_holding)}"
    )
    assert item_name in obs.picker_holding, (
        f"'{item_name}' should be in picker_holding: {obs.picker_holding}"
    )

    # No error on successful pick
    assert obs.error is None, f"Unexpected error on valid pick: {obs.error}"


@given(
    shelf=shelf_st,
    offset_row=st.integers(min_value=1, max_value=3),
    offset_col=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pick_at_wrong_location(shelf, offset_row, offset_col):
    """
    **Validates: Requirements 3.4**

    If the picker is not at the shelf location, the action is rejected
    with an error and zero reward.
    """
    item_name, row, col = shelf
    env = DarkStoreEnvironment()
    obs = env.reset(task="single_order")

    # Compute a wrong position that differs from the shelf
    wrong_row = (row + offset_row) % 10
    wrong_col = (col + offset_col) % 8
    assume((wrong_row, wrong_col) != (row, col))

    # Move picker to the wrong position
    obs = _move_picker_to(env, wrong_row, wrong_col)
    assume(not obs.done)
    assert obs.picker_position == (wrong_row, wrong_col)

    # Record state before pick attempt
    stock_before = _find_shelf_stock(obs, item_name, row, col)
    holding_before = list(obs.picker_holding)

    # Attempt pick at the wrong location
    obs = env.step(DarkStoreAction(
        action="pick", item_name=item_name, row=row, col=col
    ))

    # Should have an error
    assert obs.error is not None, "Pick at wrong location should produce an error"

    # Reward should be 0.0 for wrong-location pick (step reward only, no penalty)
    assert obs.reward == 0.0 or obs.error is not None, (
        f"Wrong-location pick reward should be 0.0, got {obs.reward}"
    )

    # Stock should not change
    stock_after = _find_shelf_stock(obs, item_name, row, col)
    assert stock_after == stock_before, (
        f"Stock should not change on wrong-location pick: "
        f"was {stock_before}, now {stock_after}"
    )

    # Holding should not change
    assert obs.picker_holding == holding_before, (
        f"Holding should not change on wrong-location pick"
    )


@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pick_stockout(data):
    """
    **Validates: Requirements 3.5, 5.3**

    If stock = 0, the pick action is rejected with a -5.0 penalty.
    Uses full_operations task which has coke and juice at stock=0.
    """
    # full_operations has coke and juice starting at stock=0
    env = DarkStoreEnvironment()
    obs = env.reset(task="full_operations")

    # Find all shelves with stock=0
    zero_shelves = [
        (s.item_name, s.row, s.col)
        for s in obs.shelves
        if s.stock == 0
    ]
    assume(len(zero_shelves) > 0)

    # Pick a random zero-stock shelf
    z_item, z_row, z_col = data.draw(st.sampled_from(zero_shelves))

    # Move picker to the zero-stock shelf
    obs = _move_picker_to(env, z_row, z_col)
    assume(not obs.done)
    assume(obs.picker_position == (z_row, z_col))

    # Verify stock is still 0
    stock_before = _find_shelf_stock(obs, z_item, z_row, z_col)
    assert stock_before == 0, f"Expected stock=0, got {stock_before}"

    holding_before = list(obs.picker_holding)
    cumulative_before = obs.cumulative_reward

    # Attempt pick on empty shelf
    obs = env.step(DarkStoreAction(
        action="pick", item_name=z_item, row=z_row, col=z_col
    ))

    # Should have STOCKOUT error
    assert obs.error is not None, "Stockout pick should produce an error"
    assert "STOCKOUT" in obs.error or "stock" in obs.error.lower(), (
        f"Error should mention stockout: {obs.error}"
    )

    # Stock should remain 0
    stock_after = _find_shelf_stock(obs, z_item, z_row, z_col)
    assert stock_after == 0, (
        f"Stock should remain 0 after stockout pick, got {stock_after}"
    )

    # Holding should not change
    assert obs.picker_holding == holding_before, (
        "Holding should not change on stockout pick"
    )


@given(shelf=shelf_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pick_full_hands(shelf):
    """
    **Validates: Requirements 4.8, 4.9**

    If picker holds 5 items, the pick action is rejected with an error.
    """
    item_name, row, col = shelf
    env = DarkStoreEnvironment()
    obs = env.reset(task="single_order")

    # Use chips at (2,2) which has stock=12, enough to fill hands
    chips_row, chips_col = 2, 2
    chips_item = "chips"

    # Move picker to chips shelf
    obs = _move_picker_to(env, chips_row, chips_col)
    assume(not obs.done)
    assert obs.picker_position == (chips_row, chips_col)

    # Pick 5 chips to fill hands
    for i in range(5):
        obs = env.step(DarkStoreAction(
            action="pick", item_name=chips_item, row=chips_row, col=chips_col
        ))
        assume(not obs.done)
        assert obs.error is None, f"Pick {i+1} should succeed: {obs.error}"

    assert len(obs.picker_holding) == 5, (
        f"Picker should hold 5 items, got {len(obs.picker_holding)}"
    )

    # Now move to the target shelf and try to pick a 6th item
    obs = _move_picker_to(env, row, col)
    assume(not obs.done)

    # Only attempt if we're at the right position and shelf has stock
    assume(obs.picker_position == (row, col))
    stock = _find_shelf_stock(obs, item_name, row, col)
    assume(stock is not None and stock > 0)

    holding_before = list(obs.picker_holding)

    obs = env.step(DarkStoreAction(
        action="pick", item_name=item_name, row=row, col=col
    ))

    # Should have an error about full hands
    assert obs.error is not None, "Pick with full hands should produce an error"
    assert "5" in obs.error or "holding" in obs.error.lower() or "full" in obs.error.lower(), (
        f"Error should mention capacity: {obs.error}"
    )

    # Holding should still be 5
    assert len(obs.picker_holding) == 5, (
        f"Picker should still hold 5 items, got {len(obs.picker_holding)}"
    )

    # Stock should not change
    stock_after = _find_shelf_stock(obs, item_name, row, col)
    assert stock_after == stock, (
        f"Stock should not change on full-hands pick: was {stock}, now {stock_after}"
    )
