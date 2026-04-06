"""
Shared fixtures and Hypothesis strategies for Dark Store Simulator tests.

Provides:
- Fresh DarkStoreEnvironment instances
- Reset-to-task fixtures for single_order, concurrent_orders, full_operations
- Hypothesis strategies for valid and invalid DarkStoreAction instances
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import strategies as st

from my_env.server.dark_store_environment import (
    DarkStoreEnvironment,
    SHELF_LAYOUT,
    TASK_REGISTRY,
)
from my_env.models import DarkStoreAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Create a fresh DarkStoreEnvironment instance."""
    return DarkStoreEnvironment()


@pytest.fixture
def single_order_env(env):
    """Environment reset to the single_order task."""
    obs = env.reset(task="single_order")
    return env, obs


@pytest.fixture
def concurrent_orders_env(env):
    """Environment reset to the concurrent_orders task."""
    obs = env.reset(task="concurrent_orders")
    return env, obs


@pytest.fixture
def full_operations_env(env):
    """Environment reset to the full_operations task."""
    obs = env.reset(task="full_operations")
    return env, obs


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# All shelf positions from the layout
_SHELF_POSITIONS = [(s["row"], s["col"]) for s in SHELF_LAYOUT]
_ITEM_NAMES = list({s["item_name"] for s in SHELF_LAYOUT})
_RIDER_IDS = ["Rider-A", "Rider-B", "Rider-C"]
_ORDER_IDS = [f"Order-{i}" for i in range(1, 11)]

VALID_ACTION_TYPES = [
    "move_picker", "pick", "pack", "assign_rider",
    "batch_delivery", "restock", "wait",
]


def valid_move_picker_st():
    """Strategy for valid move_picker actions."""
    return st.builds(
        DarkStoreAction,
        action=st.just("move_picker"),
        row=st.integers(min_value=0, max_value=9),
        col=st.integers(min_value=0, max_value=7),
    )


def valid_pick_st():
    """Strategy for valid pick actions targeting real shelf positions."""
    shelf_idx = st.integers(min_value=0, max_value=len(SHELF_LAYOUT) - 1)
    return shelf_idx.map(lambda i: DarkStoreAction(
        action="pick",
        item_name=SHELF_LAYOUT[i]["item_name"],
        row=SHELF_LAYOUT[i]["row"],
        col=SHELF_LAYOUT[i]["col"],
    ))


def valid_pack_st():
    """Strategy for pack actions with plausible order IDs."""
    return st.builds(
        DarkStoreAction,
        action=st.just("pack"),
        order_id=st.sampled_from(_ORDER_IDS),
    )


def valid_assign_rider_st():
    """Strategy for assign_rider actions."""
    return st.builds(
        DarkStoreAction,
        action=st.just("assign_rider"),
        order_id=st.sampled_from(_ORDER_IDS),
        rider_id=st.sampled_from(_RIDER_IDS),
    )


def valid_batch_delivery_st():
    """Strategy for batch_delivery actions."""
    return st.builds(
        DarkStoreAction,
        action=st.just("batch_delivery"),
        order_a=st.sampled_from(_ORDER_IDS[:5]),
        order_b=st.sampled_from(_ORDER_IDS[5:]),
        rider_id=st.sampled_from(_RIDER_IDS),
    )


def valid_restock_st():
    """Strategy for restock actions."""
    return st.builds(
        DarkStoreAction,
        action=st.just("restock"),
        item_name=st.sampled_from(_ITEM_NAMES),
        quantity=st.integers(min_value=1, max_value=10),
    )


def valid_wait_st():
    """Strategy for wait actions."""
    return st.just(DarkStoreAction(action="wait"))


def valid_action_st():
    """Strategy that generates any valid DarkStoreAction."""
    return st.one_of(
        valid_move_picker_st(),
        valid_pick_st(),
        valid_pack_st(),
        valid_assign_rider_st(),
        valid_batch_delivery_st(),
        valid_restock_st(),
        valid_wait_st(),
    )


def invalid_action_st():
    """Strategy that generates invalid/malformed DarkStoreAction instances."""
    return st.one_of(
        # Unrecognized action type
        st.builds(
            DarkStoreAction,
            action=st.sampled_from(["fly", "teleport", "noop", "explode", ""]),
        ),
        # pick with missing fields
        st.builds(
            DarkStoreAction,
            action=st.just("pick"),
            item_name=st.none(),
            row=st.none(),
            col=st.none(),
        ),
        # pack with no order_id
        st.builds(
            DarkStoreAction,
            action=st.just("pack"),
            order_id=st.none(),
        ),
        # assign_rider with missing fields
        st.builds(
            DarkStoreAction,
            action=st.just("assign_rider"),
            order_id=st.none(),
            rider_id=st.none(),
        ),
        # move_picker with missing coords
        st.builds(
            DarkStoreAction,
            action=st.just("move_picker"),
            row=st.none(),
            col=st.none(),
        ),
    )
