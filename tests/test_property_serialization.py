# Feature: dark-store-simulator, Property 17: Observation Serialization Round-Trip
"""
Property-based test for DarkStoreObservation serialization round-trip.

**Validates: Requirements 9.2**

For any DarkStoreObservation, serializing to JSON via .model_dump_json()
and deserializing back via DarkStoreObservation.model_validate_json()
should produce an equivalent observation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from typing import List, Tuple

from my_env.models import (
    CompletedDeliveryInfo,
    DarkStoreObservation,
    DeliveryInfo,
    OrderInfo,
    RiderInfo,
    ShelfInfo,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for nested models
# ---------------------------------------------------------------------------

# Valid item names from the design document
ITEM_NAMES = [
    "milk", "bread", "dal", "curd", "chips", "oil",
    "juice", "coke", "curd2", "eggs", "biscuits", "juice2",
    "butter", "rice", "chips2",
]

item_name_st = st.sampled_from(ITEM_NAMES)

rider_statuses = st.sampled_from(["idle", "delivering", "returning"])


# Dark store grid: 10 rows x 8 cols
dark_store_pos_st = st.tuples(
    st.integers(min_value=0, max_value=9),
    st.integers(min_value=0, max_value=7),
)

# City grid: 8 rows x 8 cols
city_pos_st = st.tuples(
    st.integers(min_value=0, max_value=7),
    st.integers(min_value=0, max_value=7),
)


@st.composite
def shelf_info_st(draw):
    return ShelfInfo(
        item_name=draw(item_name_st),
        row=draw(st.integers(min_value=0, max_value=9)),
        col=draw(st.integers(min_value=0, max_value=7)),
        stock=draw(st.integers(min_value=0, max_value=50)),
        expiry_ticks=draw(st.integers(min_value=-1, max_value=100)),
    )


@st.composite
def order_info_st(draw):
    items = draw(st.lists(item_name_st, min_size=1, max_size=5))
    picked = draw(st.lists(item_name_st, min_size=0, max_size=len(items)))
    return OrderInfo(
        order_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)),
        items=items,
        picked_items=picked,
        timer_ticks=draw(st.integers(min_value=-5, max_value=40)),
        customer_position=draw(city_pos_st),
    )


@st.composite
def rider_info_st(draw):
    return RiderInfo(
        rider_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)),
        position=draw(city_pos_st),
        status=draw(rider_statuses),
        eta_ticks=draw(st.integers(min_value=0, max_value=30)),
    )


@st.composite
def delivery_info_st(draw):
    return DeliveryInfo(
        order_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)),
        customer_position=draw(city_pos_st),
        timer_ticks=draw(st.integers(min_value=-5, max_value=40)),
        rider_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)),
    )


@st.composite
def completed_delivery_info_st(draw):
    return CompletedDeliveryInfo(
        order_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)),
        on_time=draw(st.booleans()),
    )


@st.composite
def dark_store_observation_st(draw):
    """Generate a random valid DarkStoreObservation with all nested models."""
    tick = draw(st.integers(min_value=0, max_value=100))
    budget = draw(st.integers(min_value=tick, max_value=200))
    ticks_remaining = budget - tick

    return DarkStoreObservation(
        # Inherited from Observation base
        done=draw(st.booleans()),
        reward=draw(st.one_of(st.none(), st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))),
        metadata=draw(st.fixed_dictionaries({}, optional={"info": st.text(min_size=0, max_size=20)})),
        # DarkStoreObservation fields
        tick=tick,
        ticks_remaining=ticks_remaining,
        picker_position=draw(dark_store_pos_st),
        picker_holding=draw(st.lists(item_name_st, min_size=0, max_size=5)),
        shelves=draw(st.lists(shelf_info_st(), min_size=0, max_size=15)),
        pending_orders=draw(st.lists(order_info_st(), min_size=0, max_size=10)),
        packed_orders=draw(st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12),
            min_size=0, max_size=5,
        )),
        riders=draw(st.lists(rider_info_st(), min_size=0, max_size=3)),
        active_deliveries=draw(st.lists(delivery_info_st(), min_size=0, max_size=5)),
        completed_deliveries=draw(st.lists(completed_delivery_info_st(), min_size=0, max_size=10)),
        cumulative_reward=draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False)),
        error=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        text=draw(st.text(min_size=0, max_size=200)),
    )


# ---------------------------------------------------------------------------
# Property 17: Observation Serialization Round-Trip
# ---------------------------------------------------------------------------


@given(obs=dark_store_observation_st())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_observation_serialization_round_trip(obs: DarkStoreObservation):
    """
    **Validates: Requirements 9.2**

    For any DarkStoreObservation, serializing to JSON via .model_dump_json()
    and deserializing back via DarkStoreObservation.model_validate_json()
    should produce an equivalent observation.
    """
    # Serialize to JSON
    json_str = obs.model_dump_json()

    # Deserialize back
    restored = DarkStoreObservation.model_validate_json(json_str)

    # The round-tripped observation must be equivalent
    assert restored.tick == obs.tick
    assert restored.ticks_remaining == obs.ticks_remaining
    assert restored.picker_position == obs.picker_position
    assert restored.picker_holding == obs.picker_holding
    assert restored.done == obs.done
    assert restored.error == obs.error
    assert restored.text == obs.text
    assert restored.packed_orders == obs.packed_orders

    # Compare cumulative_reward with float tolerance
    assert abs(restored.cumulative_reward - obs.cumulative_reward) < 1e-9

    # Compare reward (can be None, bool, int, or float)
    if obs.reward is None:
        assert restored.reward is None
    elif isinstance(obs.reward, float):
        assert abs(restored.reward - obs.reward) < 1e-9
    else:
        assert restored.reward == obs.reward

    # Compare nested lists
    assert len(restored.shelves) == len(obs.shelves)
    for r_shelf, o_shelf in zip(restored.shelves, obs.shelves):
        assert r_shelf == o_shelf

    assert len(restored.pending_orders) == len(obs.pending_orders)
    for r_order, o_order in zip(restored.pending_orders, obs.pending_orders):
        assert r_order == o_order

    assert len(restored.riders) == len(obs.riders)
    for r_rider, o_rider in zip(restored.riders, obs.riders):
        assert r_rider == o_rider

    assert len(restored.active_deliveries) == len(obs.active_deliveries)
    for r_del, o_del in zip(restored.active_deliveries, obs.active_deliveries):
        assert r_del == o_del

    assert len(restored.completed_deliveries) == len(obs.completed_deliveries)
    for r_cd, o_cd in zip(restored.completed_deliveries, obs.completed_deliveries):
        assert r_cd == o_cd

    # Full model equality check via model_dump
    assert restored.model_dump() == obs.model_dump()
