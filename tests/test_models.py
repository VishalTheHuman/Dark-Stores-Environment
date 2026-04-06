"""
Serialization tests for Dark Store data models.

Tests cover:
- DarkStoreAction serialization/deserialization
- DarkStoreObservation serialization/deserialization
- All helper models: ShelfInfo, OrderInfo, RiderInfo, DeliveryInfo, CompletedDeliveryInfo
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from my_env.models import (
    DarkStoreAction,
    DarkStoreObservation,
    ShelfInfo,
    OrderInfo,
    RiderInfo,
    DeliveryInfo,
    CompletedDeliveryInfo,
)


# ---------------------------------------------------------------------------
# DarkStoreAction serialization
# ---------------------------------------------------------------------------

class TestDarkStoreActionSerialization:
    """Test DarkStoreAction serialization and deserialization."""

    def test_wait_action_roundtrip(self):
        action = DarkStoreAction(action="wait")
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "wait"

    def test_move_picker_roundtrip(self):
        action = DarkStoreAction(action="move_picker", row=3, col=5)
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "move_picker"
        assert restored.row == 3
        assert restored.col == 5

    def test_pick_action_roundtrip(self):
        action = DarkStoreAction(action="pick", item_name="milk", row=1, col=1)
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "pick"
        assert restored.item_name == "milk"
        assert restored.row == 1
        assert restored.col == 1

    def test_pack_action_roundtrip(self):
        action = DarkStoreAction(action="pack", order_id="Order-1")
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "pack"
        assert restored.order_id == "Order-1"

    def test_assign_rider_roundtrip(self):
        action = DarkStoreAction(
            action="assign_rider", order_id="Order-1", rider_id="Rider-A"
        )
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "assign_rider"
        assert restored.order_id == "Order-1"
        assert restored.rider_id == "Rider-A"

    def test_batch_delivery_roundtrip(self):
        action = DarkStoreAction(
            action="batch_delivery",
            order_a="Order-1",
            order_b="Order-2",
            rider_id="Rider-A",
        )
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "batch_delivery"
        assert restored.order_a == "Order-1"
        assert restored.order_b == "Order-2"
        assert restored.rider_id == "Rider-A"

    def test_restock_action_roundtrip(self):
        action = DarkStoreAction(
            action="restock", item_name="coke", quantity=5
        )
        json_str = action.model_dump_json()
        restored = DarkStoreAction.model_validate_json(json_str)
        assert restored.action == "restock"
        assert restored.item_name == "coke"
        assert restored.quantity == 5

    def test_action_model_dump_dict(self):
        action = DarkStoreAction(action="pick", item_name="eggs", row=1, col=5)
        d = action.model_dump()
        assert d["action"] == "pick"
        assert d["item_name"] == "eggs"
        assert d["row"] == 1
        assert d["col"] == 5


# ---------------------------------------------------------------------------
# DarkStoreObservation serialization
# ---------------------------------------------------------------------------

class TestDarkStoreObservationSerialization:
    """Test DarkStoreObservation serialization and deserialization."""

    def _make_sample_observation(self):
        """Create a sample observation with all fields populated."""
        return DarkStoreObservation(
            tick=5,
            ticks_remaining=15,
            picker_position=(2, 3),
            picker_holding=["milk", "chips"],
            shelves=[
                ShelfInfo(item_name="milk", row=1, col=1, stock=7, expiry_ticks=45),
                ShelfInfo(item_name="chips", row=2, col=2, stock=11, expiry_ticks=-1),
            ],
            pending_orders=[
                OrderInfo(
                    order_id="Order-1",
                    items=["milk", "chips", "eggs"],
                    picked_items=["milk"],
                    timer_ticks=15,
                    customer_position=(3, 1),
                ),
            ],
            packed_orders=["Order-2"],
            riders=[
                RiderInfo(rider_id="Rider-A", position=(0, 0), status="idle", eta_ticks=0),
                RiderInfo(rider_id="Rider-B", position=(3, 2), status="delivering", eta_ticks=4),
            ],
            active_deliveries=[
                DeliveryInfo(
                    order_id="Order-3",
                    customer_position=(5, 5),
                    timer_ticks=10,
                    rider_id="Rider-B",
                ),
            ],
            completed_deliveries=[
                CompletedDeliveryInfo(order_id="Order-4", on_time=True),
            ],
            cumulative_reward=12.5,
            done=False,
            reward=2.0,
            error=None,
            text="=== DARK STORE ===",
        )

    def test_observation_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert restored.tick == obs.tick
        assert restored.ticks_remaining == obs.ticks_remaining
        assert restored.picker_position == obs.picker_position
        assert restored.picker_holding == obs.picker_holding
        assert restored.packed_orders == obs.packed_orders
        assert restored.cumulative_reward == obs.cumulative_reward
        assert restored.done == obs.done
        assert restored.reward == obs.reward
        assert restored.error == obs.error
        assert restored.text == obs.text

    def test_observation_shelves_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert len(restored.shelves) == len(obs.shelves)
        for s1, s2 in zip(obs.shelves, restored.shelves):
            assert s1.item_name == s2.item_name
            assert s1.row == s2.row
            assert s1.col == s2.col
            assert s1.stock == s2.stock
            assert s1.expiry_ticks == s2.expiry_ticks

    def test_observation_orders_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert len(restored.pending_orders) == len(obs.pending_orders)
        o1 = obs.pending_orders[0]
        o2 = restored.pending_orders[0]
        assert o1.order_id == o2.order_id
        assert o1.items == o2.items
        assert o1.picked_items == o2.picked_items
        assert o1.timer_ticks == o2.timer_ticks
        assert o1.customer_position == o2.customer_position

    def test_observation_riders_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert len(restored.riders) == len(obs.riders)
        for r1, r2 in zip(obs.riders, restored.riders):
            assert r1.rider_id == r2.rider_id
            assert r1.position == r2.position
            assert r1.status == r2.status
            assert r1.eta_ticks == r2.eta_ticks

    def test_observation_deliveries_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert len(restored.active_deliveries) == len(obs.active_deliveries)
        d1 = obs.active_deliveries[0]
        d2 = restored.active_deliveries[0]
        assert d1.order_id == d2.order_id
        assert d1.customer_position == d2.customer_position
        assert d1.timer_ticks == d2.timer_ticks
        assert d1.rider_id == d2.rider_id

    def test_observation_completed_roundtrip(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)

        assert len(restored.completed_deliveries) == len(obs.completed_deliveries)
        c1 = obs.completed_deliveries[0]
        c2 = restored.completed_deliveries[0]
        assert c1.order_id == c2.order_id
        assert c1.on_time == c2.on_time

    def test_observation_with_error(self):
        obs = DarkStoreObservation(
            tick=1,
            ticks_remaining=19,
            picker_position=(0, 7),
            picker_holding=[],
            shelves=[],
            pending_orders=[],
            packed_orders=[],
            riders=[],
            active_deliveries=[],
            completed_deliveries=[],
            cumulative_reward=0.0,
            done=False,
            reward=0.0,
            error="Invalid action type 'fly'",
            text="test",
        )
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)
        assert restored.error == "Invalid action type 'fly'"

    def test_observation_full_model_dump_equality(self):
        obs = self._make_sample_observation()
        json_str = obs.model_dump_json()
        restored = DarkStoreObservation.model_validate_json(json_str)
        assert obs.model_dump() == restored.model_dump()


# ---------------------------------------------------------------------------
# Helper model serialization
# ---------------------------------------------------------------------------

class TestShelfInfoSerialization:
    def test_roundtrip(self):
        shelf = ShelfInfo(item_name="milk", row=1, col=1, stock=8, expiry_ticks=50)
        json_str = shelf.model_dump_json()
        restored = ShelfInfo.model_validate_json(json_str)
        assert shelf.model_dump() == restored.model_dump()

    def test_non_perishable(self):
        shelf = ShelfInfo(item_name="chips", row=2, col=2, stock=12, expiry_ticks=-1)
        json_str = shelf.model_dump_json()
        restored = ShelfInfo.model_validate_json(json_str)
        assert restored.expiry_ticks == -1


class TestOrderInfoSerialization:
    def test_roundtrip(self):
        order = OrderInfo(
            order_id="Order-1",
            items=["milk", "chips", "eggs"],
            picked_items=["milk"],
            timer_ticks=15,
            customer_position=(3, 1),
        )
        json_str = order.model_dump_json()
        restored = OrderInfo.model_validate_json(json_str)
        assert order.model_dump() == restored.model_dump()


class TestRiderInfoSerialization:
    def test_roundtrip(self):
        rider = RiderInfo(
            rider_id="Rider-A", position=(0, 0), status="idle", eta_ticks=0
        )
        json_str = rider.model_dump_json()
        restored = RiderInfo.model_validate_json(json_str)
        assert rider.model_dump() == restored.model_dump()

    def test_delivering_rider(self):
        rider = RiderInfo(
            rider_id="Rider-B", position=(3, 2), status="delivering", eta_ticks=5
        )
        json_str = rider.model_dump_json()
        restored = RiderInfo.model_validate_json(json_str)
        assert restored.status == "delivering"
        assert restored.eta_ticks == 5


class TestDeliveryInfoSerialization:
    def test_roundtrip(self):
        delivery = DeliveryInfo(
            order_id="Order-1",
            customer_position=(5, 5),
            timer_ticks=10,
            rider_id="Rider-A",
        )
        json_str = delivery.model_dump_json()
        restored = DeliveryInfo.model_validate_json(json_str)
        assert delivery.model_dump() == restored.model_dump()


class TestCompletedDeliveryInfoSerialization:
    def test_on_time_roundtrip(self):
        completed = CompletedDeliveryInfo(order_id="Order-1", on_time=True)
        json_str = completed.model_dump_json()
        restored = CompletedDeliveryInfo.model_validate_json(json_str)
        assert restored.on_time is True

    def test_late_roundtrip(self):
        completed = CompletedDeliveryInfo(order_id="Order-2", on_time=False)
        json_str = completed.model_dump_json()
        restored = CompletedDeliveryInfo.model_validate_json(json_str)
        assert restored.on_time is False
