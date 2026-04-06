"""
Task configuration tests for DarkStoreEnvironment.

Tests cover:
- single_order config: 1 order, 20 ticks, 1 rider, all stock available
- concurrent_orders config: 5 orders in waves, 30 ticks, 3 riders
- full_operations config: 10 orders, 40 ticks, 2+1 riders, coke/juice at 0, bread expiry at tick 3
- Same seed produces identical initial states
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from my_env.server.dark_store_environment import (
    DarkStoreEnvironment,
    TASK_REGISTRY,
    SHELF_LAYOUT,
)
from my_env.models import DarkStoreAction


# ---------------------------------------------------------------------------
# single_order config
# ---------------------------------------------------------------------------

class TestSingleOrderConfig:
    """Verify single_order task configuration."""

    def test_config_exists(self):
        assert "single_order" in TASK_REGISTRY

    def test_tick_budget(self):
        cfg = TASK_REGISTRY["single_order"]
        assert cfg.tick_budget == 20

    def test_seed(self):
        cfg = TASK_REGISTRY["single_order"]
        assert cfg.seed == 42

    def test_one_order_at_tick_zero(self):
        cfg = TASK_REGISTRY["single_order"]
        # Should have exactly 1 order schedule entry at tick 0
        assert len(cfg.order_schedule) == 1
        tick, orders = cfg.order_schedule[0]
        assert tick == 0
        assert len(orders) == 1
        assert orders[0].order_id == "Order-1"
        assert set(orders[0].items) == {"milk", "chips", "eggs"}

    def test_one_rider(self):
        cfg = TASK_REGISTRY["single_order"]
        assert len(cfg.rider_specs) == 1
        assert cfg.rider_specs[0].rider_id == "Rider-A"
        assert cfg.rider_specs[0].status == "idle"

    def test_all_stock_available(self):
        """All shelves should have stock > 0 (no stock overrides to 0)."""
        env = DarkStoreEnvironment()
        obs = env.reset(task="single_order")
        for shelf in obs.shelves:
            assert shelf.stock > 0, (
                f"{shelf.item_name} at ({shelf.row},{shelf.col}) has stock=0"
            )

    def test_initial_observation(self):
        env = DarkStoreEnvironment()
        obs = env.reset(task="single_order")
        assert obs.tick == 0
        assert obs.ticks_remaining == 20
        assert obs.picker_position == (0, 7)
        assert len(obs.pending_orders) == 1
        assert len(obs.riders) == 1
        assert obs.riders[0].status == "idle"
        assert obs.done is False


# ---------------------------------------------------------------------------
# concurrent_orders config
# ---------------------------------------------------------------------------

class TestConcurrentOrdersConfig:
    """Verify concurrent_orders task configuration."""

    def test_config_exists(self):
        assert "concurrent_orders" in TASK_REGISTRY

    def test_tick_budget(self):
        cfg = TASK_REGISTRY["concurrent_orders"]
        assert cfg.tick_budget == 30

    def test_seed(self):
        cfg = TASK_REGISTRY["concurrent_orders"]
        assert cfg.seed == 123

    def test_five_orders_in_waves(self):
        cfg = TASK_REGISTRY["concurrent_orders"]
        total_orders = sum(len(orders) for _, orders in cfg.order_schedule)
        assert total_orders == 5

        # Wave structure: 2 at tick 0, 1 at tick 4, 2 at tick 10
        schedule_ticks = [(tick, len(orders)) for tick, orders in cfg.order_schedule]
        assert schedule_ticks == [(0, 2), (4, 1), (10, 2)]

    def test_three_riders(self):
        cfg = TASK_REGISTRY["concurrent_orders"]
        assert len(cfg.rider_specs) == 3
        for rs in cfg.rider_specs:
            assert rs.status == "idle"

    def test_initial_observation(self):
        env = DarkStoreEnvironment()
        obs = env.reset(task="concurrent_orders")
        assert obs.tick == 0
        assert obs.ticks_remaining == 30
        # Only tick-0 orders should be present initially
        assert len(obs.pending_orders) == 2
        assert len(obs.riders) == 3

    def test_orders_arrive_in_waves(self):
        """Orders should appear at their scheduled ticks."""
        env = DarkStoreEnvironment()
        obs = env.reset(task="concurrent_orders")

        # Tick 0: 2 orders
        assert len(obs.pending_orders) == 2

        # Advance to tick 4: 1 more order
        for _ in range(4):
            obs = env.step(DarkStoreAction(action="wait"))

        pending_ids = {o.order_id for o in obs.pending_orders}
        assert "Order-3" in pending_ids

        # Advance to tick 10: 2 more orders
        for _ in range(6):
            obs = env.step(DarkStoreAction(action="wait"))

        pending_ids = {o.order_id for o in obs.pending_orders}
        assert "Order-4" in pending_ids
        assert "Order-5" in pending_ids


# ---------------------------------------------------------------------------
# full_operations config
# ---------------------------------------------------------------------------

class TestFullOperationsConfig:
    """Verify full_operations task configuration."""

    def test_config_exists(self):
        assert "full_operations" in TASK_REGISTRY

    def test_tick_budget(self):
        cfg = TASK_REGISTRY["full_operations"]
        assert cfg.tick_budget == 40

    def test_seed(self):
        cfg = TASK_REGISTRY["full_operations"]
        assert cfg.seed == 456

    def test_ten_orders(self):
        cfg = TASK_REGISTRY["full_operations"]
        total_orders = sum(len(orders) for _, orders in cfg.order_schedule)
        assert total_orders == 10

    def test_riders_config(self):
        """2 idle riders + 1 returning rider."""
        cfg = TASK_REGISTRY["full_operations"]
        assert len(cfg.rider_specs) == 3

        idle_riders = [r for r in cfg.rider_specs if r.status == "idle"]
        returning_riders = [r for r in cfg.rider_specs if r.status == "returning"]
        assert len(idle_riders) == 2
        assert len(returning_riders) == 1

    def test_coke_and_juice_at_zero_stock(self):
        """coke and juice should start at stock=0."""
        cfg = TASK_REGISTRY["full_operations"]
        assert cfg.stock_overrides.get("coke") == 0
        assert cfg.stock_overrides.get("juice") == 0

        env = DarkStoreEnvironment()
        obs = env.reset(task="full_operations")

        coke_shelves = [s for s in obs.shelves if s.item_name == "coke"]
        juice_shelves = [s for s in obs.shelves if s.item_name == "juice"]

        for s in coke_shelves:
            assert s.stock == 0, f"coke at ({s.row},{s.col}) should have stock=0"
        for s in juice_shelves:
            assert s.stock == 0, f"juice at ({s.row},{s.col}) should have stock=0"

    def test_bread_expiry_at_tick_3(self):
        """bread should expire at tick 3."""
        cfg = TASK_REGISTRY["full_operations"]
        assert cfg.expiry_overrides.get("bread") == 3

        env = DarkStoreEnvironment()
        obs = env.reset(task="full_operations")

        bread_shelves = [s for s in obs.shelves if s.item_name == "bread"]
        for s in bread_shelves:
            assert s.expiry_ticks == 3, (
                f"bread at ({s.row},{s.col}) should have expiry=3, got {s.expiry_ticks}"
            )

    def test_returning_rider_initial_state(self):
        """Rider-C should start as returning from (5,5)."""
        env = DarkStoreEnvironment()
        obs = env.reset(task="full_operations")

        rider_c = [r for r in obs.riders if r.rider_id == "Rider-C"]
        assert len(rider_c) == 1
        assert rider_c[0].status == "returning"
        assert rider_c[0].position == (5, 5)

    def test_initial_observation(self):
        env = DarkStoreEnvironment()
        obs = env.reset(task="full_operations")
        assert obs.tick == 0
        assert obs.ticks_remaining == 40
        # Tick 0 has 2 orders
        assert len(obs.pending_orders) == 2
        assert len(obs.riders) == 3


# ---------------------------------------------------------------------------
# Deterministic initial states
# ---------------------------------------------------------------------------

class TestDeterministicStates:
    """Verify same seed produces identical initial states."""

    def test_same_seed_identical_states_single_order(self):
        env1 = DarkStoreEnvironment()
        env2 = DarkStoreEnvironment()
        obs1 = env1.reset(task="single_order")
        obs2 = env2.reset(task="single_order")
        assert obs1.model_dump() == obs2.model_dump()

    def test_same_seed_identical_states_concurrent_orders(self):
        env1 = DarkStoreEnvironment()
        env2 = DarkStoreEnvironment()
        obs1 = env1.reset(task="concurrent_orders")
        obs2 = env2.reset(task="concurrent_orders")
        assert obs1.model_dump() == obs2.model_dump()

    def test_same_seed_identical_states_full_operations(self):
        env1 = DarkStoreEnvironment()
        env2 = DarkStoreEnvironment()
        obs1 = env1.reset(task="full_operations")
        obs2 = env2.reset(task="full_operations")
        assert obs1.model_dump() == obs2.model_dump()

    def test_reset_twice_same_env(self):
        """Resetting the same env twice produces identical states."""
        env = DarkStoreEnvironment()
        for task_name in ["single_order", "concurrent_orders", "full_operations"]:
            obs1 = env.reset(task=task_name)
            obs2 = env.reset(task=task_name)
            assert obs1.model_dump() == obs2.model_dump(), (
                f"Task {task_name}: reset twice should produce identical states"
            )
