"""
Dark Store Environment — Quick Commerce Simulator.

A complete OpenEnv-compliant environment that models a Zepto/Blinkit-style
dark store. The agent coordinates picking, packing, dispatching, batching,
and restocking across a 10x8 warehouse grid and an 8x8 city delivery grid.

Tasks:
  single_order      — 1 order, 20 ticks, easy
  concurrent_orders — 5 orders in waves, 30 ticks, medium
  full_operations   — 10 orders, stockouts, expiry, 40 ticks, hard
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CompletedDeliveryInfo,
        DarkStoreAction,
        DarkStoreObservation,
        DeliveryInfo,
        OrderInfo,
        RiderInfo,
        ShelfInfo,
    )
except ImportError:
    from models import (
        CompletedDeliveryInfo,
        DarkStoreAction,
        DarkStoreObservation,
        DeliveryInfo,
        OrderInfo,
        RiderInfo,
        ShelfInfo,
    )


# ---------------------------------------------------------------------------
# Internal state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Shelf:
    """Internal shelf state."""
    item_name: str
    row: int
    col: int
    stock: int
    expiry_ticks: int  # -1 = non-perishable


@dataclass
class Order:
    """Internal order state."""
    order_id: str
    items: List[str]
    picked_items: List[str]
    customer_pos: Tuple[int, int]
    timer_ticks: int
    status: str  # pending | packing | packed | delivering | delivered
    packing_ticks_left: int = 0
    is_batch: bool = False


@dataclass
class Rider:
    """Internal rider state."""
    rider_id: str
    position: Tuple[int, int]
    status: str  # idle | delivering | returning
    eta_ticks: int = 0
    delivering_orders: List[str] = field(default_factory=list)
    destination: Optional[Tuple[int, int]] = None
    return_destination: Tuple[int, int] = (0, 0)
    # For batch: list of (customer_pos, order_id) stops
    delivery_stops: List[Tuple[Tuple[int, int], str]] = field(default_factory=list)


@dataclass
class OrderSpec:
    """Specification for an order in a task config."""
    order_id: str
    items: List[str]
    customer_pos: Tuple[int, int]
    timer_ticks: int = 20  # default 10 min = 20 ticks


@dataclass
class RiderSpec:
    """Specification for a rider in a task config."""
    rider_id: str
    position: Tuple[int, int] = (0, 0)
    status: str = "idle"
    eta_ticks: int = 0
    returning_from: Optional[Tuple[int, int]] = None


@dataclass
class TaskConfig:
    """Full task configuration."""
    name: str
    seed: int
    tick_budget: int
    order_schedule: List[Tuple[int, List[OrderSpec]]]  # (tick, [orders])
    rider_specs: List[RiderSpec] = field(default_factory=list)
    stock_overrides: Dict[str, int] = field(default_factory=dict)
    expiry_overrides: Dict[str, int] = field(default_factory=dict)
    max_theoretical_reward: float = 0.0


# ---------------------------------------------------------------------------
# Shelf layout — 15 shelf positions, 12 unique SKU types
# ---------------------------------------------------------------------------

SHELF_LAYOUT: List[Dict[str, Any]] = [
    {"item_name": "milk",     "row": 1, "col": 1, "stock": 8,  "expiry_ticks": 50},
    {"item_name": "bread",    "row": 2, "col": 1, "stock": 6,  "expiry_ticks": 50},
    {"item_name": "dal",      "row": 3, "col": 1, "stock": 7,  "expiry_ticks": -1},
    {"item_name": "curd",     "row": 1, "col": 2, "stock": 5,  "expiry_ticks": 50},
    {"item_name": "chips",    "row": 2, "col": 2, "stock": 12, "expiry_ticks": -1},
    {"item_name": "oil",      "row": 3, "col": 2, "stock": 5,  "expiry_ticks": -1},
    {"item_name": "juice",    "row": 1, "col": 4, "stock": 8,  "expiry_ticks": 50},
    {"item_name": "coke",     "row": 2, "col": 4, "stock": 10, "expiry_ticks": -1},
    {"item_name": "curd",     "row": 3, "col": 4, "stock": 4,  "expiry_ticks": 50},
    {"item_name": "eggs",     "row": 1, "col": 5, "stock": 4,  "expiry_ticks": 50},
    {"item_name": "biscuits", "row": 2, "col": 5, "stock": 9,  "expiry_ticks": -1},
    {"item_name": "juice",    "row": 3, "col": 5, "stock": 8,  "expiry_ticks": 50},
    {"item_name": "butter",   "row": 1, "col": 7, "stock": 8,  "expiry_ticks": 50},
    {"item_name": "rice",     "row": 2, "col": 7, "stock": 3,  "expiry_ticks": -1},
    {"item_name": "chips",    "row": 3, "col": 7, "stock": 11, "expiry_ticks": -1},
]


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def _build_task_registry() -> Dict[str, TaskConfig]:
    """Build the three task configurations."""

    # --- Task 1: single_order (easy) ---
    single_order = TaskConfig(
        name="single_order",
        seed=42,
        tick_budget=40,
        order_schedule=[
            (0, [
                OrderSpec(
                    order_id="Order-1",
                    items=["milk", "chips", "eggs"],
                    customer_pos=(3, 1),
                    timer_ticks=40,
                ),
            ]),
        ],
        rider_specs=[
            RiderSpec(rider_id="Rider-A", position=(0, 0), status="idle"),
        ],
        # Max: +10.0 (on-time) - ~1.0 movement ~ 9.0
        max_theoretical_reward=9.0,
    )

    # --- Task 2: concurrent_orders (medium) ---
    concurrent_orders = TaskConfig(
        name="concurrent_orders",
        seed=123,
        tick_budget=60,
        order_schedule=[
            (0, [
                OrderSpec(
                    order_id="Order-1",
                    items=["milk", "bread", "curd"],
                    customer_pos=(3, 1),
                    timer_ticks=40,
                ),
                OrderSpec(
                    order_id="Order-2",
                    items=["chips", "eggs"],
                    customer_pos=(4, 2),
                    timer_ticks=40,
                ),
            ]),
            (8, [
                OrderSpec(
                    order_id="Order-3",
                    items=["rice", "dal"],
                    customer_pos=(6, 5),
                    timer_ticks=40,
                ),
            ]),
            (20, [
                OrderSpec(
                    order_id="Order-4",
                    items=["butter", "oil"],
                    customer_pos=(1, 6),
                    timer_ticks=40,
                ),
                OrderSpec(
                    order_id="Order-5",
                    items=["biscuits", "coke"],
                    customer_pos=(2, 7),
                    timer_ticks=40,
                ),
            ]),
        ],
        rider_specs=[
            RiderSpec(rider_id="Rider-A", position=(0, 0), status="idle"),
            RiderSpec(rider_id="Rider-B", position=(0, 0), status="idle"),
            RiderSpec(rider_id="Rider-C", position=(0, 0), status="idle"),
        ],
        # Max: 5x10.0 + 2x2.0 - movement ~ 50.0
        max_theoretical_reward=50.0,
    )

    # --- Task 3: full_operations (hard) ---
    full_operations = TaskConfig(
        name="full_operations",
        seed=456,
        tick_budget=80,
        order_schedule=[
            (0, [
                OrderSpec(order_id="Order-1", items=["milk", "bread", "eggs"],
                          customer_pos=(3, 1), timer_ticks=40),
                OrderSpec(order_id="Order-2", items=["chips", "curd"],
                          customer_pos=(4, 2), timer_ticks=40),
            ]),
            (5, [
                OrderSpec(order_id="Order-3", items=["rice", "dal"],
                          customer_pos=(2, 3), timer_ticks=40),
            ]),
            (10, [
                OrderSpec(order_id="Order-4", items=["butter", "oil"],
                          customer_pos=(1, 6), timer_ticks=40),
            ]),
            (15, [
                OrderSpec(order_id="Order-5", items=["biscuits", "coke"],
                          customer_pos=(2, 7), timer_ticks=40),
            ]),
            (20, [
                OrderSpec(order_id="Order-6", items=["juice", "milk"],
                          customer_pos=(5, 3), timer_ticks=40),
            ]),
            (30, [
                OrderSpec(order_id="Order-7", items=["bread", "eggs", "curd"],
                          customer_pos=(6, 5), timer_ticks=40),
            ]),
            (40, [
                OrderSpec(order_id="Order-8", items=["chips", "dal"],
                          customer_pos=(3, 6), timer_ticks=40),
            ]),
            (50, [
                OrderSpec(order_id="Order-9", items=["rice", "butter"],
                          customer_pos=(7, 2), timer_ticks=30),
            ]),
            (60, [
                OrderSpec(order_id="Order-10", items=["oil", "biscuits"],
                          customer_pos=(4, 7), timer_ticks=20),
            ]),
        ],
        rider_specs=[
            RiderSpec(rider_id="Rider-A", position=(0, 0), status="idle"),
            RiderSpec(rider_id="Rider-B", position=(0, 0), status="idle"),
            RiderSpec(rider_id="Rider-C", position=(5, 5), status="returning",
                      eta_ticks=6, returning_from=(5, 5)),
        ],
        stock_overrides={"coke": 0, "juice": 0},
        expiry_overrides={"bread": 3},
        # Max: 10x10.0 + 3x2.0 - movement - restock ~ 95.0
        max_theoretical_reward=95.0,
    )

    return {
        "single_order": single_order,
        "concurrent_orders": concurrent_orders,
        "full_operations": full_operations,
    }


TASK_REGISTRY: Dict[str, TaskConfig] = _build_task_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_one_step(
    current: Tuple[int, int], target: Tuple[int, int]
) -> Tuple[int, int]:
    """Move one cell toward *target* (Manhattan movement, row first)."""
    r, c = current
    tr, tc = target
    if r < tr:
        return (r + 1, c)
    elif r > tr:
        return (r - 1, c)
    elif c < tc:
        return (r, c + 1)
    elif c > tc:
        return (r, c - 1)
    return current  # already at target


def _is_walkable(row: int, col: int) -> bool:
    """Check if a cell is walkable in the dark store grid (10 cols x 8 rows).

    Walkable cells:
    - Walkway columns: col 0 and col 9 (full-length corridors)
    - Top corridor: row 0 (all columns)
    - Bottom corridor: row 7 (all columns)
    - Shelf cells: any cell that has a shelf (picker can stand there to pick)
    """
    if row < 0 or row > 7 or col < 0 or col > 9:
        return False
    # Walkway columns
    if col == 0 or col == 9:
        return True
    # Top and bottom corridors
    if row == 0 or row == 7:
        return True
    # Shelf cells are walkable (picker goes there to pick items)
    shelf_positions = {(s["row"], s["col"]) for s in SHELF_LAYOUT}
    if (row, col) in shelf_positions:
        return True
    return False


def _bfs_path_cost(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """BFS shortest path cost through walkable cells in the dark store.

    Returns the number of steps in the shortest path, or Manhattan distance
    as fallback if no path found (shouldn't happen with proper grid).
    """
    if start == end:
        return 0

    from collections import deque
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) == end:
                return dist + 1
            if (nr, nc) not in visited and _is_walkable(nr, nc):
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))

    # Fallback — should not happen if grid is connected
    return _manhattan(start, end)


# ---------------------------------------------------------------------------
# DarkStoreEnvironment
# ---------------------------------------------------------------------------

class DarkStoreEnvironment(Environment):
    """OpenEnv-compliant Dark Store simulation environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    VALID_ACTIONS = frozenset([
        "move_picker", "pick", "pack", "assign_rider",
        "batch_delivery", "restock", "wait",
    ])

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "single_order"
        self._task_config: Optional[TaskConfig] = None
        self._rng: random.Random = random.Random(42)

        # Simulation state (populated on reset)
        self._tick: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0

        self._picker_pos: Tuple[int, int] = (0, 7)
        self._picker_holding: List[str] = []

        self._shelves: List[Shelf] = []
        self._orders: Dict[str, Order] = {}
        self._riders: Dict[str, Rider] = {}
        self._completed: List[CompletedDeliveryInfo] = []
        self._packed_orders: List[str] = []

        # Restock queue: list of (arrival_tick, item_name, quantity)
        self._restock_queue: List[Tuple[int, str, int]] = []

        # Order schedule remaining (populated on reset)
        self._order_schedule: List[Tuple[int, List[OrderSpec]]] = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> DarkStoreObservation:
        """Reset the environment for a new episode.

        Args:
            seed: ignored (task seed is used for determinism)
            episode_id: optional episode identifier
            task: task name — one of single_order, concurrent_orders, full_operations
        """
        task_name = task or kwargs.get("task_name", "single_order")
        if task_name not in TASK_REGISTRY:
            task_name = "single_order"

        self._task_name = task_name
        self._task_config = TASK_REGISTRY[task_name]
        cfg = self._task_config

        self._rng = random.Random(cfg.seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
        )

        self._tick = 0
        self._done = False
        self._cumulative_reward = 0.0

        # Picker starts at (0, 7)
        self._picker_pos = (0, 7)
        self._picker_holding = []

        # Shelves — copy from layout, apply overrides
        self._shelves = []
        for s in SHELF_LAYOUT:
            expiry = s["expiry_ticks"]
            stock = s["stock"]
            name = s["item_name"]
            if name in cfg.stock_overrides:
                stock = cfg.stock_overrides[name]
            if name in cfg.expiry_overrides:
                expiry = cfg.expiry_overrides[name]
            self._shelves.append(Shelf(
                item_name=name,
                row=s["row"],
                col=s["col"],
                stock=stock,
                expiry_ticks=expiry,
            ))

        # Riders
        self._riders = {}
        for rs in cfg.rider_specs:
            rider = Rider(
                rider_id=rs.rider_id,
                position=rs.position,
                status=rs.status,
                eta_ticks=rs.eta_ticks,
            )
            if rs.status == "returning" and rs.returning_from:
                rider.destination = (0, 0)
                rider.position = rs.returning_from
            self._riders[rs.rider_id] = rider

        # Orders
        self._orders = {}
        self._packed_orders = []
        self._completed = []
        self._restock_queue = []
        self._order_schedule = []

        for tick, order_specs in cfg.order_schedule:
            self._order_schedule.append((tick, list(order_specs)))

        # Spawn tick-0 orders immediately
        self._spawn_scheduled_orders()

        return self._build_observation(reward=0.0, error=None)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: DarkStoreAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DarkStoreObservation:
        """Execute one action and advance the simulation by one tick."""
        try:
            return self._step_inner(action)
        except Exception as exc:
            # Catch-all: environment NEVER crashes
            self._tick += 1
            self._state.step_count += 1
            if self._task_config and self._tick >= self._task_config.tick_budget:
                self._done = True
            return self._build_observation(
                reward=0.0,
                error=f"Internal error: {exc}",
            )

    def _step_inner(self, action: DarkStoreAction) -> DarkStoreObservation:
        """Core step logic, wrapped by step() for safety."""

        # Episode already ended?
        if self._done:
            return self._build_observation(reward=0.0, error="Episode has ended")

        cfg = self._task_config
        assert cfg is not None

        step_reward = 0.0
        error: Optional[str] = None

        action_type = getattr(action, "action", None)
        if not action_type or action_type not in self.VALID_ACTIONS:
            error = (
                f"Invalid action type '{action_type}'. "
                f"Valid types: {', '.join(sorted(self.VALID_ACTIONS))}"
            )
            self._advance_tick()
            self._cumulative_reward += -0.5
            return self._build_observation(reward=-0.5, error=error)

        # --- Dispatch to action handlers ---
        if action_type == "move_picker":
            step_reward, error = self._handle_move_picker(action)
        elif action_type == "pick":
            step_reward, error = self._handle_pick(action)
        elif action_type == "pack":
            step_reward, error = self._handle_pack(action)
        elif action_type == "assign_rider":
            step_reward, error = self._handle_assign_rider(action)
        elif action_type == "batch_delivery":
            step_reward, error = self._handle_batch_delivery(action)
        elif action_type == "restock":
            step_reward, error = self._handle_restock(action)
        elif action_type == "wait":
            step_reward, error = 0.0, None

        # Penalize invalid/failed actions (except stockout which already has -5.0)
        if error is not None and step_reward == 0.0:
            step_reward = -0.5

        # --- Advance simulation ---
        tick_reward = self._advance_tick()
        step_reward += tick_reward

        self._cumulative_reward += step_reward
        return self._build_observation(reward=step_reward, error=error)

    # ------------------------------------------------------------------
    # Action handlers — each returns (reward, error_or_None)
    # ------------------------------------------------------------------

    def _handle_move_picker(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Move picker to target via walkable path. Costs -0.05 per BFS step."""
        target_row = action.row
        target_col = action.col
        if target_row is None or target_col is None:
            return 0.0, "move_picker requires row and col"

        target = (int(target_row), int(target_col))
        if self._picker_pos == target:
            return 0.0, "Already at target — try pick or another action"

        if not _is_walkable(target[0], target[1]):
            return 0.0, (
                f"Target ({target[0]},{target[1]}) is not walkable. "
                f"Picker can move to: walkways (col 0,9), corridors (row 0,7), or shelf cells."
            )

        path_cost = _bfs_path_cost(self._picker_pos, target)
        self._picker_pos = target
        return -0.05 * path_cost, None

    def _handle_pick(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Pick an item from a shelf."""
        item_name = action.item_name
        row = action.row
        col = action.col

        if item_name is None or row is None or col is None:
            return 0.0, "pick requires item_name, row, and col"

        target = (int(row), int(col))

        # Picker must be at the shelf
        if self._picker_pos != target:
            return 0.0, (
                f"Picker at {self._picker_pos} but shelf is at {target}"
            )

        # Holding capacity
        if len(self._picker_holding) >= 5:
            return 0.0, "Picker already holding 5 items"

        # Find the shelf at this exact position with this item
        shelf = self._find_shelf(item_name, target)
        if shelf is None:
            return 0.0, f"No shelf for '{item_name}' at {target}"

        # Stockout check
        if shelf.stock <= 0:
            return -5.0, f"STOCKOUT: {item_name} has 0 stock"

        # Check if any pending order actually needs this item
        item_needed = False
        for order in self._orders.values():
            if order.status == "pending" and item_name in order.items:
                needed = order.items.count(item_name)
                have = order.picked_items.count(item_name)
                if have < needed:
                    item_needed = True
                    break
        if not item_needed:
            return 0.0, f"No pending order needs {item_name} right now"

        # Success — pick the item
        shelf.stock -= 1
        self._picker_holding.append(item_name)

        # Mark item as picked in any pending order that needs it
        for order in self._orders.values():
            if order.status == "pending" and item_name in order.items:
                needed = order.items.count(item_name)
                have = order.picked_items.count(item_name)
                if have < needed:
                    order.picked_items.append(item_name)
                    break  # only credit one order per pick

        return 0.0, None

    def _handle_pack(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Pack an order at the packing station."""
        order_id = action.order_id
        if order_id is None:
            return 0.0, "pack requires order_id"

        # Picker must be at packing station (0, 0)
        if self._picker_pos != (0, 0):
            return 0.0, "Picker must be at packing station (0,0)"

        order = self._orders.get(order_id)
        if order is None:
            return 0.0, f"Order {order_id} not found"

        if order.status != "pending":
            return 0.0, f"Order {order_id} is {order.status}, not pending"

        # Check all items picked
        missing = []
        for item in set(order.items):
            needed = order.items.count(item)
            have = order.picked_items.count(item)
            if have < needed:
                missing.append(item)

        if missing:
            return 0.0, f"Order {order_id} missing items: {', '.join(missing)}"

        # Start packing — takes 2 ticks
        order.status = "packing"
        order.packing_ticks_left = 2

        # Remove packed items from picker holding
        for item in order.items:
            if item in self._picker_holding:
                self._picker_holding.remove(item)

        return 0.0, None

    def _handle_assign_rider(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Assign an idle rider to deliver a packed order."""
        order_id = action.order_id
        rider_id = action.rider_id

        if order_id is None or rider_id is None:
            return 0.0, "assign_rider requires order_id and rider_id"

        order = self._orders.get(order_id)
        if order is None:
            return 0.0, f"Order {order_id} not found"
        if order.status != "packed":
            return 0.0, f"Order {order_id} is {order.status}, not packed"

        rider = self._riders.get(rider_id)
        if rider is None:
            return 0.0, f"Rider {rider_id} not found"
        if rider.status != "idle":
            return 0.0, f"Rider {rider_id} is {rider.status}, not idle"

        # Dispatch rider
        order.status = "delivering"
        if order_id in self._packed_orders:
            self._packed_orders.remove(order_id)

        rider.status = "delivering"
        rider.delivering_orders = [order_id]
        rider.destination = order.customer_pos
        rider.delivery_stops = [(order.customer_pos, order_id)]

        return 0.0, None

    def _handle_batch_delivery(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Batch two packed orders onto one rider."""
        order_a_id = action.order_a
        order_b_id = action.order_b
        rider_id = action.rider_id

        if order_a_id is None or order_b_id is None or rider_id is None:
            return 0.0, "batch_delivery requires order_a, order_b, and rider_id"

        order_a = self._orders.get(order_a_id)
        order_b = self._orders.get(order_b_id)

        if order_a is None:
            return 0.0, f"Order {order_a_id} not found"
        if order_b is None:
            return 0.0, f"Order {order_b_id} not found"
        if order_a.status != "packed":
            return 0.0, f"Order {order_a_id} is {order_a.status}, not packed"
        if order_b.status != "packed":
            return 0.0, f"Order {order_b_id} is {order_b.status}, not packed"

        rider = self._riders.get(rider_id)
        if rider is None:
            return 0.0, f"Rider {rider_id} not found"
        if rider.status != "idle":
            return 0.0, f"Rider {rider_id} is {rider.status}, not idle"

        # Determine delivery order: nearer customer first (from rider position)
        dist_a = _manhattan(rider.position, order_a.customer_pos)
        dist_b = _manhattan(rider.position, order_b.customer_pos)

        if dist_a <= dist_b:
            first, second = order_a, order_b
        else:
            first, second = order_b, order_a

        # Mark both as delivering
        first.status = "delivering"
        first.is_batch = True
        second.status = "delivering"
        second.is_batch = True

        for oid in [order_a_id, order_b_id]:
            if oid in self._packed_orders:
                self._packed_orders.remove(oid)

        rider.status = "delivering"
        rider.delivering_orders = [first.order_id, second.order_id]
        rider.destination = first.customer_pos
        rider.delivery_stops = [
            (first.customer_pos, first.order_id),
            (second.customer_pos, second.order_id),
        ]

        return 0.0, None

    def _handle_restock(
        self, action: DarkStoreAction
    ) -> Tuple[float, Optional[str]]:
        """Schedule a restock arrival in 4 ticks."""
        item_name = action.item_name
        quantity = action.quantity

        if item_name is None:
            return 0.0, "restock requires item_name"
        if quantity is None or quantity <= 0:
            quantity = 5  # default restock quantity

        # Arrives 4 ticks after the NEXT tick (since tick advances after action)
        arrival_tick = self._tick + 1 + 4
        self._restock_queue.append((arrival_tick, item_name, int(quantity)))
        return -1.0, None

    # ------------------------------------------------------------------
    # Tick advancement — simulation mechanics
    # ------------------------------------------------------------------

    def _advance_tick(self) -> float:
        """Advance the simulation by one tick. Returns reward from tick events."""
        self._tick += 1
        self._state.step_count += 1
        cfg = self._task_config
        assert cfg is not None

        tick_reward = 0.0

        # 1. Decrement order timers for non-delivered orders
        for order in self._orders.values():
            if order.status != "delivered":
                order.timer_ticks -= 1

        # 2. Process packing timers
        for order in self._orders.values():
            if order.status == "packing":
                order.packing_ticks_left -= 1
                if order.packing_ticks_left <= 0:
                    order.status = "packed"
                    self._packed_orders.append(order.order_id)

        # 3. Move delivering riders and handle deliveries
        for rider in self._riders.values():
            if rider.status == "delivering" and rider.delivery_stops:
                current_stop = rider.delivery_stops[0]
                dest = current_stop[0]
                rider.position = _move_one_step(rider.position, dest)

                # Check if arrived at current stop
                if rider.position == dest:
                    delivered_order_id = current_stop[1]
                    order = self._orders.get(delivered_order_id)
                    if order:
                        order.status = "delivered"
                        on_time = order.timer_ticks > 0
                        self._completed.append(CompletedDeliveryInfo(
                            order_id=delivered_order_id,
                            on_time=on_time,
                        ))
                        if on_time:
                            tick_reward += 10.0
                        else:
                            tick_reward += -15.0
                        # Batch bonus per order delivered in a batch
                        if order.is_batch:
                            tick_reward += 2.0

                    rider.delivery_stops.pop(0)
                    if delivered_order_id in rider.delivering_orders:
                        rider.delivering_orders.remove(delivered_order_id)

                    # More stops?
                    if rider.delivery_stops:
                        rider.destination = rider.delivery_stops[0][0]
                    else:
                        # Start returning
                        rider.status = "returning"
                        rider.destination = rider.return_destination
                        rider.delivering_orders = []

            elif rider.status == "returning":
                rider.position = _move_one_step(
                    rider.position, rider.return_destination
                )
                if rider.position == rider.return_destination:
                    rider.status = "idle"
                    rider.destination = None
                    rider.eta_ticks = 0

        # 4. Process restock arrivals
        remaining_restocks = []
        for arrival_tick, item_name, quantity in self._restock_queue:
            if self._tick >= arrival_tick:
                for shelf in self._shelves:
                    if shelf.item_name == item_name:
                        shelf.stock += quantity
            else:
                remaining_restocks.append((arrival_tick, item_name, quantity))
        self._restock_queue = remaining_restocks

        # 5. Process perishable expiry
        for shelf in self._shelves:
            if shelf.expiry_ticks > 0:
                shelf.expiry_ticks -= 1
                if shelf.expiry_ticks == 0 and shelf.stock > 0:
                    expired_units = shelf.stock
                    tick_reward += -3.0 * expired_units
                    shelf.stock = 0

        # 6. Spawn new orders at scheduled ticks
        self._spawn_scheduled_orders()

        # 7. Check episode termination
        if self._tick >= cfg.tick_budget:
            self._done = True
            # Penalize undelivered orders at episode end
            for order in self._orders.values():
                if order.status not in ("delivered",):
                    tick_reward += -15.0
                    self._completed.append(CompletedDeliveryInfo(
                        order_id=order.order_id,
                        on_time=False,
                    ))
                    order.status = "delivered"
        elif self._all_orders_delivered():
            self._done = True

        return tick_reward

    def _spawn_scheduled_orders(self) -> None:
        """Add orders scheduled for the current tick."""
        remaining = []
        for sched_tick, order_specs in self._order_schedule:
            if sched_tick <= self._tick:
                for spec in order_specs:
                    if spec.order_id not in self._orders:
                        self._orders[spec.order_id] = Order(
                            order_id=spec.order_id,
                            items=list(spec.items),
                            picked_items=[],
                            customer_pos=spec.customer_pos,
                            timer_ticks=spec.timer_ticks,
                            status="pending",
                        )
            else:
                remaining.append((sched_tick, order_specs))
        self._order_schedule = remaining

    def _all_orders_delivered(self) -> bool:
        """Check if all orders (current + future) are delivered."""
        if self._order_schedule:
            return False  # more orders coming
        for order in self._orders.values():
            if order.status != "delivered":
                return False
        return True

    def _find_shelf(
        self, item_name: str, position: Tuple[int, int]
    ) -> Optional[Shelf]:
        """Find a shelf matching item_name at the given position."""
        for shelf in self._shelves:
            if shelf.item_name == item_name and (shelf.row, shelf.col) == position:
                return shelf
        return None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        error: Optional[str],
    ) -> DarkStoreObservation:
        """Construct a DarkStoreObservation from current internal state."""
        cfg = self._task_config
        tick_budget = cfg.tick_budget if cfg else 20

        shelves = [
            ShelfInfo(
                item_name=s.item_name,
                row=s.row,
                col=s.col,
                stock=s.stock,
                expiry_ticks=s.expiry_ticks,
            )
            for s in self._shelves
        ]

        pending_orders = [
            OrderInfo(
                order_id=o.order_id,
                items=list(o.items),
                picked_items=list(o.picked_items),
                timer_ticks=o.timer_ticks,
                customer_position=o.customer_pos,
            )
            for o in self._orders.values()
            if o.status in ("pending", "packing")
        ]

        riders = [
            RiderInfo(
                rider_id=r.rider_id,
                position=r.position,
                status=r.status,
                eta_ticks=self._compute_rider_eta(r),
            )
            for r in self._riders.values()
        ]

        active_deliveries = []
        for o in self._orders.values():
            if o.status == "delivering":
                rider_id = ""
                for r in self._riders.values():
                    if o.order_id in r.delivering_orders:
                        rider_id = r.rider_id
                        break
                active_deliveries.append(DeliveryInfo(
                    order_id=o.order_id,
                    customer_position=o.customer_pos,
                    timer_ticks=o.timer_ticks,
                    rider_id=rider_id,
                ))

        self._last_error = error
        text = self._render_text()

        return DarkStoreObservation(
            tick=self._tick,
            ticks_remaining=max(0, tick_budget - self._tick),
            picker_position=self._picker_pos,
            picker_holding=list(self._picker_holding),
            shelves=shelves,
            pending_orders=pending_orders,
            packed_orders=list(self._packed_orders),
            riders=riders,
            active_deliveries=active_deliveries,
            completed_deliveries=list(self._completed),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            reward=round(reward, 4),
            error=error,
            text=text,
        )

    def _compute_rider_eta(self, rider: Rider) -> int:
        """Compute ETA ticks for a rider to finish current task."""
        if rider.status == "idle":
            return 0
        if rider.status == "returning":
            return _manhattan(rider.position, rider.return_destination)
        if rider.status == "delivering" and rider.delivery_stops:
            eta = 0
            pos = rider.position
            for stop_pos, _ in rider.delivery_stops:
                eta += _manhattan(pos, stop_pos)
                pos = stop_pos
            eta += _manhattan(pos, rider.return_destination)
            return eta
        return 0

    # ------------------------------------------------------------------
    # Grader
    # ------------------------------------------------------------------

    def compute_score(self) -> float:
        """Compute normalized score in (0.0, 1.0) — strictly between 0 and 1."""
        if self._task_config is None:
            return 0.001
        max_reward = self._task_config.max_theoretical_reward
        if max_reward <= 0:
            return 0.001
        score = self._cumulative_reward / max_reward
        # Clamp to open interval (0, 1) — competition requires strictly between 0 and 1
        return max(0.001, min(0.999, score))

    # ------------------------------------------------------------------
    # Action hint for LLM
    # ------------------------------------------------------------------

    def _compute_action_hint(self) -> str:
        """Compute a contextual hint about the current situation."""
        # 1. Packed orders + idle rider
        if self._packed_orders:
            idle_riders = [r for r in self._riders.values() if r.status == "idle"]
            if idle_riders:
                return f"Order {self._packed_orders[0]} is packed and {idle_riders[0].rider_id} is idle — ready for dispatch"

        # 2. Picker at (0,0) with all items for an order
        if self._picker_pos == (0, 0):
            for order in self._orders.values():
                if order.status == "pending":
                    picked_counts: Dict[str, int] = {}
                    for pi in order.picked_items:
                        picked_counts[pi] = picked_counts.get(pi, 0) + 1
                    missing = [item for item in set(order.items)
                               if picked_counts.get(item, 0) < order.items.count(item)]
                    if not missing:
                        return f"All items for {order.order_id} are picked and picker is at packing station"

        # 3. All items picked → need to go to packing station
        for order in self._orders.values():
            if order.status == "pending":
                picked_counts: Dict[str, int] = {}
                for pi in order.picked_items:
                    picked_counts[pi] = picked_counts.get(pi, 0) + 1
                missing = [item for item in set(order.items)
                           if picked_counts.get(item, 0) < order.items.count(item)]
                if not missing and self._picker_pos != (0, 0):
                    return f"All items for {order.order_id} picked — picker needs to reach packing station (0,0)"

        # 4. Picker is at a shelf with a needed item
        for order in self._orders.values():
            if order.status != "pending":
                continue
            picked_counts: Dict[str, int] = {}
            for pi in order.picked_items:
                picked_counts[pi] = picked_counts.get(pi, 0) + 1
            for item in order.items:
                if picked_counts.get(item, 0) >= order.items.count(item):
                    continue
                for shelf in self._shelves:
                    if (shelf.item_name == item
                            and (shelf.row, shelf.col) == self._picker_pos
                            and shelf.stock > 0):
                        return f"Picker is at {item} shelf ({shelf.row},{shelf.col}) — item needed for {order.order_id}"

        # 5. Need to pick items
        for order in self._orders.values():
            if order.status != "pending":
                continue
            picked_counts: Dict[str, int] = {}
            for pi in order.picked_items:
                picked_counts[pi] = picked_counts.get(pi, 0) + 1
            for item in order.items:
                if picked_counts.get(item, 0) >= order.items.count(item):
                    continue
                for shelf in self._shelves:
                    if shelf.item_name == item and shelf.stock > 0:
                        return f"Need {item} for {order.order_id} — available at shelf ({shelf.row},{shelf.col})"

        return ""

    # ------------------------------------------------------------------
    # Text rendering for LLM consumption
    # ------------------------------------------------------------------

    def _render_text(self) -> str:
        """Render a human-readable text observation for LLM agents."""
        cfg = self._task_config
        tick_budget = cfg.tick_budget if cfg else 20
        remaining = max(0, tick_budget - self._tick)

        lines: List[str] = []
        lines.append(
            f"=== DARK STORE — Tick {self._tick} / {tick_budget} "
            f"| {remaining} ticks remaining ==="
        )

        # Picker
        holding_str = (
            ", ".join(self._picker_holding) if self._picker_holding else "empty"
        )
        lines.append(
            f"PICKER  position=({self._picker_pos[0]}, {self._picker_pos[1]})  "
            f"holding=[{holding_str}]"
        )

        # Show hint only when agent made an error (to help it recover)
        if getattr(self, '_last_error', None):
            hint = self._compute_action_hint()
            if hint:
                lines.append(f"HINT: {hint}")

        # Pending orders
        pending = [
            o for o in self._orders.values()
            if o.status in ("pending", "packing")
        ]
        if pending:
            lines.append("PENDING ORDERS:")
            for o in pending:
                items_display = []
                picked_counts: Dict[str, int] = {}
                for pi in o.picked_items:
                    picked_counts[pi] = picked_counts.get(pi, 0) + 1
                used_counts: Dict[str, int] = {}
                for item in o.items:
                    used_counts[item] = used_counts.get(item, 0) + 1
                    if used_counts[item] <= picked_counts.get(item, 0):
                        items_display.append(f"{item}\u2713")
                    else:
                        items_display.append(item)
                urgency = "  URGENT" if o.timer_ticks < 5 else ""
                packing_note = "  [PACKING]" if o.status == "packing" else ""
                lines.append(
                    f"  {o.order_id}  items=[{', '.join(items_display)}]  "
                    f"timer={o.timer_ticks}  "
                    f"customer=({o.customer_pos[0]},{o.customer_pos[1]})"
                    f"{urgency}{packing_note}"
                )
        else:
            lines.append("PENDING ORDERS: none")

        # Shelves
        lines.append("SHELVES:")
        for s in self._shelves:
            markers = []
            if s.stock == 0:
                markers.append("STOCKOUT")
            if s.expiry_ticks > 0 and s.expiry_ticks < 3:
                markers.append("EXPIRING")
            marker_str = "  " + " ".join(markers) if markers else ""
            if s.expiry_ticks >= 0:
                expiry_str = f"expiry={s.expiry_ticks}"
            else:
                expiry_str = "no-expiry"
            lines.append(
                f"  {s.item_name} \u2192 ({s.row},{s.col}) stock={s.stock} "
                f"{expiry_str}{marker_str}"
            )

        # Packed orders
        if self._packed_orders:
            lines.append(
                f"PACKED ORDERS: [{', '.join(self._packed_orders)}]"
            )
        else:
            lines.append("PACKED ORDERS: none")

        # Riders
        lines.append("RIDERS:")
        for r in self._riders.values():
            eta = self._compute_rider_eta(r)
            eta_str = f"  ETA={eta}" if r.status != "idle" else ""
            lines.append(
                f"  {r.rider_id}  {r.status} at "
                f"({r.position[0]},{r.position[1]}){eta_str}"
            )

        # Active deliveries
        active = [
            o for o in self._orders.values() if o.status == "delivering"
        ]
        if active:
            lines.append("ACTIVE DELIVERIES:")
            for o in active:
                rider_id = ""
                for r in self._riders.values():
                    if o.order_id in r.delivering_orders:
                        rider_id = r.rider_id
                        break
                urgency = "  URGENT" if o.timer_ticks < 5 else ""
                lines.append(
                    f"  {o.order_id} \u2192 "
                    f"({o.customer_pos[0]},{o.customer_pos[1]}) "
                    f"timer={o.timer_ticks} rider={rider_id}{urgency}"
                )

        # Completed
        on_time_count = sum(1 for c in self._completed if c.on_time)
        late_count = sum(1 for c in self._completed if not c.on_time)
        lines.append(f"COMPLETED: {on_time_count} on-time, {late_count} late")

        # Reward
        lines.append(f"REWARD: {self._cumulative_reward:.2f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return current environment state metadata."""
        return self._state
