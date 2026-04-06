"""
Data models for the Dark Store Simulator Environment.

Defines the action and observation Pydantic models for the quick commerce
dark store simulation, extending the OpenEnv base types.
"""

from typing import List, Optional, Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DarkStoreAction(Action):
    """Action for the Dark Store environment.

    The ``action`` field selects the operation type. Additional fields are
    required or ignored depending on the chosen action type:

    - move_picker: row, col
    - pick: row, col, item_name
    - pack: order_id
    - assign_rider: order_id, rider_id
    - batch_delivery: order_a, order_b, rider_id
    - restock: item_name, quantity
    - wait: (no extra fields)
    """

    action: str = Field(..., description="Action type: move_picker | pick | pack | assign_rider | batch_delivery | restock | wait")
    row: Optional[int] = Field(default=None, description="Target row on the grid")
    col: Optional[int] = Field(default=None, description="Target column on the grid")
    item_name: Optional[str] = Field(default=None, description="SKU / item name")
    order_id: Optional[str] = Field(default=None, description="Order identifier")
    rider_id: Optional[str] = Field(default=None, description="Rider identifier")
    order_a: Optional[str] = Field(default=None, description="First order id for batch delivery")
    order_b: Optional[str] = Field(default=None, description="Second order id for batch delivery")
    quantity: Optional[int] = Field(default=None, description="Quantity for restock action")


# ---------------------------------------------------------------------------
# Helper Pydantic models (used inside DarkStoreObservation)
# ---------------------------------------------------------------------------

class ShelfInfo(BaseModel):
    """Snapshot of a single shelf / SKU slot."""

    item_name: str = Field(..., description="SKU name on this shelf")
    row: int = Field(..., description="Shelf row on the dark-store grid")
    col: int = Field(..., description="Shelf column on the dark-store grid")
    stock: int = Field(..., description="Current stock quantity")
    expiry_ticks: int = Field(..., description="Ticks until expiry (-1 = non-perishable)")


class OrderInfo(BaseModel):
    """Snapshot of a pending order."""

    order_id: str = Field(..., description="Unique order identifier")
    items: List[str] = Field(default_factory=list, description="Required item names")
    picked_items: List[str] = Field(default_factory=list, description="Items already picked")
    timer_ticks: int = Field(..., description="Ticks remaining on the order timer")
    customer_position: Tuple[int, int] = Field(..., description="Customer (row, col) on city grid")


class RiderInfo(BaseModel):
    """Snapshot of a rider's current state."""

    rider_id: str = Field(..., description="Unique rider identifier")
    position: Tuple[int, int] = Field(..., description="Current (row, col) on city grid")
    status: str = Field(..., description="idle | delivering | returning")
    eta_ticks: int = Field(default=0, description="Estimated ticks to destination (0 if idle)")


class DeliveryInfo(BaseModel):
    """Snapshot of an active (in-flight) delivery."""

    order_id: str = Field(..., description="Order being delivered")
    customer_position: Tuple[int, int] = Field(..., description="Destination (row, col) on city grid")
    timer_ticks: int = Field(..., description="Ticks remaining on the order timer")
    rider_id: str = Field(..., description="Assigned rider identifier")


class CompletedDeliveryInfo(BaseModel):
    """Record of a completed delivery."""

    order_id: str = Field(..., description="Delivered order identifier")
    on_time: bool = Field(..., description="True if delivered within the timer limit")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class DarkStoreObservation(Observation):
    """Observation returned by the Dark Store environment after every step/reset.

    Inherits ``done``, ``reward``, and ``metadata`` from the base
    :class:`Observation`.
    """

    tick: int = Field(default=0, description="Current simulation tick")
    ticks_remaining: int = Field(default=0, description="Ticks left in the episode")
    picker_position: Tuple[int, int] = Field(default=(0, 7), description="Picker (row, col) on dark-store grid")
    picker_holding: List[str] = Field(default_factory=list, description="Items currently held by the picker (max 5)")
    shelves: List[ShelfInfo] = Field(default_factory=list, description="Current state of all shelves")
    pending_orders: List[OrderInfo] = Field(default_factory=list, description="Orders awaiting fulfilment")
    packed_orders: List[str] = Field(default_factory=list, description="Order ids packed and ready for dispatch")
    riders: List[RiderInfo] = Field(default_factory=list, description="All rider statuses")
    active_deliveries: List[DeliveryInfo] = Field(default_factory=list, description="Deliveries currently in progress")
    completed_deliveries: List[CompletedDeliveryInfo] = Field(default_factory=list, description="Deliveries that have been completed")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated so far")
    error: Optional[str] = Field(default=None, description="Error message if the last action was invalid")
    text: str = Field(default="", description="Human-readable state summary for LLM consumption")
