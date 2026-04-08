---
title: Dark Store Simulator
emoji: 🏪
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Dark Store Simulator

A quick-commerce dark store simulation environment built on [OpenEnv](https://github.com/meta-llama/openenv). An AI agent coordinates picking, packing, dispatching, batching, and restocking across a 10×8 warehouse grid and an 8×8 city delivery grid, fulfilling customer orders under tight time constraints — modelling the operations of a Zepto/Blinkit-style instant delivery warehouse.

## Quick Start

```python
from my_env import DarkStoreAction, DarkStoreClient

try:
    # Create environment from Docker image
    client = DarkStoreClient.from_docker_image("dark_store-env:latest")

    # Reset to a task
    result = client.reset(task="single_order")
    obs = result.observation
    print(f"Tick {obs.tick} | Picker at {obs.picker_position}")
    print(f"Pending orders: {len(obs.pending_orders)}")

    # Pick an item from a shelf
    result = client.step(DarkStoreAction(action="move_picker", row=1, col=1))
    print(f"Picker moved to {result.observation.picker_position}")

    result = client.step(DarkStoreAction(action="pick", row=1, col=1, item_name="milk"))
    print(f"Holding: {result.observation.picker_holding}")

    # Wait a tick
    result = client.step(DarkStoreAction(action="wait"))
    print(f"Reward so far: {result.observation.cumulative_reward}")

finally:
    client.close()
```

Or use the context manager:

```python
with DarkStoreClient(base_url="http://localhost:8000") as client:
    result = client.reset(task="single_order")
    while not result.observation.done:
        result = client.step(DarkStoreAction(action="wait"))
    print(f"Final score: {result.observation.cumulative_reward}")
```

## Action Space

The agent sends one `DarkStoreAction` per tick. The `action` field selects the operation; additional fields are required depending on the type.

| Action | Required Fields | Description |
|---|---|---|
| `move_picker` | `row`, `col` | Move the picker one cell toward target per tick. Costs -0.05 per cell moved. |
| `pick` | `row`, `col`, `item_name` | Pick an item from a shelf. Picker must be at the shelf location, stock > 0, and holding < 5 items. |
| `pack` | `order_id` | Pack a completed order at the packing station (0,0). Takes 2 ticks. Picker must be at (0,0) with all items picked. |
| `assign_rider` | `order_id`, `rider_id` | Dispatch an idle rider to deliver a packed order to the customer. |
| `batch_delivery` | `order_a`, `order_b`, `rider_id` | Dispatch one idle rider to deliver two packed orders. Visits the nearer customer first. +2.0 bonus. |
| `restock` | `item_name`, `quantity` | Schedule emergency restock arriving in 4 ticks. Costs -1.0 flat penalty. |
| `wait` | _(none)_ | Advance the simulation by 1 tick with no other effect. |

## Observation Space

Every call to `step()` and `reset()` returns a `DarkStoreObservation` with:

| Field | Type | Description |
|---|---|---|
| `tick` | `int` | Current simulation tick |
| `ticks_remaining` | `int` | Ticks left in the episode |
| `picker_position` | `(int, int)` | Picker (row, col) on the 10×8 warehouse grid |
| `picker_holding` | `List[str]` | Items held by the picker (max 5) |
| `shelves` | `List[ShelfInfo]` | All shelf states: item_name, (row, col), stock, expiry_ticks |
| `pending_orders` | `List[OrderInfo]` | Orders awaiting fulfilment: order_id, items, picked_items, timer_ticks, customer_position |
| `packed_orders` | `List[str]` | Order IDs packed and ready for dispatch |
| `riders` | `List[RiderInfo]` | Rider states: rider_id, position, status (idle/delivering/returning), eta_ticks |
| `active_deliveries` | `List[DeliveryInfo]` | In-flight deliveries: order_id, customer_position, timer_ticks, rider_id |
| `completed_deliveries` | `List[CompletedDeliveryInfo]` | Completed deliveries: order_id, on_time |
| `cumulative_reward` | `float` | Total reward accumulated so far |
| `error` | `Optional[str]` | Error message if the last action was invalid |
| `text` | `str` | Human-readable state summary for LLM consumption |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward from the last step |

The `text` field includes `URGENT` (order timer < 5), `STOCKOUT` (stock = 0), and `EXPIRING` (expiry < 3 ticks) markers for LLM agents.

## Tasks

Three tasks of increasing difficulty, each with a deterministic seed for reproducible evaluation:

| Task | Difficulty | Seed | Ticks | Orders | Riders | Key Challenge |
|---|---|---|---|---|---|---|
| `single_order` | Easy | 42 | 20 | 1 | 1 | Basic pick → pack → deliver workflow |
| `concurrent_orders` | Medium | 123 | 30 | 5 (waves) | 3 | Prioritization, batch delivery (2 eligible pairs) |
| `full_operations` | Hard | 456 | 40 | 10 | 2+1 | Stockouts (coke, juice at 0), bread expires tick 3, restocking |

### Baseline Scores

| Task | Max Theoretical Reward | Baseline Target |
|---|---|---|
| `single_order` | ~9.30 | Score approaching 1.0 with optimal play |
| `concurrent_orders` | ~52.0 | Requires batching and wave management |
| `full_operations` | ~100.0 | Requires restocking, expiry management, and tight scheduling |

## Reward Function

| Event | Reward |
|---|---|
| On-time delivery (timer > 0) | +10.0 |
| Late delivery (timer ≤ 0) | -15.0 |
| Batch delivery bonus | +2.0 |
| Picker movement (per cell) | -0.05 |
| Stockout pick attempt | -5.0 |
| Restock action | -1.0 |
| Item expiry (per unit) | -3.0 |

The grader computes a normalized score: `max(0.0, cumulative_reward / max_theoretical_reward)` clamped to [0.0, 1.0].

## Setup

### Docker Build

```bash
# From the my_env/ directory
docker build -t dark_store-env:latest -f server/Dockerfile .
```

### Run Locally

```bash
# Start the server
cd my_env
uvicorn server.app:app --reload --port 8000
```

The server exposes:
- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Docs** at `/docs` — OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint

### Deploy to HuggingFace Spaces

```bash
# From the my_env/ directory (where openenv.yaml is located)
openenv push

# Or with options
openenv push --namespace my-org --private
```

After deployment, your space will be available at `https://huggingface.co/spaces/<repo-id>`.

### Run the Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://your-llm-api.com/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="your-token"

# Run all 3 tasks
python inference.py
```

The inference script outputs `[START]`, `[STEP]`, and `[END]` lines to stdout in the mandatory OpenEnv format.

## Project Structure

```
my_env/
├── __init__.py                # Module exports (DarkStoreClient, DarkStoreAction, DarkStoreObservation)
├── README.md                  # This file
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependencies
├── client.py                  # DarkStoreClient (WebSocket + HTTP)
├── models.py                  # DarkStoreAction, DarkStoreObservation, helper models
├── server/
│   ├── __init__.py            # Server module exports
│   ├── dark_store_environment.py  # Core simulation engine
│   ├── app.py                 # FastAPI application (HTTP + WebSocket endpoints)
│   └── Dockerfile             # Container image definition
└── tests/
    ├── __init__.py
    ├── test_property_*.py     # Property-based tests (hypothesis)
    └── ...
```
