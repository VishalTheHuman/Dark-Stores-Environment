---
title: Dark Store Simulator
emoji: 🏪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🏪 Dark Store Simulator

> *Can an AI agent run a Zepto/Blinkit-style dark store — picking, packing, and delivering groceries in under 10 minutes?*

---

## 🎯 What Is This?

When you open **Zepto** or **Blinkit** and tap "Order", a countdown begins. You expect your groceries in 10 minutes. Behind that promise is one of the hardest real-time coordination problems in modern retail — and it's happening thousands of times per hour, simultaneously, across hundreds of dark stores in every major city.

A **dark store** is a small warehouse — roughly the size of a large convenience store — that is closed to the public and exists only to fulfill instant delivery orders. It has narrow aisles, hundreds of SKUs (stock-keeping units), a packing station near the exit, and a fleet of delivery riders waiting outside. There are no customers walking the aisles. Everything is optimised for one thing: **speed**.

This OpenEnv environment simulates the full operations of such a dark store. An AI agent must learn to coordinate all of the following — **simultaneously**:

📦 **Orders arrive unpredictably** — You don't know when the next order will land or what items it will contain. A picker in aisle 3 might need to reverse course the moment a new order arrives with an item in aisle 1.

🏃 **A single picker robot** navigates the warehouse grid, collecting items from shelves and carrying them to the packing station. The path the picker takes directly affects whether an order makes it within the deadline.

📋 **Multiple orders compete for the same items** — If three customers all order milk at the same time, and there are only four units left, the picker must decide which orders to fulfill now and which to delay for restock.

🏍️ **Riders are shared across orders** — Three orders landing at nearby addresses can be batched into one rider trip, saving time and cost. But holding a packed order too long waiting for a second order to be ready risks making the first order late.

⏱️ **The 10-minute clock is unforgiving** — The entire pick → pack → dispatch → deliver chain must complete within the timer. A single bad routing decision — sending a rider the long way, or picking items in the wrong order — breaks the promise.

🧊 **Inventory depletes and expires** — Fast-moving items run out. Perishables like bread and milk have expiry counters. The agent must restock proactively — not after a customer's order fails because coke is at zero.

---

## 🧠 Why This Matters (Real-World Utility)

This isn't a toy problem. Quick commerce is a **$50B+ industry** growing rapidly in India, Southeast Asia, and beyond. Companies like Zepto, Blinkit, Swiggy Instamart, and Getir face exactly this coordination challenge at scale.

Training AI agents on this environment could help with:
- 🔬 **Research** — studying multi-objective optimization under time pressure
- 🏭 **Operations** — testing warehouse automation strategies before deploying them
- 🤖 **Agent evaluation** — benchmarking how well LLMs handle structured, multi-step planning with competing constraints
- 📊 **Reward design** — exploring dense reward shaping for complex logistics tasks

The environment fills a gap in OpenEnv: there are game environments and simple task environments, but very few that model **real-world logistics operations** with the full complexity of inventory, routing, timing, and resource allocation.

---

## ⚙️ How It Works

The simulation runs on two grids and advances one **tick** per step (1 tick = 30 seconds of simulated time):

### The Pipeline (How a Single Order Flows)

```
📱 Customer places order (items + address)
    ↓ timer starts counting down
🚶 Picker navigates warehouse aisles to each item's shelf
    ↓ 1 tick per cell moved (BFS path through walkways)
✋ Picker picks items one by one from shelves
    ↓ stock decrements, item added to picker's hands (max 5)
🚶 Picker walks back to packing station at (0,0)
    ↓
📦 Pack order — assembles items into a bag (takes 2 ticks)
    ↓
🏍️ Assign rider — idle rider dispatched to customer
    ↓ rider moves 1 cell/tick on city grid
✅ Delivery confirmed — timer stops
    ↓
💰 On-time? → +10.0 reward
❌ Late?    → -15.0 penalty
```

### What Makes It Hard

The challenge isn't any single order — it's **10 orders simultaneously**, each with different items, different customers, different timers, competing for the same picker, the same riders, and the same inventory. The agent must constantly decide:

- 🤔 *Which order to pick next?* (the most urgent? the one with items nearby?)
- 🤔 *Should I pick items for two orders in one trip?* (saves time but delays the first order)
- 🤔 *Should I batch these two deliveries?* (saves a rider but the first order waits)
- 🤔 *Should I restock coke now?* (costs -1.0 but prevents -5.0 stockout later)
- 🤔 *Should I use the expiring bread first?* (prevents -3.0/unit waste)

---

## 🗺️ The Two Grids

### 🏭 Warehouse (10 × 8)

```
     col: 0    1     2     3     4     5     6     7     8     9
          ┃         AISLE 1          AISLE 2          AISLE 3    ┃
row 0: [📦]  ·     ·     ·     ·     ·     ·     ·     ·     ·   ← corridor
row 1:   ·   [🥛]  [🧀]   ·   [🧃]  [🥚]   ·   [🧈]   ·     ·
row 2:   ·   [🍞]  [🍟]   ·   [🥤]  [🍪]   ·   [🍚]   ·     ·
row 3:   ·   [🫘]  [🫒]   ·   [🧀]  [🧃]   ·   [🍟]   ·     ·
row 4:   ·     ·     ·     ·     ·     ·     ·     ·     ·     ·
row 5:   ·     ·     ·     ·     ·     ·     ·     ·     ·     ·
row 6:   ·     ·     ·     ·     ·     ·     ·     ·     ·     ·
row 7:   ·     ·     ·     ·     ·     ·     ·   [🤖]    ·     ·   ← corridor
        walk                                                walk
```

📦 = Packing Station (0,0) · 🤖 = Picker Start (0,7) · 🛒 = Walkways: col 0, 9 · 🚶 = Corridors: row 0, 7

### 🏙️ City Delivery Grid (8 × 8)

```
     0      1      2      3      4      5      6      7
0: [🏪]    ·      ·      ·      ·      ·      ·      ·
1:   ·      ·      ·    [👤C1]   ·      ·      ·      ·
2:   ·    [🏍️A]   ·      ·    [👤C2]   ·      ·      ·
3:   ·      ·      ·      ·      ·      ·      ·      ·
4:   ·      ·      ·      ·      ·      ·      ·      ·
5:   ·      ·      ·      ·      ·    [🏍️C]   ·      ·
6:   ·      ·      ·      ·      ·      ·      ·      ·
7:   ·      ·      ·      ·      ·      ·      ·      ·
```

🏪 = Store (0,0) · 👤 = Customer · 🏍️ = Rider (1 cell/tick)

---

## 🧺 Shelf Inventory

```
SKU          📍 Location   📊 Stock   🧊 Perishable?
───────────  ───────────   ────────   ─────────────
🥛 milk      (1,1)         8          ✅ Yes
🍞 bread     (2,1)         6          ✅ Yes
🫘 dal       (3,1)         7          ❌ No
🧀 curd      (1,2)         5          ✅ Yes
🍟 chips     (2,2)         12         ❌ No
🫒 oil       (3,2)         5          ❌ No
🧃 juice     (1,4)         8          ✅ Yes
🥤 coke      (2,4)         10         ❌ No
🧀 curd      (3,4)         4          ✅ Yes
🥚 eggs      (1,5)         4          ✅ Yes
🍪 biscuits  (2,5)         9          ❌ No
🧃 juice     (3,5)         8          ✅ Yes
🧈 butter    (1,7)         8          ✅ Yes
🍚 rice      (2,7)         3          ❌ No
🍟 chips     (3,7)         11         ❌ No
```

---


## 🎮 Action Space

One action per tick. Invalid actions return an error but never crash the simulation.

### 🚶 `move_picker` — Navigate the warehouse
```json
{"action": "move_picker", "row": 1, "col": 1}
```
Teleports picker via walkable BFS path. Cost: **-0.05 per cell** traversed.

### ✋ `pick` — Grab an item from a shelf
```json
{"action": "pick", "row": 1, "col": 1, "item_name": "milk"}
```
Must be AT the shelf. Stock > 0. Max 5 items held. ⚠️ Stockout = **-5.0**

### 📦 `pack` — Pack an order at the station
```json
{"action": "pack", "order_id": "Order-1"}
```
Picker must be at (0,0). All items picked. Takes **2 ticks**.

### 🏍️ `assign_rider` — Send a delivery
```json
{"action": "assign_rider", "order_id": "Order-1", "rider_id": "Rider-A"}
```
Order must be packed. Rider must be idle.

### 🏍️🏍️ `batch_delivery` — Two orders, one trip!
```json
{"action": "batch_delivery", "order_a": "Order-1", "order_b": "Order-2", "rider_id": "Rider-A"}
```
Both packed. Nearer customer first. 🎁 **+2.0 bonus** per delivery!

### 📥 `restock` — Emergency inventory refill
```json
{"action": "restock", "item_name": "coke", "quantity": 8}
```
Arrives in **4 ticks**. Costs **-1.0** (still better than -5.0 stockout).

### ⏳ `wait` — Do nothing
```json
{"action": "wait"}
```
Sometimes the right call — wait for a second order to batch.

---

## 👁️ Observation Space

Every step returns a full state snapshot. Here's what the LLM sees:

```
=== DARK STORE — Tick 5 / 40 | 35 ticks remaining ===
🤖 PICKER  position=(2, 2)  holding=[milk, chips]
📋 PENDING ORDERS:
  Order-1  items=[milk✓, chips✓, eggs]  timer=35  customer=(3,1)
  Order-2  items=[curd, bread]          timer=33  customer=(4,2)  ■ URGENT
📊 SHELVES:
  coke → (2,4) stock=0 ← STOCKOUT
  bread → (2,1) stock=6 expiry=2 ← EXPIRING
📦 PACKED ORDERS: none
🏍️ RIDERS:
  Rider-A  idle at (0,0)
  Rider-B  delivering Order-3  ETA=4
✅ COMPLETED: 1 on-time, 0 late
💰 REWARD: +9.45
```

### 🏷️ Text Markers
- 🔴 **URGENT** — order timer < 5 ticks
- ⚠️ **STOCKOUT** — shelf stock = 0
- 🧊 **EXPIRING** — perishable with < 3 ticks left

---

## 📝 Tasks (Easy → Hard)

### 🟢 Task 1: `single_order` — Easy

```
📦 Orders: 1 (at tick 0)       ⏱️ Budget: 40 ticks
🏍️ Riders: 1 idle              🧺 Items: milk, chips, eggs
👤 Customer: (3,1)             ⚠️ Stockouts: None
```

The simplest scenario. One order, one rider, all items in stock, generous time budget. The agent must navigate to each item's shelf, pick all three in the shortest path, walk to the packing station, pack the order, assign the rider, and confirm delivery.

> 💡 **What it tests**: Can the agent follow the basic pick → pack → deliver workflow? Does it understand the action sequence?

### 🟡 Task 2: `concurrent_orders` — Medium

```
📦 Orders: 5 in waves           ⏱️ Budget: 60 ticks
🏍️ Riders: 3 idle               🔄 Waves: 2 at tick 0, 1 at tick 8, 2 at tick 20
🏍️🏍️ Batch pairs: Orders (1,2) at (3,1)/(4,2) and Orders (4,5) at (1,6)/(2,7)
```

Five orders arrive in waves. Two pairs of customers are close enough to batch. The agent must prioritize orders by urgency (lowest timer first), batch the nearby deliveries (saves 2 full rider trips), sequence the picker to serve multiple orders without backtracking, and avoid leaving riders idle while packed orders wait.

> 💡 **What it tests**: Order prioritization, batch delivery recognition, multi-order picker routing, rider utilization. A naive agent that processes orders one-at-a-time will score poorly.

### 🔴 Task 3: `full_operations` — Hard

```
📦 Orders: 10 across episode    ⏱️ Budget: 80 ticks
🏍️ Riders: 2 idle + 1 returning (Rider-C at (5,5), ETA tick 6)
⚠️ Stockouts: coke=0, juice=0 (must restock before orders need them)
🧊 Bread expires at tick 3! (6 units lost if not picked first)
🏍️🏍️ Batch pairs: 3 eligible
```

The full challenge. The agent must:
1. 📥 **Restock coke and juice proactively** — before orders arrive that need them (not reactively after a stockout)
2. 🧊 **Use near-expiry bread first** — pick it before tick 3 or lose 6 units (-18.0 penalty)
3. 🏍️🏍️ **Batch 3 pairs of nearby deliveries** — saves 3 full rider trips
4. 🏍️ **Coordinate the returning rider** — assign Rider-C immediately when it arrives at tick 6
5. 📦 **Stagger packing** — don't dump all items at the station at once (creates a bottleneck)

> 💡 **What it tests**: Proactive inventory management, FIFO expiry handling, multi-rider coordination, packing flow control. This task genuinely challenges frontier models.

---

## 📊 Order Flow

```
📱 Order arrives
    ↓
🚶 Picker navigates to shelves
    ↓
✋ Items picked one by one
    ↓
🚶 Picker returns to (0,0)
    ↓
📦 Pack order (2 ticks)
    ↓
🏍️ Assign rider → delivers to customer
    ↓
✅ On-time (+10.0) or ❌ Late (-15.0)
```

---

## 💰 Reward Function

The reward is **dense and continuous** — the agent receives meaningful signal on every tick, not just at episode end. Rewards are shaped to guide productive behavior.

```
Event                      Reward          Details
─────────────────────────  ──────────────  ──────────────────────────
✅ On-time delivery         +10 to +15     Scales with time remaining
                                           (+0.25 per tick left, max +5 bonus)
📦 Order packed             +3.0           All items picked, packing started
🏍️ Rider dispatched         +2.0           Packed order sent for delivery
✋ Item picked              +1.0           Each successful pick from shelf
🚶 Productive move          -0.025/cell    Moving toward needed shelf (half cost)
🚶 Wasteful move            -0.10/cell     Moving away from targets (double cost)
🏍️🏍️ Batch delivery bonus    +2.0/order     Two orders, one rider trip
❌ Late/undelivered         -5.0           Per order at episode end
⚠️ Stockout pick            -5.0           Trying to pick from empty shelf
📥 Restock fee              -1.0           Emergency inventory refill
🧊 Item expiry              -3.0/unit      Perishable items lost on shelf
❌ Invalid action           -0.1           Wrong action type or parameters
```

### 🧮 Reward Design Philosophy

The reward function is **shaped, not sparse**:

- **Movement is always negative** but productive moves cost half — the agent learns efficient routing without being punished for necessary travel
- **Delivery reward scales with time** — delivering with 20 ticks left earns +15.0, delivering with 1 tick left earns +10.25. This incentivizes speed, not just completion
- **Progress milestones** (+1 pick, +3 pack, +2 dispatch) give the agent positive feedback throughout the episode, not just at delivery
- **Penalties are proportional** — a small mistake (-0.1) is recoverable, but stockouts (-5.0) and late deliveries (-5.0) are significant

**Score** = `max(0.001, min(0.999, cumulative / max_theoretical))`

### 🏆 Max Theoretical Rewards
- 🟢 `single_order`: ~24.0 (3 picks + pack + dispatch + on-time delivery with bonus)
- 🟡 `concurrent_orders`: ~115.0 (13 picks + 5 packs + 5 dispatches + 5 deliveries + batch bonuses)
- 🔴 `full_operations`: ~214.0 (25 picks + 10 packs + 10 dispatches + 10 deliveries + batch - restock)

---

## 🎓 Worked Example: Completing Task 1 (single_order)

Here's a step-by-step walkthrough of an optimal agent completing Task 1. The order is: **milk, chips, eggs** for a customer at **(3,1)**.

```
Tick 0:  📱 Order-1 arrives: [milk, chips, eggs] → customer (3,1)
         🤖 Picker at (0,7). Rider-A idle at (0,0).

Tick 1:  🚶 move_picker → (1,5)  [eggs shelf]
         Cost: 3 cells × -0.05 = -0.15
         Picker teleports via walkway: (0,7)→(0,5)→(1,5)

Tick 2:  ✋ pick eggs at (1,5)
         Holding: [eggs]. Eggs stock: 4→3.

Tick 3:  🚶 move_picker → (2,2)  [chips shelf]
         Cost: 4 cells × -0.05 = -0.20
         Path: (1,5)→(1,4)→(1,2)→(2,2) via shelf cells

Tick 4:  ✋ pick chips at (2,2)
         Holding: [eggs, chips]. Chips stock: 12→11.

Tick 5:  🚶 move_picker → (1,1)  [milk shelf]
         Cost: 2 cells × -0.05 = -0.10

Tick 6:  ✋ pick milk at (1,1)
         Holding: [eggs, chips, milk]. All items picked!

Tick 7:  🚶 move_picker → (0,0)  [packing station]
         Cost: 2 cells × -0.05 = -0.10

Tick 8:  📦 pack Order-1
         Packing starts (takes 2 ticks). Items removed from hands.

Tick 9:  ⏳ wait (packing in progress...)

Tick 10: 📦 Order-1 is now PACKED and ready!
         🏍️ assign_rider Order-1 → Rider-A
         Rider-A departs (0,0) heading to customer (3,1).

Tick 11-13: 🏍️ Rider-A moving... (0,0)→(1,0)→(2,0)→(3,0)→(3,1)

Tick 14: ✅ DELIVERED! Timer had 26 ticks left. ON-TIME!
         Reward: +10.0
         Total: +10.0 - 0.55 (movement) = +9.45
         Score: 9.45 / 9.0 → clamped to 0.999
```

> 💡 Notice how the agent picks eggs first (closest to start), then chips, then milk — minimizing backtracking. A naive agent going milk→chips→eggs would take 2 extra cells.

---

## 🆚 Smart Agent vs Naive Agent (Task 2 Examples)

### Example 1: Batch Delivery Decision

**Situation**: Orders 1 and 2 are both packed. Customer 1 at (3,1), Customer 2 at (4,2). Rider-A and Rider-C both idle.

❌ **Naive agent** — sends two riders:
```json
{"action": "assign_rider", "order_id": "Order-1", "rider_id": "Rider-A"}
{"action": "assign_rider", "order_id": "Order-2", "rider_id": "Rider-C"}
```
Result: Both riders busy for 8+ ticks. No rider available for Order-3.

✅ **Smart agent** — batches:
```json
{"action": "batch_delivery", "order_a": "Order-1", "order_b": "Order-2", "rider_id": "Rider-A"}
```
Result: Rider-A delivers both (+2.0 bonus each). Rider-C stays free for Order-3. Saves 8 ticks of rider time.

### Example 2: Proactive Restocking (Task 3)

**Situation**: Tick 2. Coke stock = 0. Order-5 (arriving tick 15) will need coke.

❌ **Reactive agent** — ignores stockout, waits until Order-5 arrives:
```
Tick 15: Order-5 arrives needing coke
Tick 15: pick coke → STOCKOUT! -5.0 penalty
Tick 15: restock coke → arrives tick 20
Tick 20: finally picks coke, but timer nearly expired
Result: Late delivery (-15.0) + stockout (-5.0) + restock (-1.0) = -21.0
```

✅ **Proactive agent** — restocks immediately:
```
Tick 2: restock coke (-1.0) → arrives tick 7
Tick 15: Order-5 arrives, coke in stock!
Tick 18: On-time delivery (+10.0)
Result: +10.0 - 1.0 = +9.0 net
```

> 💡 The proactive agent saves **30.0 reward points** by spending -1.0 early.

### Example 3: Expiry-Aware Picking (Task 3)

**Situation**: Tick 1. Bread has 6 units, expires at tick 3. Order-1 needs bread.

❌ **Naive agent** — picks only what's needed:
```
Tick 1: pick 1 bread for Order-1
Tick 3: bread expires! 5 remaining units lost.
         Penalty: 5 × -3.0 = -15.0
```

✅ **Smart agent** — pre-picks expiring items:
```
Tick 1: pick 1 bread for Order-1
Tick 2: pick 1 extra bread (holds for future Order-7)
Tick 3: bread expires. Only 4 units lost.
         Penalty: 4 × -3.0 = -12.0 (saved 3.0!)
```

> 💡 Every bread unit saved from expiry is worth +3.0. The movement cost to pick it is only -0.05.

---

## 🔧 API Quick Start

### Python (Direct)

```python
from models import DarkStoreAction
from server.dark_store_environment import DarkStoreEnvironment

env = DarkStoreEnvironment()
obs = env.reset(task="single_order")

print(obs.text)  # Human-readable state

# Pick-pack-deliver loop
obs = env.step(DarkStoreAction(action="move_picker", row=1, col=5))
obs = env.step(DarkStoreAction(action="pick", row=1, col=5, item_name="eggs"))
obs = env.step(DarkStoreAction(action="move_picker", row=0, col=0))
obs = env.step(DarkStoreAction(action="pack", order_id="Order-1"))
obs = env.step(DarkStoreAction(action="wait"))  # packing...
obs = env.step(DarkStoreAction(action="assign_rider",
               order_id="Order-1", rider_id="Rider-A"))

print(f"Score: {env.compute_score():.3f}")
```

### HTTP API (curl)

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "single_order"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "move_picker", "row": 1, "col": 5}}'

# State
curl http://localhost:8000/state
```

---

## 🚀 Setup & Usage

### 💻 Run Locally
```bash
cd my_env
uv sync
uv run server    # → http://localhost:8000/web
```

### 🐳 Docker
```bash
docker build -t dark_store-env:latest -f server/Dockerfile .
docker run -p 8000:8000 dark_store-env:latest
```

### ☁️ Deploy to HuggingFace
```bash
openenv push
```

### 🧪 Run Inference
```bash
export HF_TOKEN="your-token"
python inference.py
```

### ✅ Validate
```bash
openenv validate
```

---

## 📈 Baseline Scores

Scores depend on the LLM model. Larger models handle the multi-step coordination better.

| Agent | single_order | concurrent_orders | full_operations | avg |
|---|---|---|---|---|
| 🤷 Do-nothing | 0.001 | 0.001 | 0.001 | 0.001 |
| 🤖 Gemma2-27B (Ollama local) | 0.936 | 0.912 | 0.001 | 0.617 |

Task 3 scored 0.001 because the model wasted ticks restocking items repeatedly instead of picking — demonstrating that the hard task genuinely challenges even capable models. A smarter agent that restocks once and moves on would score significantly higher.

The difficulty progression is clear: Task 1 is solvable by most models, Task 2 requires multi-order coordination that challenges mid-tier models, and Task 3 with stockouts/expiry/batching genuinely challenges frontier models.

---

## 📁 Project Structure

```
my_env/
├── 📄 inference.py              # Baseline inference script
├── 📄 openenv.yaml              # OpenEnv manifest
├── 📄 README.md                 # You are here
├── 📄 models.py                 # Action & Observation types
├── 📄 client.py                 # WebSocket/HTTP client
└── 📁 server/
    ├── 📄 app.py                # FastAPI + Gradio web UI
    ├── 📄 dark_store_environment.py  # 🧠 Core simulation
    └── 📄 Dockerfile            # Container build
```
