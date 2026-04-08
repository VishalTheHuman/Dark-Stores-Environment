"""
Inference Script — Dark Store Simulator
========================================
Runs all 3 tasks (single_order, concurrent_orders, full_operations) sequentially
using an OpenAI-compatible LLM to generate actions from observation text.

Environment variables:
    API_BASE_URL   - LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     - Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       - API key for authentication

STDOUT format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import DarkStoreAction
from server.dark_store_environment import DarkStoreEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:27b")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or "ollama"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = HF_TOKEN  # OpenAI client uses this as the api_key
BENCHMARK = "dark_store"
TASKS = ["single_order", "concurrent_orders", "full_operations"]
TEMPERATURE = 0.3
MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You control a warehouse picker robot in a dark store. Each turn you output EXACTLY ONE JSON action.

=== YOUR GOAL ===
Fulfill grocery orders by: picking items from shelves → packing at station → sending riders to deliver.
You earn points for progress and lose points for mistakes.

=== READING THE OBSERVATION ===
Each turn you see the full warehouse state:
- PICKER position=(row, col) holding=[items] — where you are and what you carry (max 5 items)
- PENDING ORDERS — orders that need fulfilling. Items with ✓ are already picked. Items without ✓ still need picking.
- SHELVES — every item's location as (row,col) and stock count. STOCKOUT means stock=0.
- PACKED ORDERS — orders ready to be sent out with a rider
- RIDERS — delivery riders. "idle" means available. "delivering" means busy.
- HINT — if shown, follow it to recover from an error

=== THE 7 ACTIONS (output exactly one JSON per turn) ===

1. MOVE to a shelf or station:
   {"action": "move_picker", "row": 1, "col": 1}
   - Moves picker instantly to target. Costs -0.05 per cell of distance.
   - Target must be a shelf cell, walkway (col 0 or 9), or corridor (row 0 or 7).
   - You MUST move to a shelf BEFORE you can pick from it.

2. PICK an item from a shelf:
   {"action": "pick", "row": 1, "col": 1, "item_name": "milk"}
   - Picker MUST already be at position (1,1) to pick from shelf (1,1).
   - The item must be needed by a pending order (no ✓ next to it).
   - Earns +1.0 reward on success.
   - Fails if: wrong position, stock=0 (STOCKOUT), hands full (5 items), or item not needed.

3. PACK an order at the packing station:
   {"action": "pack", "order_id": "Order-1"}
   - Picker MUST be at position (0,0) — the packing station.
   - ALL items for the order must be picked (all have ✓).
   - Takes 2 ticks to complete. Earns +3.0 reward.
   - After packing, wait 2 ticks, then the order appears in PACKED ORDERS.

4. ASSIGN a rider to deliver a packed order:
   {"action": "assign_rider", "order_id": "Order-1", "rider_id": "Rider-A"}
   - Order must be in PACKED ORDERS (not just packed — wait for it to appear).
   - Rider must be "idle". Earns +2.0 reward.
   - Rider then travels to customer. When arrived: +10.0 if on time, -5.0 if late.

5. BATCH DELIVERY — send one rider with two orders:
   {"action": "batch_delivery", "order_a": "Order-1", "order_b": "Order-2", "rider_id": "Rider-A"}
   - Both orders must be in PACKED ORDERS. Rider must be idle.
   - Earns +2.0 bonus per order delivered. Saves a rider trip.

6. RESTOCK a depleted item:
   {"action": "restock", "item_name": "coke", "quantity": 8}
   - Use when a shelf shows STOCKOUT (stock=0) and an order needs that item.
   - Items arrive after 4 ticks. Costs -1.0. Only restock ONCE per item.

7. WAIT — do nothing for one tick:
   {"action": "wait"}
   - Use while waiting for: packing to finish, rider to deliver, or restock to arrive.

=== STEP-BY-STEP WORKFLOW FOR EACH ORDER ===

1. Read PENDING ORDERS. Find items without ✓ — those need picking.
2. For each unpicked item:
   a. Find its shelf location in SHELVES (e.g., milk → (1,1))
   b. Move there: {"action": "move_picker", "row": 1, "col": 1}
   c. Pick it: {"action": "pick", "row": 1, "col": 1, "item_name": "milk"}
3. When ALL items have ✓, move to (0,0): {"action": "move_picker", "row": 0, "col": 0}
4. Pack: {"action": "pack", "order_id": "Order-1"}
5. Wait 2 ticks: {"action": "wait"} (twice)
6. Assign rider: {"action": "assign_rider", "order_id": "Order-1", "rider_id": "Rider-A"}
7. Move on to next order or wait for delivery.

=== COMMON MISTAKES TO AVOID ===
- Do NOT try to pick before moving to the shelf — you'll get an error.
- Do NOT try to pack before moving to (0,0) — you'll get an error.
- Do NOT assign_rider right after pack — wait 2 ticks for packing to finish.
- Do NOT pick items that already have ✓ — they're already picked.
- Do NOT invent actions like "look_pending_orders" — only the 7 actions above exist.
- Do NOT restock items that are already in stock — waste of -1.0.
- If you see STOCKOUT for an item you need, restock it ONCE, then wait 4 ticks.

=== REWARDS ===
+1.0  — each item picked successfully
+3.0  — order packed
+2.0  — rider dispatched
+10.0 — order delivered on time
+2.0  — batch delivery bonus (per order)
-0.05 — per cell of picker movement
-0.1  — invalid action / error
-1.0  — restock fee
-5.0  — late delivery or undelivered order
-5.0  — trying to pick from empty shelf (STOCKOUT)

Output ONLY the JSON object. No explanation, no markdown, no text before or after.\
""")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def parse_action_from_response(text: str) -> DarkStoreAction:
    """Parse LLM response text into a DarkStoreAction.

    Attempts to extract a JSON object from the response. Falls back to wait
    if parsing fails.
    """
    import re
    original = text.strip()

    # First try: find JSON in the FULL text (including inside think tags)
    start = original.find("{")
    end = original.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = original[start : end + 1]
        # Find the LAST complete JSON object (after thinking, the action is usually last)
        # Try from the end backwards
        for i in range(len(original) - 1, -1, -1):
            if original[i] == "}":
                # Find matching opening brace
                for j in range(i, -1, -1):
                    if original[j] == "{":
                        candidate = original[j : i + 1]
                        try:
                            data: Dict[str, Any] = json.loads(candidate)
                            if "action" in data:
                                return DarkStoreAction(**data)
                        except (json.JSONDecodeError, Exception):
                            pass
                        break

    # Second try: strip think tags, then look for JSON
    text = re.sub(r"<think>.*?</think>", "", original, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            data = json.loads(json_str)
            return DarkStoreAction(**data)
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback
    return DarkStoreAction(action="wait")


def get_llm_action(
    client: OpenAI,
    observation_text: str,
    error_text: Optional[str] = None,
) -> DarkStoreAction:
    """Query the LLM for the next action given the observation text."""
    user_content = observation_text
    if error_text:
        user_content += f"\n\nLAST ACTION ERROR: {error_text}\nPlease choose a different action."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_action_from_response(response_text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return DarkStoreAction(action="wait")


def action_to_str(action: DarkStoreAction) -> str:
    """Convert a DarkStoreAction to a compact string for logging."""
    parts = [action.action]
    if action.row is not None:
        parts.append(f"row={action.row}")
    if action.col is not None:
        parts.append(f"col={action.col}")
    if action.item_name is not None:
        parts.append(f"item={action.item_name}")
    if action.order_id is not None:
        parts.append(f"order={action.order_id}")
    if action.rider_id is not None:
        parts.append(f"rider={action.rider_id}")
    if action.order_a is not None:
        parts.append(f"a={action.order_a}")
    if action.order_b is not None:
        parts.append(f"b={action.order_b}")
    if action.quantity is not None:
        parts.append(f"qty={action.quantity}")
    return "(" + ",".join(parts) + ")"


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

async def run_task(
    client: OpenAI,
    env: DarkStoreEnvironment,
    task_name: str,
) -> float:
    """Run a single task episode and return the normalized score."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task_name)
        last_error: Optional[str] = obs.error

        tick_budget = obs.tick + obs.ticks_remaining

        for step in range(1, tick_budget + 2):  # +2 for safety margin
            if obs.done:
                break

            action = get_llm_action(client, obs.text, last_error)
            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            last_error = obs.error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        score = env.compute_score()
        # Clamp to open interval (0, 1) — competition requires strictly between 0 and 1
        score = max(0.001, min(0.999, score))
        success = score > 0.001

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all 3 tasks sequentially."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Use Docker image if LOCAL_IMAGE_NAME is set, otherwise run directly
    if LOCAL_IMAGE_NAME:
        from client import DarkStoreClient
        env = await DarkStoreClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = DarkStoreEnvironment()

    scores: List[float] = []

    try:
        for task_name in TASKS:
            score = await run_task(client, env, task_name)
            scores.append(score)
    finally:
        try:
            if hasattr(env, "close") and callable(env.close):
                result = env.close()
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    if scores:
        avg_score = sum(scores) / len(scores)
        print(
            f"\n[SUMMARY] avg_score={avg_score:.3f} "
            f"scores={','.join(f'{s:.3f}' for s in scores)}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
