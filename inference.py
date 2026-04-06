"""
Inference Script — Dark Store Simulator
========================================
Runs all 3 tasks (single_order, concurrent_orders, full_operations) sequentially
using an OpenAI-compatible LLM to generate actions from observation text.

Environment variables:
    API_BASE_URL   - LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     - Model identifier (default: Qwen/Qwen2.5-0.5B-Instruct)
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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
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
You are an AI agent managing a quick-commerce dark store. You coordinate a picker robot, \
packing station, and delivery riders to fulfil grocery orders under tight time constraints.

WAREHOUSE: 10x8 grid with shelves at fixed positions. Packing station at (0,0). Picker starts at (0,7).
CITY: 8x8 delivery grid. Riders start at (0,0) and return there after each delivery.

ACTION SPACE — reply with exactly ONE JSON object (no markdown, no explanation):

1. move_picker: Move picker one cell/tick toward target.
   {"action": "move_picker", "row": <int>, "col": <int>}

2. pick: Pick an item from a shelf. Picker must be AT the shelf cell. Max 5 items held.
   {"action": "pick", "row": <int>, "col": <int>, "item_name": "<str>"}

3. pack: Pack a completed order at packing station (0,0). Takes 2 ticks. All items must be picked.
   {"action": "pack", "order_id": "<str>"}

4. assign_rider: Dispatch an idle rider to deliver a packed order.
   {"action": "assign_rider", "order_id": "<str>", "rider_id": "<str>"}

5. batch_delivery: Send one idle rider to deliver TWO packed orders (nearer customer first). +2.0 bonus.
   {"action": "batch_delivery", "order_a": "<str>", "order_b": "<str>", "rider_id": "<str>"}

6. restock: Emergency restock a depleted SKU. Arrives in 4 ticks. Costs -1.0.
   {"action": "restock", "item_name": "<str>", "quantity": <int>}

7. wait: Do nothing this tick.
   {"action": "wait"}

REWARDS:
  On-time delivery: +10.0 | Late delivery: -15.0 | Stockout pick: -5.0
  Picker move/tick: -0.05 | Batch bonus: +2.0 | Restock: -1.0 | Expiry: -3.0/unit

STRATEGY HINTS:
- Prioritize URGENT orders (timer < 5 ticks) — late delivery costs -15.0.
- Batch nearby deliveries when two packed orders have close customer positions.
- Restock depleted items (STOCKOUT) early so they arrive in time.
- Pick items for multiple orders in one trip when shelves are close together.
- Move picker efficiently — each move costs -0.05.
- Pack as soon as all items for an order are picked and picker is at (0,0).
- Assign riders immediately after packing to avoid timer expiry during transit.

RESPOND WITH ONLY A SINGLE JSON OBJECT. No text before or after.\
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
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            data: Dict[str, Any] = json.loads(json_str)
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
        score = max(0.0, min(1.0, score))
        success = score > 0.0

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
