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
You are a warehouse robot controller. Reply with ONLY one JSON object per turn. No text.

STEP-BY-STEP PROCESS for each order:
1. Look at PENDING ORDERS to see what items are needed (items WITHOUT a checkmark ✓)
2. Look at SHELVES to find the (row,col) of the needed item
3. Use move_picker to go to that shelf: {"action": "move_picker", "row": R, "col": C}
4. Use pick to grab the item: {"action": "pick", "row": R, "col": C, "item_name": "NAME"}
5. Repeat steps 2-4 for each unpicked item in the order
6. Move to packing station: {"action": "move_picker", "row": 0, "col": 0}
7. Pack the order: {"action": "pack", "order_id": "ORDER_ID"}
8. Wait one tick for packing: {"action": "wait"}
9. Assign a rider: {"action": "assign_rider", "order_id": "ORDER_ID", "rider_id": "RIDER_ID"}
10. Start next order or wait

CRITICAL RULES:
- You can hold MAX 5 items. Only pick items that orders actually need.
- pick ONLY works when your position EXACTLY matches the shelf position
- pack ONLY works at position (0,0) when ALL items have ✓ checkmarks
- assign_rider ONLY works when order status is PACKED (wait 2 ticks after pack)
- If you get an error, READ THE HINT and follow it
- If STOCKOUT appears, use: {"action": "restock", "item_name": "NAME", "quantity": 8}

BATCH DELIVERY (bonus +2.0 per order):
When 2 orders are PACKED and customers are nearby:
{"action": "batch_delivery", "order_a": "ID1", "order_b": "ID2", "rider_id": "RIDER_ID"}

Reply with ONLY the JSON. Nothing else.\
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
