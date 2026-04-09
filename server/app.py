# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Dark Store Environment.

This module creates an HTTP server that exposes the DarkStoreEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os

# Disable the default Gradio web interface at /web
os.environ["ENABLE_WEB_INTERFACE"] = "false"

# Point the web interface to our README for the sidebar (strip frontmatter)
_readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
if os.path.exists(_readme_path):
    _content = open(_readme_path, encoding="utf-8").read()
    # Strip YAML frontmatter (--- ... ---) so it doesn't render as text
    if _content.startswith("---"):
        _end = _content.find("---", 3)
        if _end != -1:
            _content = _content[_end + 3:].lstrip("\n")
    # Write stripped version to a temp location
    _stripped_path = os.path.join(os.path.dirname(__file__), ".readme_web.md")
    with open(_stripped_path, "w", encoding="utf-8") as _f:
        _f.write(_content)
    os.environ.setdefault("ENV_README_PATH", _stripped_path)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DarkStoreAction, DarkStoreObservation
    from .dark_store_environment import DarkStoreEnvironment
except (ModuleNotFoundError, ImportError):
    from models import DarkStoreAction, DarkStoreObservation
    from server.dark_store_environment import DarkStoreEnvironment


# Create the app with web interface and README integration
app = create_app(
    DarkStoreEnvironment,
    DarkStoreAction,
    DarkStoreObservation,
    env_name="dark_store",
    max_concurrent_envs=1,
)

# Custom Gradio Dashboard matching /ui
import gradio as gr
from server.gradio_ui import demo
app = gr.mount_gradio_app(app, demo, path="/web")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/web")

from fastapi import WebSocket, WebSocketDisconnect
from openai import OpenAI
import asyncio

@app.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket):
    await websocket.accept()
    from custom_inference import get_llm_action, action_to_str, SYSTEM_PROMPT
    import custom_inference as inference
    env = DarkStoreEnvironment()
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "START":
                api_key = data.get("api_key")
                base_url = data.get("base_url")
                model = data.get("model")
                task = data.get("task", "single_order")
                
                client = OpenAI(base_url=base_url, api_key=api_key)
                inference.MODEL_NAME = model  # Ensure internal consistency
                
                obs = env.reset(task=task)
                last_error = obs.error
                
                await websocket.send_json({"type": "state", "observation": obs.dict()})
                await websocket.send_json({"type": "log", "message": f"[START] task={task} model={model}"})
                
                tick_budget = obs.tick + obs.ticks_remaining
                
                for step in range(1, tick_budget + 2):
                    if obs.done:
                        break
                    
                    await websocket.send_json({"type": "log", "message": f"⏳ [AGENT THINKING...] Step {step}"})
                    action = await asyncio.to_thread(get_llm_action, client, obs.text, last_error)
                    
                    await websocket.send_json({"type": "log", "message": f"🚀 [ACTION] {action_to_str(action)}"})
                    
                    obs = env.step(action)
                    last_error = obs.error
                    
                    await websocket.send_json({"type": "state", "observation": obs.dict()})
                    await asyncio.sleep(0.5)
                    
                    if obs.done:
                        break
                        
                score = env.compute_score()
                await websocket.send_json({"type": "log", "message": f"🏁 [END] Simulation completed. Score: {score:.3f}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "log", "message": f"❌ [ERROR] {str(e)}"})
        except:
            pass



def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m my_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port != 8000:
        main(port=args.port)
    else:
        main()
