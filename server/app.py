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

# Enable the Gradio web interface at /web
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

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
