#!/usr/bin/env python3
"""Entrypoint to run the FastAPI app with Hypercorn.

This wrapper adds the repo root to `sys.path`, optionally forces CPU mode via
`API_FORCE_CPU`, enables faulthandler for native-crash visibility, and then
starts the server bound to `HOST`/`PORT`.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
import faulthandler
import logging


def main() -> None:
    # Make native crashes easier to diagnose
    faulthandler.enable(all_threads=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("run_api")

    log.info("Starting API runner")
    log.info("Python: %s", sys.version.replace("\n", " "))
    log.info("CWD: %s", os.getcwd())

    # Optional safety switch: set API_FORCE_CPU=1 to disable CUDA for the API process.
    # Default is to allow CUDA so inference can run on GPU when available.
    if os.getenv("API_FORCE_CPU", "").strip() in {"1", "true", "yes"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log.info("CUDA_VISIBLE_DEVICES=%r (API_FORCE_CPU=%r)", os.environ.get("CUDA_VISIBLE_DEVICES"), os.getenv("API_FORCE_CPU"))

    # Ensure repo root is on PYTHONPATH so `import api` works.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    log.info("Repo root added to sys.path: %s", repo_root)

    # Bind 0.0.0.0 so Windows can reach WSL via localhost port-forwarding
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8011"))
    log.info("HOST=%s PORT=%s", host, port)

    log.info("Importing FastAPI app (api.app)")
    from api.app import app  # noqa: WPS433
    log.info("Imported FastAPI app successfully")

    # Use Hypercorn to avoid rare native crashes seen with some Uvicorn stacks.
    log.info("Importing Hypercorn")
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    log.info("Imported Hypercorn successfully")

    cfg = Config()
    cfg.bind = [f"{host}:{port}"]
    cfg.use_reloader = False

    log.info("Starting server now (this should stay running)")
    asyncio.run(serve(app, cfg))


if __name__ == "__main__":
    main()

