"""HTTP and WebSocket layer. Kept thin: all control logic lives in ``core``."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.controllers import pid_defaults
from core.models import list_plants
from server.engine import TS, Simulation

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="Control Lab", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/plants")
def api_plants():
    plants = list_plants()
    return {
        "plants": plants,
        "ts": TS,
        "pid_defaults": {p["key"]: pid_defaults(p["key"]) for p in plants},
    }


@app.websocket("/ws")
async def session(sock: WebSocket):
    await sock.accept()
    sim = Simulation()
    await sock.send_json({"type": "init", **sim.describe()})

    last = time.perf_counter()
    try:
        while True:
            # Pause the wall clock for at most one tick while waiting for a
            # command, so a burst of slider drags cannot starve the stream.
            message = None
            try:
                message = await asyncio.wait_for(sock.receive_json(), timeout=0.02)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                return

            if message:
                action = message.get("type")
                if action == "config":
                    sim.apply_config(message.get("config") or {})
                    await sock.send_json({"type": "init", **sim.describe()})
                    last = time.perf_counter()
                    continue
                if action == "reset":
                    sim.reset()
                    last = time.perf_counter()
                    await sock.send_json({"type": "reset_ok", **sim.describe()})
                    continue
                if action == "push":
                    sim.inject_disturbance(float(message.get("amp", 1.0)))
                    continue
                if action == "speed":
                    sim.config["speed"] = float(message.get("speed", 1.0))
                    last = time.perf_counter()
                    continue
                if action == "pause":
                    sim.config["paused"] = bool(message.get("paused"))
                    last = time.perf_counter()
                    continue

            samples = []
            if not sim.config.get("paused"):
                now = time.perf_counter()
                speed = float(sim.config.get("speed", 1.0) or 1.0)
                steps = int(round((now - last) / (TS / speed)))
                if steps > 0:
                    samples = sim.advance(steps)
                    # Re-align instead of chasing a backlog: after a stall the
                    # demo drops simulated time rather than sprinting to catch up.
                    last = now

            await sock.send_json({
                "type": "samples",
                "data": samples,
                "stats": sim.stats(),
            })
    except (WebSocketDisconnect, RuntimeError):
        return
