"""Capture deterministic screenshots of the running demo via the Chrome DevTools Protocol.

Headless Chrome renders the real page, waits real wall-clock time so the
WebSocket simulation actually accumulates data, then writes a PNG. This is the
script that produced every image in ``docs/``.

Example::

    python tools/screenshot.py --out docs/shot.png \
        --actions "click:#reset|wait:12"

Actions are separated by ``|`` and may be ``wait:<seconds>``, ``click:<css>``,
``js:<expression>``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google/Chrome/Application/chrome.exe"),
]


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("Chrome not found")


def launch(port: int, chrome: str, profile: str) -> subprocess.Popen:
    return subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}",
         "--hide-scrollbars", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def wait_for_devtools(port: int, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as fh:
                pages = json.load(fh)
            for page in pages:
                if page.get("type") == "page":
                    return page
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit("DevTools endpoint never came up")


class Session:
    def __init__(self, ws):
        self.ws = ws
        self.next_id = 1

    async def send(self, method: str, params: dict | None = None):
        msg_id = self.next_id
        self.next_id += 1
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = json.loads(await self.ws.recv())
            if raw.get("id") == msg_id:
                if "error" in raw:
                    raise RuntimeError(f"{method}: {raw['error']}")
                return raw.get("result", {})

    async def evaluate(self, expression: str):
        result = await self.send("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
        })
        return result.get("result", {}).get("value")


async def capture(ws_url: str, url: str, out: str, width: int, height: int, actions: str):
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        s = Session(ws)
        await s.send("Page.enable")
        await s.send("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False,
        })
        await s.send("Page.navigate", {"url": url})
        await asyncio.sleep(2.0)

        for action in [a for a in actions.split("|") if a]:
            kind, _, value = action.partition(":")
            if kind == "wait":
                await asyncio.sleep(float(value))
            elif kind == "click":
                await s.evaluate(f"document.querySelector({json.dumps(value)}).click()")
                await asyncio.sleep(0.2)
            elif kind == "js":
                print("js ->", await s.evaluate(value))
            else:
                raise SystemExit(f"unknown action {action}")

        shot = await s.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        with open(out, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--actions", default="wait:10")
    args = ap.parse_args()

    chrome = find_chrome()
    profile = tempfile.mkdtemp(prefix="clshot-")
    proc = launch(args.port, chrome, profile)
    try:
        page = wait_for_devtools(args.port)
        asyncio.run(capture(page["webSocketDebuggerUrl"], args.url, args.out,
                            args.width, args.height, args.actions))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
