"""Entry point: ``python run.py`` starts the server and opens the browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Control Lab - interactive control playground")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"Control Lab running at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
