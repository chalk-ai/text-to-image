#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Generate an image with the deployed demo from the command line.

./generate.py https://<web_url> "a lighthouse on a cliff at sunrise, oil painting" [-o out.webp] [--seed 42]
"""

from __future__ import annotations

import argparse
import base64
import sys
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("prompt")
    parser.add_argument("-o", "--out", default="out.webp")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    res = httpx.post(
        args.url.rstrip("/") + "/generate",
        json={"prompt": args.prompt, "seed": args.seed},
        timeout=120,
    )
    body = res.json()
    if res.status_code != 200:
        sys.exit(f"{res.status_code}: {body.get('error')}")
    with open(args.out, "wb") as f:
        f.write(base64.b64decode(body["image"].split(",", 1)[1]))
    total_ms = (time.perf_counter() - t0) * 1000
    print(
        f"wrote {args.out} ({body['width']}x{body['height']}, seed {body['seed']}): "
        + f"{body['infer_ms']} ms on the GPU, {body['queue_ms']} ms queued, {total_ms:.0f} ms round trip"
    )


if __name__ == "__main__":
    main()
