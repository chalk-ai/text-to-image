#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the text-to-image demo (server.py + index.html) as a GPU ScalingGroup.

One replica holds SDXL with the SDXL-Lightning UNet on one GPU and serves both the page and
the /generate API, so the browser's requests are same-origin.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
Then open the printed URL and describe an image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = "/models"


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="text-to-image")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        # Start from PyTorch's CUDA runtime image rather than pip-installing torch: the image
        # builder hashes every file a build step writes, and CUDA torch's ~6 GB of wheels
        # take it past its 15-minute limit. Base-image layers are reused as-is.
        Image.base("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
        .pip_install(
            [
                "diffusers==0.41.0",
                "transformers==5.19.0",
                "accelerate==1.15.0",
                "safetensors==0.8.0",
                "huggingface-hub==1.33.0",
                "pillow==12.3.0",
                "fastapi==0.143.0",
                "uvicorn[standard]==0.54.0",
            ]
        )
        .env({"T2I_MODELS_DIR": MODELS_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_file(str(HERE / "download_weights.py"), "/app/download_weights.py", strategy="copy")
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
        env={"PORT": "8080"},
        # Weights are fetched at startup rather than baked in: 7 GB of layers plus CUDA torch
        # does not build inside the image builder's 15-minute limit.
        entrypoint=["bash", "-c", "python /app/download_weights.py && exec python /app/server.py"],
        port=8080,
        # The server only listens once the model is loaded; without an HTTP probe the replica
        # reports ready while the weights are still downloading.
        readiness_probe=ReadinessProbe.http("/status", period_seconds=5),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
