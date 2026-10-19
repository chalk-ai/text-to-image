#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the text-to-image demo (server.py + index.html) as a GPU ScalingGroup.

One replica holds SDXL with the SDXL-Lightning UNet on one GPU and serves both the page and
the /generate API, so the browser's requests are same-origin.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
Then open the printed URL and describe an image.
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = "/models"


def main() -> None:
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
        name='text-to-image',
        gpu='nvidia-l4',
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
        sg.deploy(build_timeout=2400.0, ready_timeout=1800.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
