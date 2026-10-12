"""Text-to-image with SDXL-Lightning, served from one GPU.

Runs inside the container image that deploy.py builds for a Chalk ScalingGroup; the page at
`/` is index.html.

API:
  POST /generate  {"prompt": str, "seed": int | null}
    200 {"image": "data:image/webp;base64,...", "seed", "width", "height", "steps",
         "queue_ms", "infer_ms"}
    400 {"error"}   prompt missing, too long, or refused by the prompt filter
    422 {"error"}   the image was flagged by the safety classifier and is not returned
    429 {"error"}   the demo is over its per-minute generation budget
    503 {"error"}   the queue is full
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from safetensors.torch import load_file
from transformers import pipeline as hf_pipeline

MODELS_DIR = Path(os.environ.get("T2I_MODELS_DIR", "/models"))
BASE_MODEL = MODELS_DIR / "sdxl-base"
LIGHTNING_UNET = MODELS_DIR / "sdxl-lightning" / "sdxl_lightning_4step_unet.safetensors"
VAE_MODEL = MODELS_DIR / "sdxl-vae-fp16-fix"
SAFETY_MODEL = MODELS_DIR / "nsfw-image-detection"
STEPS = int(os.environ.get("T2I_STEPS", "4"))
SIZE = int(os.environ.get("T2I_SIZE", "1024"))
PORT = int(os.environ.get("PORT", "8080"))
MAX_PROMPT_CHARS = 300
# Requests beyond this many waiting for the GPU get a 503 instead of an unbounded wait.
MAX_QUEUE = int(os.environ.get("T2I_MAX_QUEUE", "8"))
# A budget for the whole replica, not per visitor: the scaling-group gateway forwards only
# internal load-balancer addresses in X-Forwarded-For, so there is no trustworthy client IP to
# key on. One GPU finishes an image about every 1.2 s, so 40 a minute keeps the queue short.
RATE_LIMIT = int(os.environ.get("T2I_RATE_LIMIT_PER_MINUTE", "40"))
NSFW_THRESHOLD = 0.5
INDEX_HTML = Path(__file__).with_name("index.html")

# A coarse first line of defense; the image classifier below is the real check.
BLOCKED_PROMPT = re.compile(
    r"\b(nude|nudity|naked|nsfw|porn\w*|sex\w*|erotic\w*|hentai|topless|genitals?|"
    r"gore|beheading|dismember\w*|child\s*abuse|loli\w*|underage)\b",
    re.IGNORECASE,
)

log = logging.getLogger("t2i")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Generator:
    """SDXL with the SDXL-Lightning 4-step UNet, plus an NSFW classifier for its outputs."""

    def __init__(self) -> None:
        super().__init__()
        t0 = time.monotonic()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE_MODEL, subfolder="unet"))
        unet.load_state_dict(load_file(LIGHTNING_UNET, device="cpu"))
        unet = unet.to(self.device, dtype)
        # The stock SDXL VAE overflows in fp16; this one is fine-tuned to decode in fp16.
        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL,
            unet=unet,
            vae=vae,
            torch_dtype=dtype,
            # Only the fp16 files are baked into the image; on CPU they are upcast to fp32.
            variant="fp16",
        ).to(self.device)
        # SDXL-Lightning is distilled for trailing timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.safety = hf_pipeline(
            "image-classification", model=str(SAFETY_MODEL), device=0 if self.device == "cuda" else -1
        )
        self.generate("a lighthouse on a cliff at sunrise", 0)
        log.info("loaded on %s in %.1fs", self.device, time.monotonic() - t0)

    @torch.inference_mode()
    def generate(self, prompt: str, seed: int) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=SIZE,
            height=SIZE,
            generator=g,
        )
        return out.images[0]

    def nsfw_score(self, image: Image.Image) -> float:
        scores = {r["label"]: r["score"] for r in self.safety(image)}
        return float(scores.get("nsfw", 0.0))


class RateLimiter:
    """At most `per_minute` admissions in any sliding one-minute window."""

    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.hits: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self.hits and now - self.hits[0] > 60:
            self.hits.popleft()
        if len(self.hits) >= self.per_minute:
            return False
        self.hits.append(now)
        return True


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn accepts no connections until this returns, so the replica only goes ready
    # once both models are resident and warm.
    state["gen"] = await asyncio.to_thread(Generator)
    state["gpu"] = threading.Lock()
    state["waiting"] = 0
    state["limiter"] = RateLimiter(RATE_LIMIT)
    yield


app = FastAPI(lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "model": "ByteDance/SDXL-Lightning", "steps": STEPS, "queue": state["waiting"]})


@app.post("/generate")
async def generate(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return error(400, "Expected a JSON body.")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return error(400, "Describe the image you want.")
    if len(prompt) > MAX_PROMPT_CHARS:
        return error(400, f"Keep the description under {MAX_PROMPT_CHARS} characters.")
    if BLOCKED_PROMPT.search(prompt):
        return error(400, "That description isn't allowed in this demo. Try something else.")
    if not state["limiter"].allow():
        return error(429, "The demo is at capacity right now. Try again in a minute.")
    if state["waiting"] >= MAX_QUEUE:
        return error(503, "The GPU is busy. Try again in a few seconds.")

    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, int) else random.randrange(2**31)
    gen: Generator = state["gen"]
    gpu: threading.Lock = state["gpu"]

    def run() -> tuple[Image.Image, float, float, float]:
        t_req = time.perf_counter()
        with gpu:
            t_start = time.perf_counter()
            image = gen.generate(prompt, seed)
            score = gen.nsfw_score(image)
            t_end = time.perf_counter()
        return image, score, (t_start - t_req) * 1000, (t_end - t_start) * 1000

    state["waiting"] += 1
    try:
        image, score, queue_ms, infer_ms = await asyncio.to_thread(run)
    finally:
        state["waiting"] -= 1
    log.info("seed=%d nsfw=%.3f queue=%.0fms infer=%.0fms", seed, score, queue_ms, infer_ms)
    if score > NSFW_THRESHOLD:
        return error(422, "The safety check flagged that image, so it isn't shown. Try another description.")

    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=90)
    return JSONResponse(
        {
            "image": "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(),
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "steps": STEPS,
            "queue_ms": round(queue_ms),
            "infer_ms": round(infer_ms),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
