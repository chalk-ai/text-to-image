"""Downloads exactly the weights server.py loads into T2I_MODELS_DIR (default /models).

The replica runs this before starting the server. About 7 GB takes 30 to 60 seconds with
HF_XET_HIGH_PERFORMANCE, which is faster than pulling the same bytes as image layers, and keeps
the image build inside the builder's time limit. Files already present are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = Path(os.environ.get("T2I_MODELS_DIR", "/models"))

# SDXL's own UNet and full-precision weights are skipped: the UNet comes from SDXL-Lightning
# and the VAE from the fp16-fixed checkpoint, which together save about 12 GB.
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=MODELS_DIR / "sdxl-base",
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "vae/config.json",
    ],
)
hf_hub_download(
    "ByteDance/SDXL-Lightning",
    "sdxl_lightning_4step_unet.safetensors",
    local_dir=MODELS_DIR / "sdxl-lightning",
)
snapshot_download(
    "madebyollin/sdxl-vae-fp16-fix",
    local_dir=MODELS_DIR / "sdxl-vae-fp16-fix",
    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
)
snapshot_download(
    "Falconsai/nsfw_image_detection",
    local_dir=MODELS_DIR / "nsfw-image-detection",
    allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
)
