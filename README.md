# Text to image on Chalk

Describe a picture and get a 1024×1024 image back in under two seconds: 1.2 s of that is the GPU
on an NVIDIA L4, safety check included. The page, the API and the model are
served from a single GPU replica deployed with [`chalkcompute`](https://pypi.org/project/chalkcompute/).
The model is [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with ByteDance's
[SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) 4-step UNet, which produces a
finished image in four denoising steps instead of the usual 30 to 50.

```
prompt ──POST /generate──▶ GPU replica: prompt filter → SDXL-Lightning (4 steps) → safety check ──▶ WebP
```

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image (torch + diffusers) and deploys a `chalkcompute.ScalingGroup` on one GPU. |
| `download_weights.py` | Downloads exactly the 7 GB of weights the server loads. The replica runs it at startup. |
| `server.py` | FastAPI app in the container: `GET /` serves the page, `POST /generate` returns an image. |
| `index.html` | The page: a prompt box, example prompts, the result with its timing, and recent images. |
| `generate.py` | Calls `/generate` from the command line and saves the image. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule GPU workloads.
Credentials come from your existing `chalk login`.

```bash
./deploy.py
```

The first deploy builds a multi-gigabyte image, then waits for a GPU node, so expect about 15 minutes.
Later deploys reuse the cached torch layer. When it finishes, the script prints a
`web_url`. Open it and describe an image.

From the command line:

```bash
./generate.py https://<web_url> "a lighthouse on a cliff at sunrise, oil painting" -o lighthouse.webp
```

## API

`POST /generate` with `{"prompt": "...", "seed": 42}` (`seed` is optional).

| Status | Body |
| --- | --- |
| 200 | `{"image": "data:image/webp;base64,...", "seed", "width", "height", "steps", "queue_ms", "infer_ms"}` |
| 400 | `{"error"}`: the prompt is missing, longer than 300 characters, or refused by the prompt filter |
| 422 | `{"error"}`: the image was flagged by the safety classifier and is not returned |
| 429 | `{"error"}`: the replica is over its generation budget (default 40 a minute) |
| 503 | `{"error"}`: more than 8 requests are already waiting for the GPU |

`GET /status` reports the model, step count and current queue depth.

## Guardrails

The URL is public, so the server defends itself:

- **Prompt filter.** A short pattern list refuses obviously disallowed subjects before they reach the GPU.
- **Image safety check.** Every image is scored by
  [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) and
  withheld when it is flagged.
- **Budget and bounded queue.** The replica admits at most `T2I_RATE_LIMIT_PER_MINUTE` generations a
  minute (default 40) and lets at most `T2I_MAX_QUEUE` requests wait for the GPU (default 8). The
  budget is shared rather than per visitor, because the scaling-group gateway forwards only internal
  load-balancer addresses in `X-Forwarded-For`, so the server never sees a trustworthy client IP.

Neither filter is perfect. Tighten them, or add `authenticated=True` to the `ScalingGroup`, before
pointing real traffic at your own deployment.

## Notes

- **Weights.** Only the fp16 text encoders, the SDXL-Lightning UNet, an fp16-safe VAE and the safety
  classifier are downloaded, about 7 GB. The stock SDXL UNet and full-precision files are skipped.
  The replica fetches them at startup, which takes under a minute. Baking them into the image
  instead pushes the build past the image builder's 15-minute limit.
- **fp16 VAE.** The stock SDXL VAE produces NaNs in fp16, so the server uses
  [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
- **Base image.** The image starts from `pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime` instead of
  pip-installing CUDA torch. The Chalk image builder hashes every file a build step writes, and
  torch's ~6 GB of CUDA wheels push a build past its 15-minute limit; base-image layers are reused
  as-is. CUDA 12.6 runs on any 560+ NVIDIA driver.
- **Readiness.** `deploy.py` sets an HTTP readiness probe on `/status`, so traffic reaches the
  replica only after the weights are downloaded and the model is warm.
- **Clean up.** The demo keeps one GPU replica running. Delete it when you are done:

  ```bash
  uv run --with chalkcompute python -c \
    "from chalkcompute import ScalingGroup; ScalingGroup.from_name('text-to-image').delete()"
  ```

## License

The code is Apache-2.0. SDXL base 1.0 and SDXL-Lightning are released under the CreativeML
OpenRAIL++-M license, whose use restrictions apply to the images you generate. The fp16 VAE is MIT
and the safety classifier is Apache-2.0.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
