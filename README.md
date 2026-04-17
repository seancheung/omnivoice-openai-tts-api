# OmniVoice OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [OmniVoice](https://github.com/k2-fsa/OmniVoice) — a multilingual (600+ languages) zero-shot TTS model with diffusion language modelling — with zero-shot voice cloning driven by files dropped into a mounted directory.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Voice cloning** — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory; the filename is the voice id
- **Voice design** — extra `POST /v1/audio/design` endpoint using OmniVoice's instruction-driven mode (e.g. `"female, british accent"`, no reference audio needed)
- **2 images** — `cuda` and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; HuggingFace cache is mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/omnivoice-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/omnivoice-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the voices directory

```
voices/
├── alice.wav     # reference audio, 16kHz+, <=20s
├── alice.txt     # UTF-8 text: the exact transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. Voices are used by the OpenAI-compatible `/v1/audio/speech` endpoint. The `/v1/audio/design` endpoint does not need the `voices/` directory.

### 2. Run the container

GPU (recommended):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/omnivoice-openai-tts-api:cuda-latest
```

CPU:

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/omnivoice-openai-tts-api:latest
```

Model weights (≈2 GB) are pulled from HuggingFace on first start. Mounting `/root/.cache` persists them across container restarts.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver (R470+); no host CUDA toolkit required.

### 3. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default.

### GET `/v1/audio/voices`

List all usable voices.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`

OpenAI TTS-compatible endpoint (voice cloning mode).

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "omnivoice",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to 8000 characters |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Playback speed, `0.25 - 4.0`, default `1.0` |
| `language` | string | Optional language name or ISO code (e.g. `en`, `zh`) — improves prosody slightly |
| `duration` | float | Optional fixed output duration in seconds (`0.5 - 120`); overrides `speed` |
| `num_step` | int | Optional diffusion steps (`4 - 64`); lower is faster |
| `guidance_scale` | float | Optional classifier-free guidance scale (`0.0 - 10.0`) |

Output audio is mono 24 kHz; `pcm` is raw s16le, matching OpenAI's default `pcm` format.

### POST `/v1/audio/design`

Non-standard endpoint that exposes OmniVoice's **voice design** mode — no reference audio needed; describe the target voice with an `instruct` string.

```bash
curl -s http://localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "British narrator here.",
    "instruct": "male, low pitch, british accent",
    "response_format": "mp3"
  }' \
  -o out_design.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `instruct` | string | Voice attributes, e.g. `"female, british accent"` — see [voice design docs](https://github.com/k2-fsa/OmniVoice/blob/main/docs/voice-design.md) for the vocabulary |
| `response_format` | string | Same as `/speech` |
| `speed` / `duration` / `language` / `num_step` / `guidance_scale` | — | Same semantics as `/speech` |

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="omnivoice",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`language`, `duration`, `num_step`, `guidance_scale`) can be passed through `extra_body={...}`.

### GET `/healthz`

Returns model name, device, dtype, sample rate and status for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | HuggingFace repo id or local path |
| `OMNIVOICE_DEVICE` | `auto` | `auto` → CUDA > MPS > CPU. Or `cuda` / `mps` / `cpu` |
| `OMNIVOICE_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `OMNIVOICE_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`; CPU auto-downgrades to `float32` |
| `OMNIVOICE_CACHE_DIR` | — | Sets `HF_HOME` before model load |
| `OMNIVOICE_VOICES_DIR` | `/voices` | Voices directory |
| `OMNIVOICE_PROMPT_CACHE_SIZE` | `32` | In-memory LRU of preprocessed voice-clone prompts (keyed by file + mtime) |
| `OMNIVOICE_NUM_STEP` | `32` | Default diffusion steps |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | Default classifier-free guidance scale |
| `OMNIVOICE_T_SHIFT` | `0.1` | |
| `OMNIVOICE_DENOISE` | `true` | |
| `OMNIVOICE_POSTPROCESS_OUTPUT` | `true` | |
| `OMNIVOICE_LAYER_PENALTY_FACTOR` | `5.0` | |
| `OMNIVOICE_POSITION_TEMPERATURE` | `5.0` | |
| `OMNIVOICE_CLASS_TEMPERATURE` | `0.0` | |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t omnivoice-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t omnivoice-openai-tts-api:cpu .
```

## Caveats

- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). OmniVoice is zero-shot; to get a stable voice under those names, just drop `alloy.wav` + `alloy.txt` into `voices/`.
- **Concurrency** — a single OmniVoice instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Long text** — requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413; OmniVoice itself splits long text into ~15-second chunks and cross-fades them internally.
- **Streaming is not supported** — the endpoint returns the complete audio when generation finishes; OmniVoice's diffusion + cross-fade post-processing is an offline pipeline.
- **Voice-clone prompt caching** — the first request for a given voice runs OmniVoice's reference preprocessing (Whisper transcription, silence trimming, etc.); subsequent requests reuse the cached prompt. The cache is invalidated automatically when the `.wav` / `.txt` mtime changes.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.
- **Pronunciation & non-verbal tags** — supported via OmniVoice's native syntax: pinyin (`这批ZHE2出售`), CMU phones (`[B EY1 S]`), `[laughter]`, `[sigh]`, etc.

## Project layout

```
.
├── OmniVoice/                 # read-only submodule, never modified
├── app/                       # FastAPI application
│   ├── server.py
│   ├── engine.py              # model loading + voice-clone prompt LRU + inference
│   ├── voices.py              # voices directory scanner
│   ├── audio.py               # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml       # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Apache 2.0).
