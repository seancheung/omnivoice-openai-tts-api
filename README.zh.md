# OmniVoice OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 [OmniVoice](https://github.com/k2-fsa/OmniVoice)（支持 600+ 语言的零样本 TTS 扩散语言模型）进行封装，支持从挂载目录零样本克隆音色。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致
- **音色克隆**：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对，文件名即音色 id
- **音色设计**：额外提供 `POST /v1/audio/design`，使用 OmniVoice 的指令式生成（如 `"female, british accent"`，无需参考音频）
- **2 个镜像**：`cuda` 与 `cpu`
- **模型运行时下载**：不打包进镜像，HuggingFace 缓存目录挂载后可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/omnivoice-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/omnivoice-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备音色目录

```
voices/
├── alice.wav     # 参考音频，16kHz 以上，<=20s
├── alice.txt     # UTF-8 纯文本，内容为 alice.wav 中说出的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。音色目录仅服务于 OpenAI 兼容的 `/v1/audio/speech` 端点；`/v1/audio/design` 端点不需要 `voices/`。

### 2. 运行容器

GPU 版本（推荐）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/omnivoice-openai-tts-api:cuda-latest
```

CPU 版本：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/omnivoice-openai-tts-api:latest
```

首次启动会从 HuggingFace 下载模型权重（约 2 GB）。挂载 `/root/.cache/huggingface` 可让权重在容器重启后复用。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。

### 3. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。

### GET `/v1/audio/voices`

列出所有可用音色。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`

OpenAI TTS 兼容接口（语音克隆模式）。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "omnivoice",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 8000 字符 |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 语速，范围 `0.25 - 4.0`，默认 `1.0` |
| `language` | string | 可选语言名或 ISO 代码（如 `en` / `zh`），对韵律略有提升 |
| `duration` | float | 可选，固定时长（秒，`0.5 - 120`），会覆盖 `speed` |
| `num_step` | int | 可选扩散步数（`4 - 64`），越低越快 |
| `guidance_scale` | float | 可选 classifier-free guidance 强度（`0.0 - 10.0`） |

输出音频为单声道 24 kHz；`pcm` 为裸的 s16le 数据，与 OpenAI 默认 `pcm` 格式一致。

### POST `/v1/audio/design`

非标准端点，暴露 OmniVoice 的 **voice design**（语音设计）模式——无需参考音频，通过 `instruct` 字符串描述目标音色。

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

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `instruct` | string | 音色属性，如 `"female, british accent"`，属性词表见 [voice design 文档](https://github.com/k2-fsa/OmniVoice/blob/main/docs/voice-design.md) |
| `response_format` | string | 同 `/speech` |
| `speed` / `duration` / `language` / `num_step` / `guidance_scale` | — | 语义同 `/speech` |

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="omnivoice",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`language`、`duration`、`num_step`、`guidance_scale` 等扩展字段可通过 `extra_body={...}` 传入。

### GET `/healthz`

返回模型名、设备、dtype、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | HuggingFace 仓库 id 或本地路径 |
| `OMNIVOICE_DEVICE` | `auto` | `auto` 按 CUDA > MPS > CPU 优先级。也可强制 `cuda` / `mps` / `cpu` |
| `OMNIVOICE_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `OMNIVOICE_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`，CPU 自动降级为 `float32` |
| `OMNIVOICE_CACHE_DIR` | — | 加载模型前写入 `HF_HOME` |
| `OMNIVOICE_VOICES_DIR` | `/voices` | 音色目录 |
| `OMNIVOICE_PROMPT_CACHE_SIZE` | `32` | voice-clone prompt 内存 LRU 大小（key 含文件 mtime） |
| `OMNIVOICE_NUM_STEP` | `32` | 默认扩散步数 |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | 默认 classifier-free guidance 强度 |
| `OMNIVOICE_T_SHIFT` | `0.1` | |
| `OMNIVOICE_DENOISE` | `true` | |
| `OMNIVOICE_POSTPROCESS_OUTPUT` | `true` | |
| `OMNIVOICE_LAYER_PENALTY_FACTOR` | `5.0` | |
| `OMNIVOICE_POSITION_TEMPERATURE` | `5.0` | |
| `OMNIVOICE_CLASS_TEMPERATURE` | `0.0` | |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t omnivoice-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t omnivoice-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）。OmniVoice 本身是零样本，没有内置音色；若想通过这些名字调用稳定的声音，直接在 `voices/` 放同名 `.wav` + `.txt` 即可。
- **并发**：OmniVoice 模型在单实例下非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413；OmniVoice 内部按约 15 秒切块并 cross-fade 拼接。
- **不支持流式返回**：生成完成后一次性返回；OmniVoice 的扩散 + cross-fade 后处理是离线管线。
- **Voice-clone prompt 缓存**：首次请求某个音色时会跑 OmniVoice 的参考音频预处理（Whisper 转录、去静音等），之后复用缓存；文件 mtime 变化自动失效。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。
- **发音与非语言标签**：沿用 OmniVoice 原生语法——拼音（`这批ZHE2出售`）、CMU 音标（`[B EY1 S]`）、`[laughter]` / `[sigh]` 等。

## 目录结构

```
.
├── OmniVoice/                 # 只读 submodule，不修改
├── app/                       # FastAPI 应用
│   ├── server.py
│   ├── engine.py              # 模型加载 + voice-clone prompt LRU + 推理
│   ├── voices.py              # 音色扫描
│   ├── audio.py               # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml       # cuda + cpu 矩阵构建
├── voices/                    # 运行时挂载
└── README.md
```

## 致谢

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)（Apache 2.0）。
