# Vision Hook

A real-time voice AI agent that watches your facial expressions through the webcam and adapts its responses to your emotional state. Speak to it naturally — it sees if you're confused, distracted, or engaged and adjusts its tone accordingly.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## How it works

```
Webcam → Emotion Detection (FER) ──┐
                                    ▼
Microphone → Speech-to-Text → LLM (Groq) → Text-to-Speech → Speaker
                                    ▲
                          Visual context injected
                          into the system prompt
```

Every 3 seconds a video frame is analyzed for facial emotion (happy, sad, neutral, fear, surprise, angry). The result is injected into the LLM's system prompt so it can adapt — slowing down if you look confused, re-engaging if you look distracted, or responding warmly if you look upset.

---

## Features

- **Emotion-aware responses** — LLM behavior changes based on detected attention and affect
- **Real-time voice** — AudioWorklet-based VAD (voice activity detection) with no push-to-talk
- **Pluggable providers** — swap STT/TTS/LLM via environment variables, no code changes
- **Session isolation** — each browser tab gets its own pipeline, history, and visual context
- **Auto-reconnect** — WebSocket reconnects with exponential backoff on network drops
- **Rate limiting** — per-IP request throttling via slowapi
- **TTL-based GC** — idle sessions are evicted automatically after 30 minutes
- **Zero-dependency frontend** — single HTML file, no build step, no npm

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| STT | Groq Whisper (`whisper-large-v3`) |
| TTS | Sarvam AI (`bulbul:v1`) or gTTS |
| Vision | OpenCV + FER (FER2013) |
| Frontend | Vanilla JS + AudioWorklet |
| Container | Docker + docker-compose |

---

## Quick start

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works)
- A [Sarvam AI key](https://www.sarvam.ai) (free tier works) — or use `TTS_PROVIDER=gtts` for no-key TTS

### 1. Clone and install

```bash
git clone https://github.com/your-username/vision-hook.git
cd vision-hook
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
SARVAM_API_KEY=...       # leave empty if using TTS_PROVIDER=gtts
```

### 3. Run

```bash
uvicorn backend.main:app --reload
```

Open `frontend/index.html` in a browser (served from a local server or directly from disk), click **Start Session**, and allow camera + microphone access.

---

## Docker

```bash
cp .env.example .env  # fill in keys first
docker-compose up --build
```

The API is available at `http://localhost:8000`. Serve `frontend/index.html` from any static file server pointing at the same host.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key (used for both LLM and STT) |
| `SARVAM_API_KEY` | — | Required when `TTS_PROVIDER=sarvam` |
| `SARVAM_SPEAKER` | `meera` | Sarvam voice name |
| `STT_PROVIDER` | `groq` | `groq` or `local` (requires faster-whisper) |
| `TTS_PROVIDER` | `sarvam` | `sarvam` or `gtts` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Any Groq-supported model ID |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM response |
| `LLM_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `SESSION_TTL_SECONDS` | `1800` | Idle session eviction timeout (30 min) |
| `API_KEY` | _(empty)_ | If set, all requests require `X-API-Key: <value>` |
| `RATE_LIMIT` | `60/minute` | Per-IP rate limit (slowapi format) |
| `MAX_FRAME_SIZE` | `5242880` | Max video frame upload size in bytes (5 MB) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins (set explicitly in production) |

### Using the optional local STT provider

Uncomment `faster-whisper` in `requirements.txt`, install it, then set:

```env
STT_PROVIDER=local
```

This runs Whisper locally on CPU with no API calls — useful for offline or privacy-sensitive deployments.

---

## API reference

All HTTP endpoints accept `X-API-Key: <value>` when `API_KEY` is configured.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session/start` | Create a new session. Returns `{"session_id": "..."}` |
| `POST` | `/analyze-frame/{session_id}` | Send raw JPEG/PNG bytes; returns detected emotion |
| `WS` | `/ws/audio/{session_id}` | Bidirectional audio stream (see below) |
| `GET` | `/transcript/{session_id}` | Conversation history + turn count |
| `GET` | `/context/{session_id}` | Latest visual context for a session |
| `POST` | `/session/end/{session_id}` | Teardown session, returns final transcript |
| `GET` | `/health` | Server status + active provider names |

Interactive docs available at `http://localhost:8000/docs` when the server is running.

### WebSocket protocol

```
Client → Server:  first message must be text JSON: {"type": "auth", "key": "<API_KEY>"}
                  (omit if API_KEY is not configured)
Client → Server:  subsequent messages are raw PCM-16 audio bytes (16 kHz, mono)
Server → Client:  audio bytes (WAV or MP3 depending on TTS provider)
```

---

## Project structure

```
├── backend/
│   ├── config.py              # Environment config + validation
│   ├── main.py                # FastAPI app, routes, WebSocket handler
│   ├── agent/
│   │   ├── injector.py        # Visual context → LLM prompt injection
│   │   ├── llm.py             # Groq LLM wrapper
│   │   └── pipeline.py        # Session orchestration (STT → LLM → TTS)
│   ├── audio/
│   │   ├── stt.py             # Groq Whisper + optional local Whisper
│   │   └── tts.py             # Sarvam AI + gTTS
│   └── vision/
│       ├── context_store.py   # In-memory per-session visual context
│       └── frame_analyzer.py  # Emotion detection (FER + OpenCV)
├── frontend/
│   └── index.html             # Single-file web UI (no build step)
├── tests/                     # pytest + pytest-asyncio unit tests
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Running tests

```bash
pytest
```

---

## Deployment notes

- Set `CORS_ORIGINS` to your frontend's exact origin (e.g. `https://yourdomain.com`) — never leave it as `*` in production.
- Set `API_KEY` to a strong random secret to protect all endpoints.
- Run behind a reverse proxy (nginx, Caddy) that terminates TLS — the WebSocket must be `wss://` in production.
- The vision context store is in-memory. Sessions do not survive a server restart. For multi-replica deployments, migrate the context store to Redis.

---

## License

MIT
