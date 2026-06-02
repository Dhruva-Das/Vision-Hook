import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import (
    load_config, STT_PROVIDER, TTS_PROVIDER, LLM_MODEL,
    API_KEY, RATE_LIMIT, MAX_FRAME_SIZE, CORS_ORIGINS,
)
from backend.vision.frame_analyzer import analyze_frame
from backend.vision.context_store import context_store
from backend.agent.pipeline import (
    get_pipeline, destroy_pipeline, get_all_session_ids,
    pipeline_exists, cleanup_stale_pipelines,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Validate environment variables on startup
load_config()

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


async def _session_gc_task():
    """Background task: evict pipelines idle longer than SESSION_TTL_SECONDS."""
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        removed = cleanup_stale_pipelines()
        if removed:
            logger.info(f"Session GC: evicted {removed} idle session(s)")


# --- Lifespan (startup / shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    gc_task = asyncio.create_task(_session_gc_task())
    logger.info("Vision Voice Agent starting up")
    yield
    # Cancel GC task
    gc_task.cancel()
    try:
        await gc_task
    except asyncio.CancelledError:
        pass
    # Shutdown: clean up every active pipeline and context
    logger.info("Shutting down — cleaning up active sessions")
    for sid in get_all_session_ids():
        destroy_pipeline(sid)
        await context_store.clear(sid)
    logger.info("Shutdown complete")


app = FastAPI(title="Vision Voice Agent", lifespan=lifespan)
app.state.limiter = limiter


# --- Error handler for rate limits ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down."},
    )


# --- CORS ---
_origins = [o.strip() for o in CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth Middleware ---
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if not API_KEY:
        return await call_next(request)

    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


# --- Routes ---

@app.post("/session/start")
@limiter.limit(RATE_LIMIT)
async def start_session(request: Request):
    """Generates a new session ID and initializes its pipeline."""
    session_id = str(uuid.uuid4())
    get_pipeline(session_id)
    logger.info(f"Started new session: {session_id}")
    return {"session_id": session_id}


@app.post("/analyze-frame/{session_id}")
@limiter.limit(RATE_LIMIT)
async def process_frame(session_id: str, request: Request):
    """
    Receives raw JPEG/PNG bytes in the request body.
    Runs FER and updates the shared context store.
    """
    # Issue 7: guard against frames arriving for sessions that no longer exist
    if not pipeline_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame body")

    if len(image_bytes) > MAX_FRAME_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Frame too large ({len(image_bytes)} bytes). Max is {MAX_FRAME_SIZE}.",
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_frame, image_bytes)

    await context_store.update(session_id, result)

    return {
        "status": "ok",
        "emotion": result["emotion"],
        "attention": result["attention"],
        "confidence": result["confidence_score"],
    }


@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """
    Persistent WebSocket for the audio stream.
    Receives audio chunks, passes them to the pipeline, and sends TTS audio back.

    Issue 1: API key is validated via the first text message sent by the client
    ({"type": "auth", "key": "<api_key>"}), not via a URL query parameter.
    This keeps the key out of server access logs and browser history.
    """
    await websocket.accept()

    if API_KEY:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            if auth_msg.get("type") != "auth" or auth_msg.get("key") != API_KEY:
                await websocket.close(code=4401, reason="Invalid or missing API key")
                return
        except asyncio.TimeoutError:
            await websocket.close(code=4401, reason="Auth timeout")
            return
        except Exception:
            await websocket.close(code=4401, reason="Auth error")
            return

    pipeline = get_pipeline(session_id)
    logger.info(f"WebSocket connected for session: {session_id}")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            tts_audio = await pipeline.process_audio(audio_bytes)
            if tts_audio:
                await websocket.send_bytes(tts_audio)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        destroy_pipeline(session_id)
        await context_store.clear(session_id)


@app.get("/context/{session_id}")
@limiter.limit(RATE_LIMIT)
async def get_context(request: Request, session_id: str) -> Dict[str, Any]:
    """Returns the latest visual context dict for a session."""
    context = await context_store.get(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found for session")
    return context


@app.get("/transcript/{session_id}")
@limiter.limit(RATE_LIMIT)
async def get_transcript(request: Request, session_id: str) -> Dict[str, Any]:
    """Returns the conversation history and turn count."""
    # Issue 7: get_pipeline silently creates a new empty pipeline if the session
    # doesn't exist, so we must check first to avoid returning a ghost transcript.
    if not pipeline_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline = get_pipeline(session_id)
    return {
        "turns": pipeline.get_transcript(),
        "turn_count": pipeline.get_turn_count(),
    }


@app.post("/session/end/{session_id}")
@limiter.limit(RATE_LIMIT)
async def end_session(request: Request, session_id: str):
    """Cleans up the pipeline and context store, returns final transcript."""
    if not pipeline_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = get_pipeline(session_id)
    transcript = pipeline.get_transcript()

    destroy_pipeline(session_id)
    await context_store.clear(session_id)

    logger.info(f"Ended session: {session_id}")
    return {"status": "ended", "final_transcript": transcript}


@app.get("/health")
async def health_check():
    """Returns the server status and configured providers."""
    return {
        "status": "ok",
        "providers": {
            "stt": STT_PROVIDER,
            "tts": TTS_PROVIDER,
            "llm": LLM_MODEL,
        },
    }
