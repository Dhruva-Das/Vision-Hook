import uuid
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import load_config, STT_PROVIDER, TTS_PROVIDER, LLM_MODEL
from backend.vision.frame_analyzer import analyze_frame
from backend.vision.context_store import context_store
from backend.agent.pipeline import get_pipeline, destroy_pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Validate environment variables on startup
load_config()

app = FastAPI(title="Vision Voice Agent")

# Allow CORS for the frontend to hit the API without issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/session/start")
async def start_session():
    """Generates a new session ID and initializes its pipeline."""
    session_id = str(uuid.uuid4())
    get_pipeline(session_id)
    logger.info(f"Started new session: {session_id}")
    return {"session_id": session_id}


@app.post("/analyze-frame/{session_id}")
async def process_frame(session_id: str, request: Request):
    """
    Receives raw JPEG/PNG bytes in the request body.
    Runs FER and updates the shared context store.
    """
    image_bytes = await request.body()
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty frame body")
        
    result = analyze_frame(image_bytes)
    
    # Store it so the AgentPipeline can read it on the next turn
    await context_store.update(session_id, result)
    
    return {
        "status": "ok", 
        "emotion": result["emotion"], 
        "attention": result["attention"],
        "confidence": result["confidence_score"]
    }


@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: str):
    """
    Persistent WebSocket for the audio stream.
    Receives audio chunks, passes them to the pipeline, and sends TTS audio back.
    """
    await websocket.accept()
    pipeline = get_pipeline(session_id)
    
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            # Receive raw PCM/WAV bytes from the browser microphone
            audio_bytes = await websocket.receive_bytes()
            
            # Process the turn (STT -> Context Inject -> LLM -> TTS)
            tts_audio = await pipeline.process_audio(audio_bytes)
            
            # If the LLM spoke, send the synthesized audio bytes back to the browser
            if tts_audio:
                await websocket.send_bytes(tts_audio)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.get("/context/{session_id}")
async def get_context(session_id: str) -> Dict[str, Any]:
    """Returns the latest visual context dict for a session."""
    context = await context_store.get(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found for session")
    return context


@app.get("/transcript/{session_id}")
async def get_transcript(session_id: str) -> Dict[str, Any]:
    """Returns the conversation history and turn count."""
    pipeline = get_pipeline(session_id)
    return {
        "turns": pipeline.get_transcript(),
        "turn_count": pipeline.get_turn_count()
    }


@app.post("/session/end/{session_id}")
async def end_session(session_id: str):
    """Cleans up the pipeline and context store, returns final transcript."""
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
            "llm": LLM_MODEL
        }
    }
