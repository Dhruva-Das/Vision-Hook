import asyncio
import logging
import time
from typing import List, Dict

from backend.config import BASE_SYSTEM_PROMPT, SESSION_TTL_SECONDS
from backend.audio.stt import get_stt_provider
from backend.audio.tts import get_tts_provider
from backend.agent.llm import GroqLLMClient
from backend.agent.injector import build_system_prompt, strip_visual_context
from backend.vision.context_store import context_store

# Max conversation turns to keep in history (user + assistant = 1 turn)
MAX_HISTORY_TURNS = 50

# Timeout for a single LLM generation call
LLM_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)

class AgentPipeline:
    """
    Orchestrates a complete conversation turn.
    Maintains conversation history and ties together STT, Vision, LLM, and TTS.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stt = get_stt_provider()
        self.tts = get_tts_provider()
        self.llm = GroqLLMClient()
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_count: int = 0
        logger.info(f"Initialized new AgentPipeline for session {session_id}")

    async def process_audio(self, audio_bytes: bytes) -> bytes:
        """
        Entry point for a websocket audio chunk.
        Runs STT. If speech is detected, proceeds to process the turn.
        """
        if not audio_bytes:
            return b""

        transcript = await self.stt.transcribe(audio_bytes)

        if not transcript or not transcript.strip():
            return b""

        logger.info(f"[{self.session_id}] STT Transcript: {transcript}")
        return await self._process_turn(transcript)

    async def _process_turn(self, user_text: str) -> bytes:
        """
        The core pipeline logic.
        1. Inject visual context
        2. Query LLM (with timeout)
        3. Convert response to Speech
        """
        self.conversation_history.append({"role": "user", "content": user_text})

        context = await context_store.get_or_default(self.session_id)
        is_stale = await context_store.is_stale(self.session_id)

        system_prompt = build_system_prompt(BASE_SYSTEM_PROMPT, context, is_stale)
        logger.debug(f"[{self.session_id}] System prompt generated (visuals injected)")

        # Issue 3: Enforce a hard timeout on the LLM call so a hung Groq request
        # cannot block the WebSocket indefinitely.
        try:
            response_text = await asyncio.wait_for(
                self.llm.generate(self.conversation_history, system_prompt),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.session_id}] LLM timed out after {LLM_TIMEOUT_SECONDS}s")
            response_text = "I'm taking a moment to think. Could you repeat that?"

        logger.info(f"[{self.session_id}] LLM Response: {response_text}")
        self.conversation_history.append({"role": "assistant", "content": response_text})

        try:
            audio_bytes = await self.tts.synthesize(response_text)
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS failed: {e}")
            audio_bytes = b""

        self.turn_count += 1

        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

        return audio_bytes

    def get_transcript(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def get_turn_count(self) -> int:
        return self.turn_count

    def reset(self) -> None:
        self.conversation_history = []
        self.turn_count = 0
        logger.info(f"Pipeline reset for session {self.session_id}")


# Module-level registry with last-active timestamps for TTL-based GC (Issue 2)
_pipelines: Dict[str, AgentPipeline] = {}
_pipeline_last_active: Dict[str, float] = {}


def pipeline_exists(session_id: str) -> bool:
    """Returns True if a live pipeline exists for this session."""
    return session_id in _pipelines


def get_pipeline(session_id: str) -> AgentPipeline:
    """Retrieves an existing pipeline or creates a new one for the session."""
    if session_id not in _pipelines:
        _pipelines[session_id] = AgentPipeline(session_id)
    _pipeline_last_active[session_id] = time.monotonic()
    return _pipelines[session_id]


def destroy_pipeline(session_id: str) -> None:
    """Removes a pipeline from memory."""
    _pipelines.pop(session_id, None)
    _pipeline_last_active.pop(session_id, None)
    logger.info(f"Destroyed pipeline for session {session_id}")


def cleanup_stale_pipelines() -> int:
    """
    Evicts pipelines that have been idle longer than SESSION_TTL_SECONDS.
    Called periodically by the GC background task in main.py.
    Returns the number of sessions removed.
    """
    now = time.monotonic()
    stale = [
        sid for sid, last in list(_pipeline_last_active.items())
        if now - last > SESSION_TTL_SECONDS
    ]
    for sid in stale:
        destroy_pipeline(sid)
        logger.info(f"GC: evicted idle session {sid}")
    return len(stale)


def get_all_session_ids() -> list[str]:
    """Returns all active session IDs. Used for graceful shutdown cleanup."""
    return list(_pipelines.keys())
