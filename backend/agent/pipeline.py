import logging
from typing import List, Dict

from backend.config import BASE_SYSTEM_PROMPT
from backend.audio.stt import get_stt_provider
from backend.audio.tts import get_tts_provider
from backend.agent.llm import GroqLLMClient
from backend.agent.injector import build_system_prompt, strip_visual_context
from backend.vision.context_store import context_store

# Max conversation turns to keep in history (user + assistant = 1 turn)
MAX_HISTORY_TURNS = 50

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
        
        # If no speech was detected or transcription failed
        if not transcript or not transcript.strip():
            return b""
            
        logger.info(f"[{self.session_id}] STT Transcript: {transcript}")
        return await self._process_turn(transcript)

    async def _process_turn(self, user_text: str) -> bytes:
        """
        The core pipeline logic.
        1. Inject visual context
        2. Query LLM
        3. Convert response to Speech
        """
        # Step 1: Record user message
        self.conversation_history.append({"role": "user", "content": user_text})
        
        # Step 2: Retrieve the latest visual context
        context = await context_store.get_or_default(self.session_id)
        is_stale = await context_store.is_stale(self.session_id)
        
        # Step 3: Build the enriched system prompt
        system_prompt = build_system_prompt(BASE_SYSTEM_PROMPT, context, is_stale)
        
        # Log clean system prompt (without the massive visual block) to keep logs readable
        clean_prompt = strip_visual_context(system_prompt)
        logger.debug(f"[{self.session_id}] System prompt generated (visuals injected)")
        
        # Step 4: Generate LLM response
        response_text = await self.llm.generate(self.conversation_history, system_prompt)
        logger.info(f"[{self.session_id}] LLM Response: {response_text}")
        
        # Step 5: Record assistant message
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Step 6: Synthesize Speech
        try:
            audio_bytes = await self.tts.synthesize(response_text)
        except Exception as e:
            logger.error(f"[{self.session_id}] TTS failed: {e}")
            # Even if TTS fails, we return empty bytes so the pipeline doesn't crash the websocket
            audio_bytes = b""
            
        # Step 7: Update state
        self.turn_count += 1

        # Trim history to avoid exceeding LLM context window
        max_messages = MAX_HISTORY_TURNS * 2  # each turn is user + assistant
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

        return audio_bytes

    def get_transcript(self) -> List[Dict[str, str]]:
        """Returns the conversation history."""
        return self.conversation_history

    def get_turn_count(self) -> int:
        """Returns the number of completed turns."""
        return self.turn_count

    def reset(self) -> None:
        """Clears the history and resets the turn count."""
        self.conversation_history = []
        self.turn_count = 0
        logger.info(f"Pipeline reset for session {self.session_id}")


# Module-level registry to hold active pipelines
_pipelines: Dict[str, AgentPipeline] = {}

def get_pipeline(session_id: str) -> AgentPipeline:
    """Retrieves an existing pipeline or creates a new one for the session."""
    if session_id not in _pipelines:
        _pipelines[session_id] = AgentPipeline(session_id)
    return _pipelines[session_id]

def destroy_pipeline(session_id: str) -> None:
    """Removes a pipeline from memory."""
    if session_id in _pipelines:
        del _pipelines[session_id]
        logger.info(f"Destroyed pipeline for session {session_id}")


def get_all_session_ids() -> list[str]:
    """Returns all active session IDs. Used for graceful shutdown cleanup."""
    return list(_pipelines.keys())
