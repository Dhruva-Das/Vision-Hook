import asyncio
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class ContextStore:
    """
    In-memory store that holds the latest visual context for each session.
    Acts as the bridge between the vision processing pipeline and the LLM agent pipeline.
    Uses asyncio.Lock to ensure thread-safe reads and writes.
    """
    def __init__(self):
        self._store: dict[str, dict] = {}
        # Ensure lock is created lazily to bind to the correct event loop
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def update(self, session_id: str, context: dict) -> None:
        """Stores the latest visual context for a given session."""
        if not session_id:
            return
            
        lock = self._get_lock()
        async with lock:
            self._store[session_id] = context
            logger.debug(f"Context updated for session {session_id}")

    async def get(self, session_id: str) -> dict | None:
        """Retrieves the latest context for a session, returning None if not found."""
        if not session_id:
            return None
            
        lock = self._get_lock()
        async with lock:
            return self._store.get(session_id, None)

    async def get_or_default(self, session_id: str) -> dict:
        """Retrieves the context, returning a neutral default if none exists."""
        context = await self.get(session_id)
        if context is not None:
            return context
            
        return {
            "emotion": "neutral", 
            "attention": "present", 
            "confidence_score": 0.5, 
            "face_detected": False,
            # We don't provide a timestamp here so that it immediately registers as stale if checked
        }

    async def is_stale(self, session_id: str, max_age_seconds: int = 10) -> bool:
        """
        Checks if the stored context's timestamp is older than max_age_seconds.
        Used to downweight or ignore visual context if the user's camera feed drops.
        """
        context = await self.get(session_id)
        if not context:
            return True
            
        timestamp_str = context.get("timestamp")
        if not timestamp_str:
            return True
            
        try:
            # Parse the ISO format timestamp
            context_time = datetime.fromisoformat(timestamp_str)
            now = datetime.now(timezone.utc)
            age = (now - context_time).total_seconds()
            
            return age > max_age_seconds
        except ValueError as e:
            logger.warning(f"Failed to parse context timestamp '{timestamp_str}': {e}")
            return True

    async def clear(self, session_id: str) -> None:
        """Deletes the context for a session. Should be called on session end."""
        if not session_id:
            return
            
        lock = self._get_lock()
        async with lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.debug(f"Context cleared for session {session_id}")


# Singleton instance imported by both pipelines
context_store = ContextStore()
