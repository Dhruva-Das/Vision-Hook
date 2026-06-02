"""Tests for the pipeline registry and AgentPipeline state management."""
import sys
import pytest
from unittest.mock import MagicMock

# Mock heavy dependencies that may not be installed in test env
for mod in ["gtts", "fer", "fer.FER", "groq", "faster_whisper"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from backend.agent.pipeline import (
    get_pipeline,
    destroy_pipeline,
    get_all_session_ids,
    _pipelines,
)


class TestPipelineRegistry:
    def setup_method(self):
        """Clear the global registry before each test."""
        _pipelines.clear()

    def test_get_creates_new_pipeline(self):
        p = get_pipeline("test-1")
        assert p is not None
        assert p.session_id == "test-1"

    def test_get_returns_same_pipeline(self):
        p1 = get_pipeline("test-1")
        p2 = get_pipeline("test-1")
        assert p1 is p2

    def test_destroy_removes_pipeline(self):
        get_pipeline("test-1")
        destroy_pipeline("test-1")
        assert "test-1" not in _pipelines

    def test_destroy_nonexistent_is_noop(self):
        destroy_pipeline("nonexistent")  # should not raise

    def test_get_all_session_ids(self):
        get_pipeline("a")
        get_pipeline("b")
        ids = get_all_session_ids()
        assert set(ids) == {"a", "b"}

    def test_transcript_starts_empty(self):
        p = get_pipeline("test-1")
        assert p.get_transcript() == []
        assert p.get_turn_count() == 0

    def test_reset_clears_state(self):
        p = get_pipeline("test-1")
        p.conversation_history.append({"role": "user", "content": "hi"})
        p.turn_count = 5
        p.reset()
        assert p.get_transcript() == []
        assert p.get_turn_count() == 0
