"""Tests for the in-memory context store."""
import pytest
from datetime import datetime, timezone, timedelta

from backend.vision.context_store import ContextStore


@pytest.fixture
def store():
    return ContextStore()


@pytest.mark.asyncio
async def test_update_and_get(store):
    ctx = {"emotion": "happy", "attention": "focused"}
    await store.update("s1", ctx)
    result = await store.get("s1")
    assert result == ctx


@pytest.mark.asyncio
async def test_get_missing_session(store):
    result = await store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_or_default_returns_default(store):
    result = await store.get_or_default("nonexistent")
    assert result["emotion"] == "neutral"
    assert result["face_detected"] is False


@pytest.mark.asyncio
async def test_get_or_default_returns_stored(store):
    ctx = {"emotion": "sad", "attention": "away"}
    await store.update("s1", ctx)
    result = await store.get_or_default("s1")
    assert result["emotion"] == "sad"


@pytest.mark.asyncio
async def test_is_stale_no_context(store):
    assert await store.is_stale("nonexistent") is True


@pytest.mark.asyncio
async def test_is_stale_no_timestamp(store):
    await store.update("s1", {"emotion": "neutral"})
    assert await store.is_stale("s1") is True


@pytest.mark.asyncio
async def test_is_stale_fresh_context(store):
    now = datetime.now(timezone.utc).isoformat()
    await store.update("s1", {"emotion": "neutral", "timestamp": now})
    assert await store.is_stale("s1", max_age_seconds=10) is False


@pytest.mark.asyncio
async def test_is_stale_old_context(store):
    old = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    await store.update("s1", {"emotion": "neutral", "timestamp": old})
    assert await store.is_stale("s1", max_age_seconds=10) is True


@pytest.mark.asyncio
async def test_clear(store):
    await store.update("s1", {"emotion": "happy"})
    await store.clear("s1")
    assert await store.get("s1") is None


@pytest.mark.asyncio
async def test_empty_session_id_ignored(store):
    await store.update("", {"emotion": "happy"})
    assert await store.get("") is None
