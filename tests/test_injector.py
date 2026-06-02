"""Tests for the system prompt injector."""
from backend.agent.injector import (
    format_emotion_description,
    should_adapt_response,
    get_adaptation_instruction,
    build_system_prompt,
    strip_visual_context,
)


class TestFormatEmotionDescription:
    def test_no_face_detected(self):
        ctx = {"face_detected": False, "attention": "away"}
        result = format_emotion_description(ctx)
        assert "not currently visible" in result

    def test_attention_away(self):
        ctx = {"face_detected": True, "attention": "away"}
        result = format_emotion_description(ctx)
        assert "not currently visible" in result

    def test_focused_and_happy(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "happy", "confidence_score": 0.8}
        result = format_emotion_description(ctx)
        assert "focused and engaged" in result

    def test_confused(self):
        ctx = {"face_detected": True, "attention": "confused", "emotion": "fear", "confidence_score": 0.7}
        result = format_emotion_description(ctx)
        assert "confused" in result

    def test_generic_present(self):
        ctx = {"face_detected": True, "attention": "present", "emotion": "angry", "confidence_score": 0.6}
        result = format_emotion_description(ctx)
        assert "present" in result
        assert "angry" in result


class TestShouldAdaptResponse:
    def test_fear_high_confidence(self):
        ctx = {"emotion": "fear", "confidence_score": 0.7, "attention": "present"}
        adapt, reason = should_adapt_response(ctx)
        assert adapt is True
        assert reason == "user_confused"

    def test_attention_away(self):
        ctx = {"emotion": "neutral", "confidence_score": 0.5, "attention": "away"}
        adapt, reason = should_adapt_response(ctx)
        assert adapt is True
        assert reason == "user_distracted"

    def test_sad_high_confidence(self):
        ctx = {"emotion": "sad", "confidence_score": 0.8, "attention": "present"}
        adapt, reason = should_adapt_response(ctx)
        assert adapt is True
        assert reason == "user_disengaged"

    def test_normal_state(self):
        ctx = {"emotion": "neutral", "confidence_score": 0.5, "attention": "present"}
        adapt, reason = should_adapt_response(ctx)
        assert adapt is False
        assert reason == "normal"


class TestGetAdaptationInstruction:
    def test_confused(self):
        assert "clarification" in get_adaptation_instruction("user_confused")

    def test_distracted(self):
        assert "brief" in get_adaptation_instruction("user_distracted")

    def test_disengaged(self):
        assert "warmly" in get_adaptation_instruction("user_disengaged")

    def test_normal_returns_empty(self):
        assert get_adaptation_instruction("normal") == ""


class TestBuildSystemPrompt:
    def test_contains_base_prompt(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "neutral", "confidence_score": 0.8}
        result = build_system_prompt("Base prompt here.", ctx)
        assert "Base prompt here." in result

    def test_contains_visual_context_block(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "neutral", "confidence_score": 0.8}
        result = build_system_prompt("Base.", ctx)
        assert "[VISUAL CONTEXT" in result

    def test_stale_note_added(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "neutral", "confidence_score": 0.8}
        result = build_system_prompt("Base.", ctx, is_stale=True)
        assert "paused or outdated" in result

    def test_stale_note_absent_when_fresh(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "neutral", "confidence_score": 0.8}
        result = build_system_prompt("Base.", ctx, is_stale=False)
        assert "paused or outdated" not in result


class TestStripVisualContext:
    def test_strips_visual_block(self):
        ctx = {"face_detected": True, "attention": "focused", "emotion": "neutral", "confidence_score": 0.8}
        prompt = build_system_prompt("Base prompt.", ctx)
        stripped = strip_visual_context(prompt)
        assert "[VISUAL CONTEXT" not in stripped
        assert "Base prompt." in stripped
