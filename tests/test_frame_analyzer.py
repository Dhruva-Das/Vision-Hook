"""Tests for the frame analyzer module."""
import numpy as np
import pytest

try:
    import cv2
    # Trigger the same import chain as the actual module
    from backend.vision.frame_analyzer import (
        decode_frame,
        analyze_frame,
        assess_attention,
        FrameDecodeError,
    )
    HAS_DEPS = True
except (ImportError, Exception) as e:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="OpenCV / FER / TensorFlow not available")


def make_jpeg_bytes(width=100, height=100, color=(128, 128, 128)):
    """Helper to create valid JPEG bytes from a solid-color frame."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


class TestDecodeFrame:
    def test_valid_jpeg(self):
        raw = make_jpeg_bytes()
        frame = decode_frame(raw)
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # BGR

    def test_invalid_bytes_raises(self):
        with pytest.raises(FrameDecodeError):
            decode_frame(b"not-an-image")

    def test_empty_bytes_raises(self):
        with pytest.raises(FrameDecodeError):
            decode_frame(b"")


class TestAnalyzeFrame:
    def test_empty_bytes_returns_default(self):
        result = analyze_frame(b"")
        assert result["emotion"] == "unknown"
        assert result["attention"] == "away"
        assert result["face_detected"] is False
        assert "timestamp" in result

    def test_valid_frame_returns_dict(self):
        raw = make_jpeg_bytes()
        result = analyze_frame(raw)
        assert "emotion" in result
        assert "attention" in result
        assert "confidence_score" in result
        assert "face_detected" in result
        assert "timestamp" in result

    def test_corrupt_bytes_returns_default(self):
        result = analyze_frame(b"corrupt-data-here")
        assert result["emotion"] == "unknown"
        assert result["attention"] == "away"


class TestAssessAttention:
    def test_no_face_returns_away(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotion = {"face_detected": False, "dominant": "unknown", "confidence": 0.0}
        assert assess_attention(frame, emotion) == "away"

    def test_happy_high_confidence_returns_focused(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotion = {"face_detected": True, "dominant": "happy", "confidence": 0.8}
        assert assess_attention(frame, emotion) == "focused"

    def test_fear_returns_confused(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotion = {"face_detected": True, "dominant": "fear", "confidence": 0.5}
        assert assess_attention(frame, emotion) == "confused"

    def test_sad_returns_disengaged(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotion = {"face_detected": True, "dominant": "sad", "confidence": 0.5}
        assert assess_attention(frame, emotion) == "disengaged"

    def test_low_confidence_returns_uncertain(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotion = {"face_detected": True, "dominant": "angry", "confidence": 0.2}
        assert assess_attention(frame, emotion) == "uncertain"
