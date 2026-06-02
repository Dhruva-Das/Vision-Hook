import cv2
import numpy as np
import logging
from datetime import datetime, timezone
try:
    from fer import FER
except ImportError:
    from fer.fer import FER

logger = logging.getLogger(__name__)

class FrameDecodeError(Exception):
    pass

# Initialize FER detector globally so we don't recreate it every frame.
# MTCNN is set to False because it is too slow for real-time 3s polling on CPU.
try:
    # This might print TensorFlow warnings to stdout depending on the environment,
    # but FER will function properly.
    emotion_detector = FER(mtcnn=False)
except Exception as e:
    logger.error(f"Failed to initialize FER detector: {e}")
    emotion_detector = None

# Initialize Haar cascade globally for basic face presence detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    logger.error(f"Failed to load Haar cascade: {e}")
    face_cascade = None


def decode_frame(image_bytes: bytes) -> np.ndarray:
    """Takes raw bytes (JPEG/PNG) and returns an OpenCV BGR numpy array."""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameDecodeError("cv2.imdecode returned None. Bytes may be corrupted or unsupported format.")
        return frame
    except Exception as e:
        raise FrameDecodeError(f"Failed to decode frame: {e}")


def detect_face_presence(frame: np.ndarray) -> bool:
    """Fallback fast face detection to see if the user is even looking at the screen."""
    if face_cascade is None or face_cascade.empty():
        return False
        
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return len(faces) > 0
    except Exception as e:
        logger.error(f"Error in fast face detection: {e}")
        return False


def analyze_emotions(frame: np.ndarray) -> dict:
    """Runs FER on the frame to extract emotion scores for the primary face."""
    if emotion_detector is None:
        return {"dominant": "unknown", "scores": {}, "confidence": 0.0, "face_detected": False}
        
    try:
        results = emotion_detector.detect_emotions(frame)
        
        if not results:
            return {"dominant": "unknown", "scores": {}, "confidence": 0.0, "face_detected": False}
            
        # Take the first detected face
        first_face = results[0]
        scores = first_face.get("emotions", {})
        
        if not scores:
            return {"dominant": "unknown", "scores": {}, "confidence": 0.0, "face_detected": True}
            
        dominant_emotion = max(scores, key=scores.get)
        confidence = scores[dominant_emotion]
        
        return {
            "dominant": dominant_emotion,
            "scores": scores,
            "confidence": float(confidence),
            "face_detected": True
        }
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return {"dominant": "unknown", "scores": {}, "confidence": 0.0, "face_detected": False}


def assess_attention(frame: np.ndarray, emotion_result: dict) -> str:
    """Maps the emotion and face presence data to a high-level attention string."""
    face_detected = emotion_result.get("face_detected", False)
    
    # If FER didn't find a face, use Haar cascade as a quick double-check
    if not face_detected:
        face_detected = detect_face_presence(frame)
        
    if not face_detected:
        return "away"
        
    dominant = emotion_result.get("dominant", "unknown")
    confidence = emotion_result.get("confidence", 0.0)
    
    if dominant in ["neutral", "happy"] and confidence > 0.6:
        return "focused"
    elif dominant in ["fear", "surprise"]:
        return "confused"
    elif dominant == "sad":
        return "disengaged"
    elif confidence < 0.3:
        return "uncertain"
    else:
        return "present"


def analyze_frame(image_bytes: bytes) -> dict:
    """
    Main entry point. Coordinates decoding, emotion analysis, and attention assessment.
    Never raises exceptions — always returns a safe dictionary fallback.
    """
    default_result = {
        "emotion": "unknown",
        "emotion_scores": {},
        "attention": "away",
        "confidence_score": 0.0,
        "face_detected": False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if not image_bytes:
        return default_result
        
    try:
        frame = decode_frame(image_bytes)
        emotion_result = analyze_emotions(frame)
        attention = assess_attention(frame, emotion_result)
        
        # Determine final face detection state
        face_detected = emotion_result.get("face_detected", False) or (attention != "away")
        
        return {
            "emotion": emotion_result.get("dominant", "unknown"),
            "emotion_scores": emotion_result.get("scores", {}),
            "attention": attention,
            "confidence_score": emotion_result.get("confidence", 0.0),
            "face_detected": face_detected,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except FrameDecodeError as e:
        logger.warning(f"Frame analysis bypassed due to decode error: {e}")
        return default_result
    except Exception as e:
        logger.error(f"Unexpected error in frame pipeline: {e}")
        return default_result
