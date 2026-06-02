import os
import wave
import uuid
import tempfile
import asyncio
from typing import Union
import logging

from groq import Groq, APIError

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from backend.config import GROQ_API_KEY, STT_PROVIDER

logger = logging.getLogger(__name__)

class GroqWhisperSTT:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "whisper-large-v3"
        
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not audio_bytes:
            return ""
            
        # Create a temporary file path
        tmp_path = os.path.join(tempfile.gettempdir(), f"stt_{uuid.uuid4().hex}.wav")
        
        try:
            # Save audio bytes as a temporary .wav file
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            # Groq SDK is synchronous, so we run it in an executor
            loop = asyncio.get_running_loop()

            def _do_transcribe():
                with open(tmp_path, "rb") as f:
                    return self.client.audio.transcriptions.create(
                        file=(os.path.basename(tmp_path), f.read()),
                        model=self.model,
                        response_format="text",
                        language="en"
                    )

            # Retry once on transient API errors
            for attempt in range(2):
                try:
                    transcription = await loop.run_in_executor(None, _do_transcribe)
                    return transcription.strip()
                except APIError as e:
                    if attempt == 0:
                        logger.warning(f"Groq STT error on attempt 1, retrying in 1s: {e}")
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"Groq API Error during STT: {e}")
                    return ""

        except Exception as e:
            logger.error(f"Error during STT transcription: {e}")
            return ""
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

class LocalWhisperSTT:
    def __init__(self):
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed. Please install it to use LocalWhisperSTT.")
        logger.info("Loading Local Whisper Model (faster-whisper base)...")
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not audio_bytes:
            return ""
            
        tmp_path = os.path.join(tempfile.gettempdir(), f"stt_local_{uuid.uuid4().hex}.wav")
        
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
                
            loop = asyncio.get_running_loop()
            
            def _do_transcribe():
                segments, _ = self.model.transcribe(tmp_path, beam_size=5, language="en")
                return " ".join([segment.text for segment in segments])
                
            transcription = await loop.run_in_executor(None, _do_transcribe)
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error during local STT transcription: {e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

def get_stt_provider() -> Union[GroqWhisperSTT, LocalWhisperSTT]:
    if STT_PROVIDER.lower() == "local":
        return LocalWhisperSTT()
    return GroqWhisperSTT()
