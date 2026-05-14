import base64
import io
import asyncio
import logging
from typing import Union

import httpx
from gtts import gTTS

from backend.config import SARVAM_API_KEY, SARVAM_SPEAKER, TTS_PROVIDER

logger = logging.getLogger(__name__)

class TTSError(Exception):
    pass

class SarvamTTS:
    def __init__(self):
        self.api_key = SARVAM_API_KEY
        self.speaker = SARVAM_SPEAKER
        self.base_url = "https://api.sarvam.ai/text-to-speech"
        
    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
            
        payload = {
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": self.speaker,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.0,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }
        
        headers = {
            "api-subscription-key": self.api_key, 
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                response_json = response.json()
                
                if "audios" in response_json and len(response_json["audios"]) > 0:
                    base64_audio = response_json["audios"][0]
                    return base64.b64decode(base64_audio)
                else:
                    raise TTSError("Unexpected response format from Sarvam API")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"Sarvam API HTTP error: {e.response.status_code} - {e.response.text}")
                raise TTSError(f"Sarvam API error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"Error during Sarvam TTS synthesis: {e}")
                raise TTSError(f"Error during Sarvam TTS synthesis: {e}")


class gTTSTTS:
    def __init__(self):
        self.language = "en"
        
    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
            
        loop = asyncio.get_event_loop()
        
        def _do_synthesize():
            tts = gTTS(text=text, lang=self.language, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
            
        try:
            audio_bytes = await loop.run_in_executor(None, _do_synthesize)
            return audio_bytes
        except Exception as e:
            logger.error(f"Error during gTTS synthesis: {e}")
            raise TTSError(f"Error during gTTS synthesis: {e}")


def get_tts_provider() -> Union[SarvamTTS, gTTSTTS]:
    if TTS_PROVIDER.lower() == "gtts":
        return gTTSTTS()
    return SarvamTTS()
