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

        # Retry once on transient network/API errors
        for attempt in range(2):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    response_json = response.json()

                    if "audios" in response_json and len(response_json["audios"]) > 0:
                        return base64.b64decode(response_json["audios"][0])
                    raise TTSError("Unexpected response format from Sarvam API")

            except TTSError:
                raise
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt == 0:
                    logger.warning(f"Sarvam TTS error on attempt 1, retrying in 1s: {e}")
                    await asyncio.sleep(1.0)
                    continue
                logger.error(f"Sarvam API error: {e}")
                raise TTSError(f"Sarvam API error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during Sarvam TTS synthesis: {e}")
                raise TTSError(f"Error during Sarvam TTS synthesis: {e}")


class gTTSTTS:
    def __init__(self):
        self.language = "en"
        
    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
            
        loop = asyncio.get_running_loop()
        
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
