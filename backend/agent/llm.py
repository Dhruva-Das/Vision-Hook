import asyncio
import logging
from typing import List, Dict

from groq import AsyncGroq, APIError

from backend.config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

class GroqLLMClient:
    """
    Wrapper around the Groq API.
    Uses AsyncGroq for native non-blocking calls.
    """
    def __init__(self):
        self.client = AsyncGroq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Takes the conversation history and the injected system prompt,
        queries the Groq API, and returns the response string.
        Retries once on transient API errors before returning a fallback.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=self.max_tokens,
                    temperature=LLM_TEMPERATURE,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except APIError as e:
                if attempt == 0:
                    logger.warning(f"Groq API error on attempt 1, retrying in 1s: {e}")
                    await asyncio.sleep(1.0)
                    continue
                logger.error(f"Groq API Error during generation: {e}")
                return "I didn't quite catch that. Could you repeat?"

            except Exception as e:
                logger.error(f"Unexpected error in LLM generation: {e}")
                return "I encountered a brief issue while thinking. Could you try again?"

        return "I encountered a brief issue while thinking. Could you try again?"
