import logging
from typing import List, Dict

from groq import AsyncGroq, APIError

from backend.config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

class GroqLLMClient:
    """
    Wrapper around the Groq API.
    Uses AsyncGroq for native non-blocking calls.
    """
    def __init__(self):
        # We assume the config validation has ensured GROQ_API_KEY exists
        self.client = AsyncGroq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Takes the conversation history and the injected system prompt,
        queries the Groq API, and returns the response string.
        """
        # Prepend the system prompt to the messages
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            # Extract and return the assistant's reply
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except APIError as e:
            logger.error(f"Groq API Error during generation: {e}")
            # Fallback response so the conversation doesn't completely die on a transient network error
            return "I didn't quite catch that. Could you repeat?"
        except Exception as e:
            logger.error(f"Unexpected error in LLM generation: {e}")
            return "I encountered a brief issue while thinking. Could you try again?"
