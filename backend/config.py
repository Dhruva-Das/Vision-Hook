import os
from dotenv import load_dotenv

# Initialize variables to avoid import errors before load_config is called, 
# although typically we rely on load_dotenv happening early.
# We will just load dotenv right here so it's populated on import.
load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
SARVAM_SPEAKER: str = os.getenv("SARVAM_SPEAKER", "meera")

STT_PROVIDER: str = os.getenv("STT_PROVIDER", "groq")
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "sarvam")

FRAME_ANALYSIS_INTERVAL: int = int(os.getenv("FRAME_ANALYSIS_INTERVAL", "3"))
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Session idle TTL before garbage collection (seconds)
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", str(30 * 60)))

# Auth — set to empty string to disable API key checking (dev mode)
API_KEY: str = os.getenv("API_KEY", "")

# Rate limiting
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "60/minute")

# Request size limits (bytes)
MAX_FRAME_SIZE: int = int(os.getenv("MAX_FRAME_SIZE", str(5 * 1024 * 1024)))  # 5 MB

# Allowed CORS origins — comma-separated, or "*" for dev
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

BASE_SYSTEM_PROMPT: str = """You are a helpful and perceptive AI assistant. 
You answer questions clearly and concisely. 
You are speaking directly to a user through a voice interface."""

def load_config() -> None:
    """Validates that required keys exist."""
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
        
    if TTS_PROVIDER.lower() == "sarvam" and not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is required when TTS_PROVIDER is set to 'sarvam'.")

# Run validation on import if desired, but we can also just let main.py call it.
