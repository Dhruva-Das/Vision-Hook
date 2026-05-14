import re

def format_emotion_description(context: dict) -> str:
    """Converts the raw context dict into natural language for the LLM."""
    face_detected = context.get("face_detected", False)
    attention = context.get("attention", "away")
    
    if not face_detected or attention == "away":
        return "The user's face is not currently visible — they may have looked away or stepped back."
        
    emotion = context.get("emotion", "neutral")
    confidence = context.get("confidence_score", 0.0)
    conf_pct = int(confidence * 100)
    
    if attention == "focused" and emotion in ["neutral", "happy"]:
        return "The user appears focused and engaged."
        
    if attention == "confused":
        return f"The user appears confused or uncertain ({emotion} expression, {conf_pct}% confidence)."
        
    return f"The user appears {attention} (dominant expression: {emotion}, {conf_pct}% confidence)."


def should_adapt_response(context: dict) -> tuple[bool, str]:
    """Decides whether the agent should actively adapt its behavior."""
    emotion = context.get("emotion", "unknown")
    attention = context.get("attention", "present")
    confidence = context.get("confidence_score", 0.0)
    
    if emotion in ["fear", "surprise"] and confidence > 0.5:
        return True, "user_confused"
        
    if attention == "away":
        return True, "user_distracted"
        
    if emotion == "sad" and confidence > 0.6:
        return True, "user_disengaged"
        
    return False, "normal"


def get_adaptation_instruction(reason: str) -> str:
    """Maps the adaptation reason to a specific LLM instruction."""
    if reason == "user_confused":
        return "The user looks confused. Slow down slightly, and check if they need clarification."
    elif reason == "user_distracted":
        return "The user seems distracted. Keep your response brief, and try to re-engage them with a direct question."
    elif reason == "user_disengaged":
        return "The user seems disengaged or down. Acknowledge them warmly and offer a different approach or topic."
    return ""


def build_system_prompt(base_prompt: str, context: dict, is_stale: bool = False) -> str:
    """Assembles the final system prompt sent to the LLM."""
    desc = format_emotion_description(context)
    
    should_adapt, reason = should_adapt_response(context)
    instruction = get_adaptation_instruction(reason) if should_adapt else ""
    
    stale_note = ""
    if is_stale:
        stale_note = "\n(Note: The visual context feed is currently paused or outdated.)"
        
    # We use a specific delimiter block so we can easily strip it out later for logging
    injected_block = f"""
---
[VISUAL CONTEXT — updated via webcam]
{desc}
{instruction}{stale_note}

Use this context subtly — do not say "I can see you look confused."
Instead, naturally adapt your tone, pacing, and level of detail based on their state.
---
"""
    return f"{base_prompt.strip()}\n{injected_block}"


def strip_visual_context(system_prompt: str) -> str:
    """Removes the visual context block from the system prompt for clean logging."""
    pattern = r"\n---\n\[VISUAL CONTEXT.*?---\n"
    return re.sub(pattern, "", system_prompt, flags=re.DOTALL)
