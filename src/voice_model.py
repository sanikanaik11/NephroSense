"""
voice_model.py
──────────────
Voice Screening is DISABLED. There is no established, peer-reviewed
research linking voice characteristics (pitch, jitter) to chronic
kidney disease specifically. Shipping this as a real feature to
patients would be presenting guesswork as science, so it is turned
off until real validated research/data exists to build it properly.
"""


def analyze_voice(audio_bytes):
    return {
        "voice_score": None,
        "note": "Voice Screening is currently disabled — no validated medical "
                 "research links voice patterns to kidney disease yet. This "
                 "feature is paused until that changes.",
    }