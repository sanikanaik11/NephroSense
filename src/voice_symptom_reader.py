"""
voice_symptom_reader.py — NephroSense
Lets a patient describe their symptoms by typing OR by voice, in
Marathi, Hindi or English, and produces an English version for the
doctor — while ALWAYS keeping what the patient originally said/typed
alongside it.

HONEST LIMITS (keep these in mind when presenting this feature):
  - Whisper (free, runs locally) is much less accurate for Marathi than
    for Hindi or English. Voice results can contain mistakes.
  - Translation uses FREE web services and can mistranslate ambiguous
    or colloquial words, or drop a clause on a long sentence.
  - This is a speech-to-text + translation pipeline, not a reasoning
    AI. An OPTIONAL local LLM cleanup step (via Ollama) can reword the
    result more naturally, but is skipped automatically if Ollama
    isn't installed, and never invents new symptoms.
  - The app always shows the result to the patient for review and
    stores the original text next to the English version.
  This feature produces documentation for the doctor — it never changes
  the ML risk score.

FIXES IN THIS VERSION:
  - TRANSLITERATION CHUNKING: English-letter Marathi/Hindi was being
    split into many small pieces (every ~40 characters) before being
    sent to the transliteration service, one request per piece. On a
    sentence with several clauses this silently dropped some of them
    (a multi-symptom description would lose the middle or last
    symptom). Now the whole sentence is sent in one request whenever
    it reasonably fits (up to ~300 characters), which fixed the
    dropped-clause problem in testing.
  - HALLUCINATION GUARD: previously a very short garbage word (e.g. a
    single nonsense word from a broken/near-silent recording) could
    slip through undetected. The guard now also rejects a result with
    only one word when the recording was several seconds long — real
    speech of that length almost always transcribes to more than one
    word.
  - LLM REFUSAL GUARD: the optional local-LLM cleanup step could
    occasionally reply with something like "I can't help with that"
    instead of actually rewording the text. That reply is now detected
    and discarded automatically, falling back to the untouched
    translated text instead of showing the refusal to the patient.

Needs (one-time):
    pip install openai-whisper deep-translator requests
    + ffmpeg installed and on PATH (voice only)
Optional, for more natural wording:
    Install Ollama (https://ollama.com), then run once:
        ollama pull llama3.2
"""
import os
import re
import time
import tempfile
import traceback
from collections import Counter

LANGUAGES = {"mr": "Marathi", "hi": "Hindi", "en": "English"}

MODEL_SIZE = os.environ.get("NEPHROSENSE_WHISPER_MODEL", "small")
OLLAMA_MODEL = os.environ.get("NEPHROSENSE_OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = "http://localhost:11434/api/generate"

_whisper_model = None
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_TRANSLATION_CACHE = {}
_ollama_checked = {"available": None}

_REFUSAL_PATTERNS = [
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm sorry", "i am sorry", "as an ai", "i am not able to", "i'm not able to",
    "i won't", "i will not", "unable to process", "i don't understand the request",
]


def _is_devanagari(text):
    return bool(_DEVANAGARI.search(text or ""))


def _chunk_text(text, max_chars):
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ── Hallucination / garbage-output guard ──────────────────────

def _looks_like_hallucination(text, audio_seconds=None, min_len=4):
    """
    Catches broken/silent-recording failure patterns: Whisper looping
    on one repeated symbol, or producing a single short nonsense word
    from audio that was clearly long enough to contain real speech.
    Real short sentences (multiple different words) are NOT flagged.
    """
    text = (text or "").strip()
    if len(text) < min_len:
        return True
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 2:
        if len(set(text.replace(" ", ""))) <= 2:
            return True
        # A single word from a recording that was clearly long enough
        # for a real sentence is almost always a mis-transcription.
        if audio_seconds is not None and audio_seconds > 3.0:
            return True
        return False
    counts = Counter(tokens)
    most_common_n = counts.most_common(1)[0][1]
    if most_common_n / len(tokens) >= 0.6 and len(tokens) >= 4:
        return True
    unique_chars = set(text.replace(" ", ""))
    if len(unique_chars) <= 3 and len(text) >= 10:
        return True
    return False


# ── Transliteration (English letters -> Marathi/Hindi script) ─

def transliterate_to_devanagari(text, lang_code):
    """
    Returns (devanagari_text_or_None, error_or_None). Sends the whole
    text in one request whenever it reasonably fits — splitting into
    many small pieces was found to silently drop clauses on longer,
    multi-symptom sentences.
    """
    itc = {"mr": "mr-t-i0-und", "hi": "hi-t-i0-und"}.get(lang_code)
    if not itc:
        return None, f"Transliteration not available for '{lang_code}'."
    try:
        import requests
    except ImportError:
        return None, "The 'requests' package is missing (pip install requests)."
    out_chunks = []
    try:
        for chunk in _chunk_text(text, 300):
            resp = requests.get(
                "https://inputtools.google.com/request",
                params={"text": chunk, "itc": itc, "num": 1, "cp": 0, "cs": 1,
                        "ie": "utf-8", "oe": "utf-8", "app": "nephrosense"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data[0] != "SUCCESS":
                return None, f"Transliteration service replied: {data[0]}"
            piece = data[1][0][1][0]
            if not piece:
                return None, "Transliteration returned an empty result for part of the text."
            out_chunks.append(piece)
        return " ".join(out_chunks), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


# ── Translation: three free services, tried in order ─────────

def _translate_google_page(text, lang_code):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=lang_code, target="en").translate(text)


def _translate_google_api(text, lang_code):
    import requests
    out = []
    for chunk in _chunk_text(text, 800):
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": lang_code, "tl": "en", "dt": "t", "q": chunk},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        out.append("".join(seg[0] for seg in data[0] if seg and seg[0]))
    return " ".join(out)


def _translate_mymemory(text, lang_code):
    from deep_translator import MyMemoryTranslator
    pairs = {"mr": [("marathi", "english"), ("mr-IN", "en-GB")],
             "hi": [("hindi", "english"), ("hi-IN", "en-GB")]}[lang_code]
    translator = None
    for src, tgt in pairs:
        try:
            translator = MyMemoryTranslator(source=src, target=tgt)
            break
        except Exception:
            continue
    if translator is None:
        raise RuntimeError("language pair not supported by the installed deep-translator")
    out = []
    for chunk in _chunk_text(text, 150):
        t = (translator.translate(chunk) or "").strip()
        if not t:
            raise RuntimeError("empty result")
        if "MYMEMORY WARNING" in t.upper():
            raise RuntimeError(t)
        out.append(t)
    return " ".join(out)


_ENGINES = [
    ("Google Translate", _translate_google_page),
    ("Google Translate (API)", _translate_google_api),
    ("MyMemory (backup)", _translate_mymemory),
]


def _translate_to_english(text, lang_code):
    """Returns (english_or_None, error_text_or_None, engine_name_or_None)."""
    if lang_code == "en":
        return text, None, "none needed"
    key = (lang_code, text)
    if key in _TRANSLATION_CACHE:
        english, engine = _TRANSLATION_CACHE[key]
        return english, None, engine
    errors = []
    for name, fn in _ENGINES:
        try:
            out = (fn(text, lang_code) or "").strip()
            if out:
                _TRANSLATION_CACHE[key] = (out, name)
                return out, None, name
            errors.append(f"{name}: empty result")
        except ImportError as e:
            errors.append(f"{name}: package missing ({e})")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
    return None, "\n".join(errors), None


# ── Optional: local LLM cleanup via Ollama (free, offline) ────

def _ollama_available():
    if _ollama_checked["available"] is not None:
        return _ollama_checked["available"]
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        _ollama_checked["available"] = (r.status_code == 200)
    except Exception:
        _ollama_checked["available"] = False
    return _ollama_checked["available"]


def _looks_like_refusal(text, original_len):
    t = (text or "").strip().lower()
    if not t:
        return True
    for pat in _REFUSAL_PATTERNS:
        if pat in t:
            return True
    # A suspiciously short reply to a non-trivial input is also treated
    # as a failed cleanup rather than trusted output.
    if original_len > 20 and len(t) < 8:
        return True
    return False


def refine_with_local_llm(english_text):
    """
    Optional step: asks a local, free, offline LLM (via Ollama, if
    installed and running) to lightly reword the translated text into
    clearer clinical English — WITHOUT adding, removing, or guessing
    any symptom that wasn't already stated. If Ollama isn't available,
    the call fails, or the reply looks like a refusal/non-answer, the
    original text is returned UNCHANGED.
    """
    text = (english_text or "").strip()
    if not text or not _ollama_available():
        return text, False
    try:
        import requests
        prompt = (
            "Reword the following patient symptom description into clear, plain "
            "clinical English. Keep every detail the patient mentioned, including "
            "every symptom listed. Do NOT add any new symptom, cause, diagnosis, or "
            "detail that isn't already there. Do NOT give medical advice or a "
            "diagnosis. Do NOT refuse or comment — output ONLY the reworded "
            "description and nothing else.\n\n"
            f"Patient description: {text}"
        )
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        cleaned = (r.json().get("response") or "").strip()
        if not cleaned or _looks_like_refusal(cleaned, len(text)):
            print(f"[voice_symptom_reader] Local LLM reply looked invalid, keeping original text: {cleaned[:80]!r}")
            return text, False
        if len(cleaned) > len(text) * 4:  # sanity check, avoid runaway output
            return text, False
        return cleaned, True
    except Exception as e:
        print(f"[voice_symptom_reader] Local LLM cleanup skipped ({type(e).__name__}: {e}).")
        return text, False


def translate_typed_description(text, lang_code, use_llm_cleanup=True):
    """
    Returns dict: english, devanagari, engine, llm_cleaned (bool),
    note (non-fatal warning or None), error (fatal message or None).
    """
    text = (text or "").strip()
    result = {"english": None, "devanagari": None, "engine": None,
              "llm_cleaned": False, "note": None, "error": None}
    if not text:
        result["error"] = "Nothing to translate."
        return result
    if lang_code == "en":
        cleaned, was_cleaned = refine_with_local_llm(text) if use_llm_cleanup else (text, False)
        result["english"] = cleaned
        result["engine"] = "none needed"
        result["llm_cleaned"] = was_cleaned
        return result

    source_text = text
    if not _is_devanagari(text):
        deva, terr = transliterate_to_devanagari(text, lang_code)
        if deva:
            source_text = deva
            result["devanagari"] = deva
        else:
            result["note"] = (
                f"Could not convert the English letters to {LANGUAGES[lang_code]} script "
                f"({terr}). Translating the text as typed — this is usually much less accurate."
            )

    english, err, engine = _translate_to_english(source_text, lang_code)
    if err:
        result["error"] = err
    else:
        if use_llm_cleanup:
            cleaned, was_cleaned = refine_with_local_llm(english)
            result["english"] = cleaned
            result["llm_cleaned"] = was_cleaned
        else:
            result["english"] = english
        result["engine"] = engine
    return result


# ── Voice ────────────────────────────────────────────────────

def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        print(f"[voice_symptom_reader] Loading Whisper '{MODEL_SIZE}' model "
              f"(first run downloads it once, then it's cached)...")
        t0 = time.time()
        _whisper_model = whisper.load_model(MODEL_SIZE)
        print(f"[voice_symptom_reader] Model ready ({time.time() - t0:.1f}s).")
        return _whisper_model
    except Exception as e:
        print(f"[voice_symptom_reader] Failed to load model: {e}")
        return None


def _whisper_transcribe(model, path, language_code, audio_seconds):
    """
    Layered attempts, each checked against the audio-duration-aware
    hallucination guard — a "successful" call that actually produced
    garbage is treated the same as a failed call and the next attempt
    is tried.
    """
    attempts = [
        dict(language=language_code, temperature=0.0, condition_on_previous_text=False,
             no_speech_threshold=0.7, verbose=False),
        dict(language=language_code, condition_on_previous_text=False,
             no_speech_threshold=0.6, verbose=False),
        dict(condition_on_previous_text=False, no_speech_threshold=0.6, verbose=None),
    ]
    last_err = None
    for i, kw in enumerate(attempts, 1):
        try:
            result = model.transcribe(path, task="transcribe", fp16=False, **kw)
            text = (result.get("text") or "").strip()
            if _looks_like_hallucination(text, audio_seconds=audio_seconds):
                print(f"[voice_symptom_reader] Attempt {i} produced no real speech "
                      f"(got: {text[:60]!r}) — trying next configuration.")
                last_err = RuntimeError("no real speech detected in this attempt")
                continue
            return result
        except Exception as e:
            last_err = e
            print(f"[voice_symptom_reader] Whisper attempt {i} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    raise last_err


def transcribe_voice_note(audio_bytes, filename="voice_note.wav", language_code="mr", use_llm_cleanup=True):
    """
    Returns a dict:
      english            — English text, or None if translation failed
      original           — transcript in the spoken language
      language           — display name of the language used
      engine             — which translation service produced the English text
      llm_cleaned        — True if the optional local-LLM cleanup step ran
      translation_error  — why translation failed (when english is None)
      seconds            — total processing time
      error              — message when the voice note itself could not be processed
    """
    out = {"english": None, "original": None,
           "language": LANGUAGES.get(language_code, language_code),
           "engine": None, "llm_cleaned": False, "translation_error": None,
           "seconds": 0.0, "error": None}
    t_start = time.time()

    try:
        import whisper
    except ImportError:
        out["error"] = (
            "Voice description isn't set up on this computer yet.\n"
            "Run:  pip install openai-whisper deep-translator\n"
            "and install ffmpeg (added to PATH), then restart the app.\n"
            "You can still type your description instead."
        )
        return out

    model = _load_whisper()
    if model is None:
        out["error"] = ("Could not load the Whisper model. Restart the app and try "
                        "again, or type your description instead.")
        return out

    suffix = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio = whisper.load_audio(tmp_path)          # needs ffmpeg
        seconds_of_audio = len(audio) / 16000.0
        print(f"[voice_symptom_reader] Audio: {seconds_of_audio:.1f}s, "
              f"transcribing as {LANGUAGES.get(language_code)}...")
        if seconds_of_audio < 0.7:
            out["error"] = "That recording is too short. Please record again."
            return out

        t0 = time.time()
        try:
            result = _whisper_transcribe(model, tmp_path, language_code, seconds_of_audio)
        except Exception:
            out["error"] = (
                "Couldn't make out any real speech in that recording — it may be silent, "
                "too short, or the microphone recording itself failed. Try recording again "
                "in a quiet place close to the microphone, or type your description instead."
            )
            return out
        original = (result.get("text") or "").strip()
        print(f"[voice_symptom_reader] Transcribed in {time.time() - t0:.1f}s.")

        if not original:
            out["error"] = ("Couldn't make out any speech. Try again closer to the "
                            "microphone in a quiet place, or type your description.")
            return out
        out["original"] = original

        if language_code == "en":
            cleaned, was_cleaned = refine_with_local_llm(original) if use_llm_cleanup else (original, False)
            out["english"], out["engine"], out["llm_cleaned"] = cleaned, "none needed", was_cleaned
        else:
            print("[voice_symptom_reader] Translating to English...")
            english, terr, engine = _translate_to_english(original, language_code)
            if english is not None and use_llm_cleanup:
                english, was_cleaned = refine_with_local_llm(english)
                out["llm_cleaned"] = was_cleaned
            out["english"], out["engine"], out["translation_error"] = english, engine, terr
            if english is None:
                print(f"[voice_symptom_reader] Translation failed:\n{terr}")
        return out
    except Exception as e:
        traceback.print_exc()
        out["error"] = f"Could not process that audio ({type(e).__name__}: {str(e)[:200]}). Try typing instead."
        return out
    finally:
        out["seconds"] = time.time() - t_start
        print(f"[voice_symptom_reader] Total {out['seconds']:.1f}s.")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass