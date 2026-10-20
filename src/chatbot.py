"""
chatbot.py — NephroSense NephroBot
===================================

HOW ANSWERS ARE RETRIEVED AND GENERATED
-----------------------------------------
1. LANGUAGE DETECTION (in Python, not the LLM)
   Scoring: count Marathi-specific vs Hindi-specific Devanagari markers.
   For romanised text: check word sets, but require >= 2 hits to avoid
   false positives from short common words.

2. EMERGENCY CHECK — before any API call.

3. RAG RETRIEVAL — keyword scoring over 10 curated KB articles with
   Marathi/Hindi/English summaries and source URLs (KDIGO, NKF, NIH).

4. SYSTEM PROMPT — language instruction FIRST + medical guidance +
   RAG facts + language instruction LAST.

5. LLM ENGINE (in order of availability):
   - Gemini gemini-flash-lite-latest  — fast, works on free tier
   - Gemma  gemma-4-26b-a4b-it        — backup
   - Ollama llama3.2                  — local offline backup
   - Built-in KB                       — always available

   NOTE: The Groq key on this project currently only has access to
   whisper/guard models, not text LLMs. All text generation goes through
   Gemini/Gemma.  When a working Groq text model is available again,
   add it back via GROQ_MODEL env var.

API KEYS (.streamlit/secrets.toml):
   GEMINI_API_KEY — free at aistudio.google.com
   GROQ_API_KEY   — free at console.groq.com (text models vary by plan)
"""
import json
import os
import re
import threading
import time

# ── API config ────────────────────────────────────────────────────────────────
GROQ_KEY     = os.environ.get("GROQ_API_KEY",  "")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")

# Ordered list of Gemini models to try — first one that responds gets used.
# gemini-flash-lite-latest: fast, generous free quota, good Indic language support
# gemma-4-26b-a4b-it: backup when flash-lite hits quota
GEMINI_MODELS = os.environ.get(
    "GEMINI_MODELS",
    "gemini-flash-lite-latest,gemma-4-26b-a4b-it"
).split(",")

GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.environ.get("NEPHROSENSE_OLLAMA_MODEL", "llama3.2")
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
_ollama_cache   = {"ok": None, "t": 0.0}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NephroBot, a friendly health helper inside the NephroSense app made for Indian patients of all ages and genders.

LANGUAGE — most important rule:
- Patient wrote in Marathi → reply only in simple everyday Marathi people use at home, in Devanagari script.
- Patient wrote in Hindi → reply only in simple everyday Hindi people use at home, in Devanagari script.
- Patient wrote in English → reply only in simple English.
- NEVER mix languages. NEVER use difficult bookish words.
- If you must use a medical term like CKD or UTI, immediately explain it in one plain word.

ANSWER STYLE:
- Greeting: on your FIRST reply in the conversation start with one short greeting — English: "Hello" or "Hey"; Hindi: "नमस्ते"; Marathi: "नमस्कार". NEVER start with "अरे", "अहो", "Aree", "Arre" or any similar interjection. Do NOT repeat a greeting if you already greeted in this conversation.
- Do NOT assume or mention the patient's gender unless they tell you. Use gender-neutral language.
- Do NOT write long paragraphs. Answer in short bullet points ("- "), one short line each, under small plain headings written in the patient's language.
- For symptoms use this order: one gentle reassurance line; Possible causes (2-3 bullets); What to do now (3-5 step bullets); Go to a doctor today if (2-4 bullets); then ONE follow-up question.
- For knowledge questions: a one-line direct answer, then 3-6 bullets with practical examples.
- Be accurate: give only medically correct, widely accepted advice (WHO, NKF, NHS, ICMR). Answer exactly what was asked. If you are not sure, say so and advise seeing a doctor — never guess and never invent numbers.
- Use measurements people understand: "2-3 glasses of water", "15 minutes rest".
- Never stop mid-sentence.

MEDICAL SAFETY:
- Never name medicines or doses.
- Always say when to see a doctor urgently vs when to watch at home.
- Emergency signs (chest pain, no urine, severe swelling, fainting, vomiting blood) → go to hospital NOW / call 112."""

# ── Emergency detection ───────────────────────────────────────────────────────
_EMERGENCY_WORDS = [
    "chest pain", "cannot pass urine", "can't pass urine", "no urine since",
    "not passing urine at all", "severe swelling", "can't breathe", "cannot breathe",
    "unconscious", "fainted", "vomiting blood", "black stool",
    "छाती में दर्द", "पेशाब बिल्कुल नहीं", "सांस नहीं",
    "छातीत दुखत", "लघवी अजिबात", "श्वास येत नाही",
]
EMERGENCY_REPLY = (
    "⚠️ **हे वैद्यकीय आणीबाणी असू शकते / यह मेडिकल इमर्जेंसी हो सकती है / This may be a medical emergency.**\n\n"
    "🇮🇳 **तातडीने जवळच्या रुग्णालयात जा किंवा 112 वर कॉल करा.**\n"
    "🇮🇳 **तुरंत नजदीकी अस्पताल जाएं या 112 पर कॉल करें.**\n"
    "🇬🇧 **Go to the nearest hospital or call 112 (India) immediately. Do not wait.**"
)

# ── Language detection ────────────────────────────────────────────────────────
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

_MARATHI_DEVANAGARI = re.compile(
    r"(आहे|आहेत|आहोत|आहेस|माझ|मला|तुम्ही|तुझ्या|आम्ही|आपण|"
    r"नाही|नाहीत|नाहीस|करतो|करतात|करते|करतोस|"
    r"येतो|येतात|जातो|जातात|असतो|असते|असतात|"
    r"होतो|होते|होतात|वाटत|वाटते|वाटतो|"
    r"दुखत|दुखते|दुखतो|झाले|झाली|झाला|झालात|"
    r"सांग|कळत|कळते|मूत्रपिंड|लघवी|"
    r"काय|आणि|त्याचे|त्याची|त्याचा|किती|कसे|कशाने|"
    r"तर|पण|म्हणजे|म्हणून|कारण|त्यामुळे|"
    r"डॉक्टरांना|रुग्णालय|तपासणी|घ्या|सांगा)"
)

_HINDI_DEVANAGARI = re.compile(
    r"(मुझे|मेरा|मेरी|मेरे|आपको|आपकी|आपके|"
    r"हूँ|हूं|रहा|रही|रहे|रहते|रहती|"
    r"क्या\s|कैसे|क्यों|क्यूं|"
    r"नहीं|नही|"
    r" है | हैं |है।|हैं।|हो।|"
    r"तेरा|तेरी|तुम्हारा|तुम्हारी|"
    r"पेशाब|बुखार| दर्द |दर्द।|दर्द,|"
    r"बीमारी|बीमार|इलाज|दवाई|दवा|"
    r"होता|होती|होते|होना|करना|जाना|"
    r"क्या होती|क्या है|क्या होता|"
    r"के लिए|की पथरी|के बारे|घरेलू|स्वास्थ्य|"
    r"बताएं|बताइए|करूं|करें|बचाव|लक्षण|"
    r"गुर्दे|किडनी की|किडनी के|किडनी में)"
)

# Only Marathi-specific romanised words (NOT common English words like "the", "a")
_MARATHI_ROMAN = {
    "aahe", "ahe", "majha", "majhi", "majhe", "mala", "tumhala", "tumhi", "amhi",
    "kasa", "kashi", "kase", "sanga", "hote", "hoto", "hoti", "lagto", "lagta",
    "zala", "zali", "zale", "dukhat", "dukhtay", "lavkar", "naahi",
    "kartat", "yetat", "jatat", "vatata", "asto", "aste", "aani", "tyache",
    "kiti", "kaay", "mhanje", "mhanun", "asat", "asato",
    "prakar", "lakshane", "upay", "tapasni",
}

# Only Hindi-specific romanised words — removed generic English words like "the", "a", "in"
_HINDI_ROMAN = {
    "mujhe", "mera", "meri", "mere", "aapko", "aapke", "hoon", "raha", "rahi",
    "rahe", "kaisa", "batao", "bataiye", "peshab", "bukhar",
    "theek", "bilkul", "bahut", "bohot", "kya", "kaise", "kyun", "kyon",
    "hain", "gurde", "gurda", "dawai",
    "bimari", "bimar", "ilaj", "dawa",
}
# Minimum romanised hits required to classify as Hindi/Marathi
# (prevents false positives from short shared words)
_MIN_ROMAN_HITS = 2


def _detect_language(text: str) -> str:
    """
    Returns 'marathi', 'hindi', or 'english'.
    Scoring approach: count Marathi vs Hindi markers, pick the winner.
    """
    t = (text or "").strip()
    if not t:
        return "english"

    # Devanagari script present
    if _DEVANAGARI.search(t):
        marathi_hits = len(_MARATHI_DEVANAGARI.findall(t))
        hindi_hits   = len(_HINDI_DEVANAGARI.findall(t))

        if marathi_hits > hindi_hits:
            return "marathi"
        if hindi_hits > marathi_hits:
            return "hindi"

        # Tie-break: Marathi verb endings and postpositions
        marathi_suffix = len(re.findall(r"[^\s](तो|ते|तात|ती|ताना|तांना)\b", t))
        marathi_post   = len(re.findall(r"\b(ला|ने|चे|ची|चा|शी|मध्ये)\b", t))
        # Hindi postpositions — score MORE of them so tied Devanagari defaults to Hindi
        # when the text clearly has Hindi grammar (की, के, में, से, को, लिए, बारे)
        hindi_post = len(re.findall(
            r"\b(को|से|में|पर|के|की|का|और|या|भी|तो|लिए|बारे|घरेलू|पथरी|स्वास्थ्य)\b", t
        ))

        marathi_score = marathi_suffix + marathi_post
        hindi_score   = hindi_post

        if marathi_score > hindi_score:
            return "marathi"
        if hindi_score > marathi_score:
            return "hindi"

        # Marathi question words are a strong signal
        if re.search(r"\b(काय|कसे|कशाने|कुठे|केव्हा|कोण|किती)\b", t):
            return "marathi"

        return "marathi"  # Default Devanagari → marathi for this app

    # Romanised: require >= _MIN_ROMAN_HITS to avoid false positives
    words = set(re.findall(r"[a-z]+", t.lower()))
    marathi_hits = len(words & _MARATHI_ROMAN)
    hindi_hits   = len(words & _HINDI_ROMAN)

    if marathi_hits >= _MIN_ROMAN_HITS and marathi_hits >= hindi_hits:
        return "marathi"
    if hindi_hits >= _MIN_ROMAN_HITS:
        return "hindi"
    return "english"


# ── Language-specific instructions ───────────────────────────────────────────
LANG_INSTRUCTION = {
    "english": (
        "LANGUAGE: Reply in simple, clear English. "
        "Use words a non-medical person can understand. "
        "Answer in short bullet points, not long paragraphs. "
        "Greet with 'Hello' or 'Hey' (never 'Aree'). Give only accurate, well-established medical information."
    ),
    "hindi": (
        "भाषा: रोगी ने हिंदी में पूछा है।\n"
        "जवाब केवल सरल, घरेलू हिंदी में दो — जैसे आप किसी परिवार के सदस्य को समझा रहे हों।\n"
        "कठिन शब्द बिल्कुल मत इस्तेमाल करो। अगर कोई medical शब्द लिखो तो तुरंत आसान भाषा में समझाओ।\n"
        "अंग्रेजी में एक शब्द भी मत लिखो। जवाब छोटे बुलेट पॉइंट्स (-) में दो, लंबा पैराग्राफ नहीं। पहली बार 'नमस्ते' से शुरू करो, 'अरे' से कभी नहीं। सही और पक्की जानकारी ही दो। बीच में कभी मत रुको।"
    ),
    "marathi": (
        "भाषा: रुग्णाने मराठीत विचारले आहे.\n"
        "उत्तर फक्त सोप्या, घरगुती मराठीत द्या — जसे घरच्या कोणाला सांगत आहात.\n"
        "कठीण शब्द वापरू नका. Medical शब्द असेल तर लगेच सोप्या भाषेत समजावा.\n"
        "इंग्रजीत एकही शब्द लिहू नका. उत्तर लहान बुलेट पॉइंट्स (-) मध्ये द्या, मोठा परिच्छेद नको. पहिल्यांदा 'नमस्कार' ने सुरुवात करा, 'अहो' किंवा 'अरे' ने कधीही नाही. अचूक आणि खात्रीशीर माहितीच द्या. अर्धवट सोडू नका."
    ),
}

# ── Offline fallback KB ───────────────────────────────────────────────────────
_KB_OFFLINE = [
    ("Chronic Kidney Disease (CKD)",
     ["ckd", "chronic kidney", "kidney disease", "kidney failure", "egfr", "gfr",
      "kidney", "renal", "murpind", "mootrpind"],
     "CKD means kidneys are damaged and working poorly for 3+ months. "
     "Stages G1–G5 by eGFR: G1 (≥90, damage present), G2 (60–89), G3a (45–59), "
     "G3b (30–44), G4 (15–29), G5 (<15, kidney failure). "
     "Symptoms appear late — test creatinine and urine protein yearly if diabetic or hypertensive. "
     "Main causes: diabetes, high BP, NSAIDs. "
     "Control sugar, control BP, avoid NSAID painkillers, less salt. "
     "Source: KDIGO 2024 — https://kdigo.org/"),
    ("Kidney Stones",
     ["stone", "stones", "calculi", "mutkhada", "pathri", "flank pain", "groin pain",
      "kidney pain", "side pain", "urine block"],
     "Kidney stones cause sudden severe side/back pain spreading to groin, blood in urine, nausea. "
     "Treatment: drink 2.5–3L water daily; small stones (<5mm) usually pass on their own in 1–4 weeks. "
     "Pain relief: lie down, apply a warm cloth to the back, stay hydrated. "
     "See a doctor urgently for: fever with pain (sign of infection), complete urine block, "
     "stones >7mm (need medical removal). "
     "Prevention: drink 3L water daily, less salt, less animal protein. "
     "Source: NKF — https://www.kidney.org/kidney-topics/kidney-stones"),
    ("UTI — Urine Infection",
     ["uti", "urine infection", "burning urine", "burning urination", "cystitis",
      "ladwi jalna", "peshab jalan", "lagwi jalna"],
     "UTI causes burning urination, frequent urination, lower belly pain, cloudy or smelly urine. "
     "Home care: drink 2–3L water, urinate frequently, avoid holding urine, avoid spicy food. "
     "If fever + back/flank pain: kidney infection — see doctor same day (antibiotics needed). "
     "Women: urinate after intercourse, front-to-back hygiene reduces recurrence. "
     "Source: NIH MedlinePlus — https://medlineplus.gov/urinarytractinfections.html"),
    ("Back Pain / Flank Pain",
     ["back pain", "backache", "flank pain", "side pain", "lower back", "pith dukhi",
      "pathdukhna", "pathdukhi", "kamardard", "peeth dard"],
     "Back pain has many causes: muscle strain (most common), kidney stones, UTI, disc problems, poor posture. "
     "Home relief for muscle strain: rest 1–2 days, apply warm cloth (not hot) to painful area for 15 min 3x/day, "
     "gentle stretching after 48 hours, avoid heavy lifting. "
     "See doctor if: pain with fever/chills (kidney infection), burning urine, blood in urine, "
     "pain shooting to groin (kidney stone), weakness in legs, pain lasting more than 2 weeks. "
     "Source: NIH MedlinePlus — https://medlineplus.gov/backpain.html"),
    ("Kidney Health — Diet and Lifestyle",
     ["diet", "food", "healthy kidney", "kidney health", "water intake", "salt", "potassium",
      "aahar", "jevan", "pani", "khana"],
     "For healthy kidneys: drink 2–3L water daily, limit salt to <5g/day, eat more vegetables and fruits, "
     "avoid NSAID painkillers (ibuprofen, diclofenac), control blood sugar and blood pressure. "
     "In CKD (stages 3–5): limit potassium foods (banana, coconut water, tomato, potato) and "
     "phosphorus (cola drinks, processed food) per doctor's advice. "
     "Exercise 30 min daily, stop smoking, limit alcohol. "
     "Source: NKF — https://www.kidney.org/kidney-topics/nutrition-kidney-disease"),
]


def _kb_offline_retrieve(text: str, k: int = 2) -> list[tuple[str, str]]:
    t = (text or "").lower()
    scored = []
    for name, kws, body in _KB_OFFLINE:
        s = sum(1 for kw in kws if kw in t)
        if s:
            scored.append((s, name, body))
    scored.sort(reverse=True)
    return [(n, b) for _, n, b in scored[:k]]


def _offline_answer(text: str, lang: str = "english") -> str:
    refs = _kb_offline_retrieve(text)
    if not refs:
        if lang == "marathi":
            return "मला किडनी आरोग्य, किडनी स्टोन, UTI, डायलिसिस यावर मदत करता येते. कृपया अधिक तपशील सांगा."
        if lang == "hindi":
            return "मैं किडनी स्वास्थ्य, किडनी स्टोन, UTI, डायलिसिस पर मदद कर सकता हूं। कृपया अधिक जानकारी दें।"
        return ("I can help with kidney health, kidney stones, UTI, dialysis, diet, and blood tests. "
                "Could you give me more details?")
    parts = [f"**{n}**\n\n{b}" for n, b in refs[:2]]
    return "\n\n---\n\n".join(parts) + "\n\n_General information — confirm with your doctor._"


# ── Ollama check ──────────────────────────────────────────────────────────────
def _ollama_ok() -> bool:
    now = time.time()
    if _ollama_cache["ok"] is not None and now - _ollama_cache["t"] < 30:
        return _ollama_cache["ok"]
    ok = False
    try:
        import requests
        r = requests.get(OLLAMA_TAGS_URL, timeout=1.5)
        if r.status_code == 200:
            ok = any(OLLAMA_MODEL in m.get("name", "") for m in r.json().get("models", []))
    except Exception:
        pass
    _ollama_cache.update(ok=ok, t=now)
    return ok


def chatbot_status() -> tuple[bool, str]:
    if GEMINI_KEY:
        return True, "🟢 NephroBot online — Gemini AI + built-in medical KB."
    if GROQ_KEY:
        return True, "🟢 NephroBot online — Groq AI + built-in medical KB."
    if _ollama_ok():
        return True, f"🟢 NephroBot running offline ({OLLAMA_MODEL})."
    return True, ("🟡 Basic mode — answering from built-in kidney knowledge (English only). "
                  "Add GEMINI_API_KEY in .streamlit/secrets.toml for full multilingual AI.")


# ── SSE streaming helper ──────────────────────────────────────────────────────
def _sse_lines(r):
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            yield line[5:].strip()


# ── Gemini streaming (tries each model in GEMINI_MODELS until one works) ─────
def _stream_gemini(system: str, history: list[dict]):
    """
    Tries models in GEMINI_MODELS in order.
    Uses non-streaming for reliability (streaming SSE drops chunks on some models).
    Yields text tokens one at a time so st.write_stream() works.
    """
    import requests
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in history
    ]
    last_exc = None
    for model in GEMINI_MODELS:
        model = model.strip()
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": GEMINI_KEY},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 1200,
                    },
                },
                timeout=(10, 60),
            )
            if r.status_code == 429:
                last_exc = Exception(f"{model}: quota exceeded (429)")
                continue
            if r.status_code == 404:
                last_exc = Exception(f"{model}: not found (404)")
                continue
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                last_exc = Exception(f"{model}: no candidates in response")
                continue
            text = candidates[0]["content"]["parts"][0].get("text", "")
            if not text.strip():
                last_exc = Exception(f"{model}: empty text")
                continue
            # Yield in chunks so st.write_stream shows streaming effect
            words = text.split(" ")
            chunk = ""
            for w in words:
                chunk += w + " "
                if len(chunk) >= 20:
                    yield chunk
                    chunk = ""
            if chunk:
                yield chunk
            return
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc


# ── Groq streaming ────────────────────────────────────────────────────────────
def _stream_groq(messages: list[dict]):
    import requests
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}",
                 "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages,
              "temperature": 0.3, "max_tokens": 1200, "stream": True},
        stream=True, timeout=(5, 60),
    )
    r.raise_for_status()
    for data in _sse_lines(r):
        if data == "[DONE]":
            break
        try:
            piece = json.loads(data)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError):
            continue
        if piece:
            yield piece


# ── Ollama streaming ──────────────────────────────────────────────────────────
def _stream_ollama(messages: list[dict]):
    import requests
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True,
              "keep_alive": "30m",
              "options": {"temperature": 0.3, "num_predict": 800, "num_ctx": 3000}},
        stream=True, timeout=(5, 120),
    )
    r.raise_for_status()
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        piece = (d.get("message") or {}).get("content")
        if piece:
            yield piece
        if d.get("done"):
            break


# ── Background warm-up ────────────────────────────────────────────────────────
def warm_up():
    def _go():
        try:
            if not (GROQ_KEY or GEMINI_KEY) and _ollama_ok():
                import requests
                requests.post(OLLAMA_CHAT_URL,
                              json={"model": OLLAMA_MODEL,
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "stream": False, "keep_alive": "30m",
                                    "options": {"num_predict": 1}},
                              timeout=120)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


# ── Main streaming reply generator ───────────────────────────────────────────
def stream_chatbot_reply(conversation_history: list[dict],
                         patient_context: str = ""):
    """
    Generator — yields text tokens one by one for st.write_stream().
    """
    if not conversation_history:
        yield "Please type a question."
        return

    last_msg = conversation_history[-1]["content"]

    # 1. Emergency check
    t_low = last_msg.lower()
    if any(w in t_low for w in _EMERGENCY_WORDS):
        yield EMERGENCY_REPLY
        return

    # 2. Language detection
    lang = _detect_language(last_msg)

    # 3. RAG retrieval
    recent_text = " ".join(
        m["content"] for m in conversation_history[-4:]
        if m["role"] == "user"
    )
    try:
        from rag_kb import retrieve_context, format_context_for_prompt
        kb_articles = retrieve_context(recent_text, lang, k=2)
        kb_section  = format_context_for_prompt(kb_articles, lang)
    except Exception:
        kb_articles = []
        kb_section  = ""

    # 4. No API available → offline answer
    if not (GEMINI_KEY or GROQ_KEY) and not _ollama_ok():
        yield _offline_answer(last_msg, lang)
        return

    # 5. Build system prompt — language instruction FIRST and LAST
    system = LANG_INSTRUCTION[lang] + "\n\n" + SYSTEM_PROMPT

    if patient_context:
        ctx_label = {
            "english": "PATIENT HEALTH CONTEXT (use if relevant):",
            "hindi":   "रोगी का स्वास्थ्य संदर्भ (प्रासंगिक होने पर उपयोग करें):",
            "marathi": "रुग्णाची आरोग्य माहिती (प्रासंगिक असल्यास वापरा):",
        }[lang]
        system += f"\n\n{ctx_label}\n{patient_context}"

    if kb_section:
        system += f"\n\n{kb_section}"

    system += "\n\n" + LANG_INSTRUCTION[lang]  # repeated at end

    # 6. Build messages
    history = conversation_history[-6:]
    if not history or history[-1]["role"] != "user":
        history = history + [{"role": "user", "content": last_msg}]

    messages = [{"role": "system", "content": system}] + history
    messages[-1] = {
        "role": "user",
        "content": messages[-1]["content"] + f"\n\n({LANG_INSTRUCTION[lang]})",
    }

    # 7. Engine selection
    engines = []
    if GEMINI_KEY:
        # Gemini for all languages — gemini-flash-lite is fast and handles Indic well
        engines.append(("gemini", lambda: _stream_gemini(system, history)))
    if GROQ_KEY:
        engines.append(("groq", lambda: _stream_groq(messages)))
    if _ollama_ok():
        engines.append(("ollama", lambda: _stream_ollama(messages)))

    # 8. Stream with fallback
    for name, make_stream in engines:
        started = False
        try:
            for piece in make_stream():
                started = True
                yield piece
            if started:
                return
        except Exception:
            if started:
                return
            continue

    # 9. All engines failed → offline KB
    yield _offline_answer(last_msg, lang)


def get_chatbot_reply(conversation_history: list[dict],
                      patient_context: str = "") -> tuple[str | None, str | None]:
    """Non-streaming version for legacy callers."""
    text = "".join(stream_chatbot_reply(conversation_history, patient_context)).strip()
    return (text, None) if text else (None, "NephroBot could not answer right now.")