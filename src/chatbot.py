"""
chatbot.py — NephroSense  (REPLACE existing)
NephroBot — kidney + general health chatbot.

LANGUAGE RULE (fixed): replies are in VERY SIMPLE ENGLISH by default.
Hindi/Marathi is used ONLY if the patient themselves writes in Hindi/Marathi
(Devanagari script or Hindi/Marathi words in English letters).
The language is detected in code, not left to the small model to guess.

Engines, first one that works wins:
  1. Groq   (free API)   -> env GROQ_API_KEY
  2. Gemini (free API)   -> env GEMINI_API_KEY
  3. Ollama (offline)    -> ollama pull llama3.2
  4. Built-in offline knowledge base (English)
Optional env vars: GROQ_MODEL, GEMINI_MODEL, NEPHROSENSE_OLLAMA_MODEL
"""
import json
import os
import re
import threading
import time

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.environ.get("NEPHROSENSE_OLLAMA_MODEL", "llama3.2")
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
TIMEOUT = 60
_ollama_cache = {"ok": None, "t": 0.0}

SYSTEM_PROMPT = """You are NephroBot, a kind health helper inside the NephroSense kidney-health app.
Use very simple words and short sentences. Answer in 4-7 lines.
For a symptom: one kind sentence; 3-4 common reasons (mention kidney/urine causes only if they fit); warning signs to see a doctor soon; 2 safe tips; then ONE simple follow-up question. If it has lasted weeks or months, say it needs a doctor check and name simple tests (blood test, urine test, ultrasound).
You know kidney diseases in depth (CKD stages and eGFR, sudden kidney injury, diabetic and high-BP kidney damage, nephrotic syndrome, glomerulonephritis, polycystic kidney, lupus nephritis, stones, UTI, urine blockage, prostate, dialysis, transplant, kidney diet, medicines that harm kidneys) and common illnesses. Use REFERENCE FACTS and PATIENT CONTEXT when relevant.
Never say "you definitely have". Give no medicine names or doses. Emergency signs (chest pain, no urine at all, severe swelling, trouble breathing, confusion, vomiting blood, black stools, fainting): tell them to go to a hospital or call 112 now. You are not a doctor - say so once, briefly."""

EMERGENCY_WORDS = [
    "chest pain", "cannot pass urine", "can't pass urine", "cant pass urine", "no urine since",
    "not passing urine at all", "severe swelling", "can't breathe", "cannot breathe",
    "unconscious", "fainted", "vomiting blood", "vomiting blood", "black stool",
    "छाती में दर्द", "पेशाब बिल्कुल नहीं", "छातीत दुखत", "लघवी अजिबात", "सांस नहीं",
]
EMERGENCY_REPLY = (
    "⚠️ **This may be a medical emergency.** Please go to the nearest hospital or call "
    "emergency services (112 in India) right now. Do not wait for an app or chatbot answer. "
    "If someone is with you, ask them to take you immediately."
)

# ── Language detection (done in code so a small model can't get it wrong) ──
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_HINGLISH_WORDS = {
    "mujhe", "mera", "meri", "mere", "hai", "hain", "nahi", "nahin", "kya", "kaise", "kyun", "kyon",
    "dard", "aap", "aapko", "aapke", "bukhar", "hoon", "raha", "rahi", "rahe", "karo", "kaisa",
    "batao", "bataiye", "gurde", "peshab", "pet dard", "aahe", "majha", "majhi", "mala", "kay",
    "tumhala", "hoto", "hote", "lagto", "nahi", "ahe", "kasa", "kashi", "sanga", "lavkar",
}


def _detect_language(text):
    """Returns 'hindi_marathi' only if the patient clearly wrote in Hindi/Marathi, else 'english'."""
    t = (text or "").lower()
    if _DEVANAGARI.search(t):
        return "hindi_marathi"
    words = set(re.findall(r"[a-z]+", t))
    if len(words & _HINGLISH_WORDS) >= 1:
        return "hindi_marathi"
    return "english"


LANG_INSTRUCTION = {
    "english": (
        "LANGUAGE (very important): The patient wrote in English. Reply ONLY in very simple English. "
        "Do NOT use any Hindi or Marathi words, and do not write Hindi in English letters."
    ),
    "hindi_marathi": (
        "LANGUAGE: The patient wrote in Hindi or Marathi. Reply in the same language and the same "
        "script they used, in very simple everyday words."
    ),
}

# ── Built-in knowledge base (used for retrieval AND as offline answers) ──
KB = [
    ("Stomach / belly pain that lasts for weeks or months",
     ["stomach pain", "stomach is paining", "stomach paining", "belly pain", "abdominal pain", "abdomen",
      "stomach ache", "stomachache", "gastric", "acidity", "gas", "ulcer", "indigestion", "pet dard"],
     "Belly pain for weeks or months should always be checked by a doctor. Common reasons: acidity or gastritis, stomach ulcer, constipation or irritable bowel, gallstones, infection (typhoid, worms), and in women period or ovary problems. Kidney or urine causes are also possible: kidney stones (pain in the side or lower belly that may spread to the groin), urine infection (burning urine, lower belly pain), or a blocked urine flow. See a doctor soon if there is weight loss, vomiting, black stools, blood in vomit or urine, fever, yellow eyes, or pain that wakes you at night. Useful tests: blood tests, urine test, and an ultrasound of the belly."),
    ("Chronic kidney disease (CKD) & stages",
     ["ckd", "chronic kidney", "kidney disease", "kidney failure", "stage", "egfr", "gfr", "kidney damage", "किडनी खराब", "मूत्रपिंड", "किडनी"],
     "CKD means the kidneys have been damaged or working less well for 3+ months (eGFR below 60, or protein in urine). Stages by eGFR: G1 ≥90 (normal but with damage signs), G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 <15 (kidney failure). Early CKD often has NO symptoms, so tests matter: serum creatinine/eGFR and urine albumin (ACR). Common causes: diabetes, high BP, glomerulonephritis, PKD, long-term painkillers. Control sugar and BP, avoid NSAID painkillers, eat less salt, stop smoking, stay active, and get yearly tests if at risk."),
    ("Acute kidney injury (AKI)",
     ["aki", "acute kidney", "sudden kidney", "kidney injury", "contrast", "dehydration"],
     "AKI is a sudden drop in kidney function over hours to days, often from severe dehydration, infection/sepsis, low BP, blockage, contrast dye, or drugs (NSAIDs, some antibiotics). Signs: much less urine, swelling, nausea, tiredness, confusion. It needs urgent medical care; many people recover fully if treated early."),
    ("Diabetic kidney disease",
     ["diabetes", "diabetic", "sugar", "dm", "मधुमेह", "शुगर", "साखर"],
     "High blood sugar slowly damages the kidney's tiny filters. First sign is protein (albumin) in urine, later eGFR falls. Yearly urine ACR and creatinine tests are advised for every diabetic. Good sugar control (HbA1c target set by doctor), BP control, and doctor-prescribed kidney-protective medicines slow the damage a lot."),
    ("Hypertensive kidney disease & kidney-BP link",
     ["high blood pressure", "hypertension", "bp", "blood pressure", "रक्तदाब", "ब्लड प्रेशर"],
     "Long-standing high BP harms the kidney's small vessels, and damaged kidneys raise BP further — a vicious circle. Target is usually below 130/80 for people with kidney disease (doctor decides). Cut salt, exercise, keep weight healthy, take BP medicines regularly, and check BP at home."),
    ("Glomerulonephritis, nephrotic & nephritic syndrome, IgA nephropathy",
     ["glomerulo", "nephrotic", "nephritic", "iga", "foamy", "frothy", "protein in urine", "proteinuria", "swelling face", "albumin"],
     "These are diseases of the kidney's filters. Nephrotic syndrome: heavy protein loss → foamy urine, swelling of face/legs, low blood albumin, high cholesterol. Nephritic: blood in urine (cola-coloured), high BP, swelling. IgA nephropathy is the commonest glomerular disease worldwide. Diagnosis: urine ACR/PCR, urine microscopy, blood tests, sometimes kidney biopsy. Treatment depends on the type and may include steroids/immune medicines — needs a nephrologist."),
    ("Polycystic kidney disease (PKD)",
     ["polycystic", "pkd", "cyst", "inherited", "genetic", "family history"],
     "PKD is an inherited condition where many fluid-filled cysts grow in both kidneys. Signs: flank pain, blood in urine, high BP, UTIs, stones, enlarged kidneys. Diagnosed by ultrasound; family members may be screened. Control of BP, plenty of water and regular nephrology follow-up help; newer medicines exist for some patients."),
    ("Kidney stones (mutkhada)",
     ["stone", "stones", "mutkhada", "calculi", "renal colic", "मुतखडा", "पथरी", "kidney pain", "back pain", "side pain"],
     "Stones form when minerals crystallise in urine. Signs: sudden severe pain in side/back that spreads to groin, blood in urine, nausea, burning urination. Drink 2.5-3 litres of water daily, cut excess salt, limit very high-oxalate foods if advised, and keep normal calcium intake. See a doctor for severe pain, fever with pain, or if you cannot pass urine; most small stones pass on their own, bigger ones need ultrasound/CT and urology care."),
    ("Urinary tract infection (UTI) & pyelonephritis",
     ["uti", "urine infection", "burning urine", "burning", "pyelonephritis", "cystitis", "painful urination", "जलन", "जळजळ"],
     "UTI: burning or urgent urination, going often, lower belly pain, cloudy or smelly urine. If it climbs to the kidney (pyelonephritis) you get fever, chills, back/flank pain and vomiting — needs prompt treatment. Needs urine test/culture and a doctor-prescribed antibiotic. Prevention: drink water, don't hold urine, urinate after intercourse, front-to-back hygiene. Repeated UTIs need checking for stones or blockage."),
    ("Urinary obstruction, hydronephrosis & enlarged prostate (BPH)",
     ["prostate", "bph", "hydronephrosis", "obstruction", "weak stream", "urine stream", "night urination", "nocturia"],
     "Blocked urine flow (stone, prostate enlargement, narrowing, tumour) backs up pressure and can damage the kidney (hydronephrosis). Signs: weak stream, straining, dribbling, getting up at night, incomplete emptying. Ultrasound with residual urine check, PSA in men, and creatinine are done. Treatable — don't delay."),
    ("Lupus nephritis & autoimmune kidney disease",
     ["lupus", "sle", "autoimmune", "rheumatoid", "vasculitis"],
     "In autoimmune disease the immune system can attack the kidney filters. Lupus nephritis is common in young women with lupus: foamy urine, swelling, high BP, blood/protein in urine. Needs a nephrologist and rheumatologist, immune-suppressing treatment and regular tests."),
    ("Painkillers (NSAIDs) and medicines that harm kidneys",
     ["painkiller", "nsaid", "ibuprofen", "diclofenac", "brufen", "voveran", "medicine", "herbal", "ayurvedic", "contrast dye"],
     "Regular use of NSAID painkillers (ibuprofen, diclofenac, naproxen, etc.) can injure kidneys, especially with dehydration, diabetes, BP or existing CKD. Some herbal/'ayurvedic' preparations with heavy metals also harm kidneys. Tell every doctor about your kidney problem before any new medicine or contrast scan. Never stop prescribed medicines on your own — ask your doctor."),
    ("Dialysis and kidney transplant",
     ["dialysis", "haemodialysis", "hemodialysis", "transplant", "peritoneal", "fistula", "डायलिसिस"],
     "Dialysis is needed when kidneys fail (usually eGFR below ~10-15 with symptoms). Haemodialysis cleans blood through a machine, usually 3 times a week for ~4 hours, using a fistula or catheter. Peritoneal dialysis uses the belly lining and can be done at home. A kidney transplant gives the best quality of life for suitable patients; donors can be living or deceased. Government schemes (e.g., PM-JAY / state schemes) may cover dialysis — ask a hospital social worker."),
    ("Diet and lifestyle for kidney health",
     ["diet", "food", "eat", "salt", "potassium", "protein", "water", "drink", "banana", "आहार", "खाना"],
     "For healthy kidneys: less salt (under 5 g/day), plenty of water unless a doctor restricts fluids, more vegetables, limited processed food and sugary drinks, regular exercise, no smoking, limited alcohol. In CKD diets are personalised: advanced CKD may need limits on potassium (banana, coconut water, tomato, potato), phosphorus (colas, processed food), salt and sometimes protein or fluids — a renal dietitian should guide you. Do not start a restrictive diet on your own."),
    ("Anaemia, high potassium and other CKD complications",
     ["anaemia", "anemia", "hemoglobin", "haemoglobin", "hb", "potassium", "bone", "itching", "tired", "fatigue"],
     "Damaged kidneys make less erythropoietin → anaemia (tiredness, pale look, breathlessness). They also can't remove potassium/phosphate/acid well → high potassium, weak bones, itching, nausea. These are checked with CBC, potassium, calcium, phosphate and PTH, and are treatable under a doctor's care."),
    ("Kidney function tests: creatinine, urea, eGFR, ACR",
     ["creatinine", "urea", "bun", "test", "report", "acr", "normal range", "uric", "blood test"],
     "Serum creatinine (normal roughly 0.6-1.2 mg/dL; lower in women/young) is used with age and sex to calculate eGFR (normal ≥90). Blood urea normal ~15-40 mg/dL (higher with dehydration, high-protein diet, steroids). Urine ACR (albumin-to-creatinine ratio) below 30 mg/g is normal; 30-300 is mildly raised, above 300 markedly raised. A single abnormal result needs repeating — CKD needs 3+ months of abnormality."),
    ("Diabetes, thyroid and anaemia basics",
     ["thyroid", "tsh", "hba1c", "hypothyroid", "hyperthyroid", "sugar level"],
     "Diabetes: fasting sugar above 126 mg/dL or HbA1c ≥6.5% on repeat testing. Thyroid disorders cause tiredness, weight change and mood changes and are checked with TSH; they can mildly affect kidney tests. Anaemia in women is often iron deficiency; a doctor should check the cause."),
    ("Fever, malaria, dengue, typhoid (and kidneys)",
     ["fever", "malaria", "dengue", "typhoid", "jaundice", "viral", "ताप", "बुखार"],
     "Fever with chills/shivering → malaria possible; high fever with body/eye pain and low platelets → dengue; step-ladder fever with stomach upset → typhoid. All need testing (malaria/dengue/Widal or culture) and a doctor's treatment. Severe infections, dehydration or sepsis can injure the kidneys, so drink fluids, avoid NSAID painkillers unless the doctor allows, and seek care urgently if urine becomes very little or you feel confused or very weak."),
]


def _retrieve(text, k=3):
    t = (text or "").lower()
    scored = []
    for name, kws, body in KB:
        s = sum(1 for kw in kws if kw in t)
        if s:
            scored.append((s, name, body))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(n, b) for _, n, b in scored[:k]]


def _is_emergency(text):
    t = (text or "").lower()
    return any(w in t for w in EMERGENCY_WORDS)


def _ollama_ok():
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
        ok = False
    _ollama_cache.update(ok=ok, t=now)
    return ok


def chatbot_status():
    """Always (True, text): worst case the offline knowledge base answers."""
    if GROQ_KEY:
        return True, "🟢 NephroBot is online (Groq AI + built-in kidney knowledge)."
    if GEMINI_KEY:
        return True, "🟢 NephroBot is online (Gemini AI + built-in kidney knowledge)."
    if _ollama_ok():
        return True, f"🟢 NephroBot is running offline on this computer ({OLLAMA_MODEL})."
    return True, ("🟡 Basic mode: answering from the built-in kidney knowledge base (English). "
                  "For fuller answers set a free GROQ_API_KEY or GEMINI_API_KEY.")


def _sse_lines(r):
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            yield line[5:].strip()


def _stream_groq(messages):
    import requests
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.4, "max_tokens": 450, "stream": True},
        stream=True, timeout=(5, 60))
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


def _stream_gemini(system, history):
    import requests
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in history]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent",
        params={"key": GEMINI_KEY, "alt": "sse"},
        json={"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
              "generationConfig": {"temperature": 0.4, "maxOutputTokens": 500}},
        stream=True, timeout=(5, 60))
    r.raise_for_status()
    for data in _sse_lines(r):
        try:
            parts = json.loads(data)["candidates"][0]["content"]["parts"]
        except (ValueError, KeyError, IndexError):
            continue
        for p in parts:
            if p.get("text"):
                yield p["text"]


def _stream_ollama(messages):
    import requests
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True, "keep_alive": "30m",
              "options": {"temperature": 0.3, "num_predict": 320, "num_ctx": 2048}},
        stream=True, timeout=(5, 120))
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


def warm_up():
    """Load the offline model into memory in the background so the first question isn't slow."""
    def _go():
        try:
            if not (GROQ_KEY or GEMINI_KEY) and _ollama_ok():
                import requests
                requests.post(OLLAMA_CHAT_URL,
                              json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}],
                                    "stream": False, "keep_alive": "30m", "options": {"num_predict": 1}},
                              timeout=120)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


_PERSONAL = re.compile(r"\b(i|my|me|myself|mine|we|our)\b", re.I)
_GENERAL_Q = re.compile(r"^\s*(what|why|how|when|which|who|define|explain|tell me about|meaning of|causes? of|symptoms? of)\b", re.I)


def _instant_answer(text, lang):
    """General 'what is / why / how' questions are answered at once from the built-in kidney knowledge,
    so a slow local model is not needed. Personal questions (about the patient's own symptoms) go to the AI."""
    if lang != "english" or _PERSONAL.search(text or "") or not _GENERAL_Q.search(text or ""):
        return None
    refs = _retrieve(text, k=1)
    if not refs:
        return None
    name, body = refs[0]
    return (f"**{name}**\n\n{body}\n\n_General information, not a diagnosis. "
            f"Ask me a follow-up question any time._")


def _offline_answer(refs):
    if not refs:
        return ("I can help with kidney health (CKD, stones, UTI, dialysis, transplant, diet, test reports) "
                "and common illnesses. Could you tell me a little more? For example: "
                "\"What are the signs of kidney stones?\" or \"What does eGFR mean?\"")
    parts = [f"**{n}**\n\n{b}" for n, b in refs[:2]]
    return ("\n\n---\n\n".join(parts) +
            "\n\n_This is general information, not a diagnosis. Please confirm with a doctor._")


def stream_chatbot_reply(conversation_history, patient_context=""):
    """Generator of text pieces: the answer appears word by word instead of after a long wait."""
    if not conversation_history:
        yield "Please type a question."
        return
    last = conversation_history[-1]["content"]
    if _is_emergency(last):
        yield EMERGENCY_REPLY
        return

    lang = _detect_language(last)
    if not (GROQ_KEY or GEMINI_KEY):          # only a slow local model is available -> answer common questions instantly
        quick = _instant_answer(last, lang)
        if quick:
            yield quick
            return
    recent_user_text = " ".join(m["content"] for m in conversation_history[-4:] if m["role"] == "user")
    refs = _retrieve(recent_user_text, k=2)

    system = SYSTEM_PROMPT
    if patient_context:
        system += f"\n\nPATIENT CONTEXT (only if relevant): {patient_context}"
    if refs:
        system += "\n\nREFERENCE FACTS:\n" + "\n".join(f"- {n}: {b[:420]}" for n, b in refs)
    system += "\n\n" + LANG_INSTRUCTION[lang]

    history = conversation_history[-6:]
    if lang == "english":
        history = [m for m in history if m["role"] == "user" or _detect_language(m["content"]) == "english"]
    if not history or history[-1]["role"] != "user":
        history = history + [{"role": "user", "content": last}]

    messages = [{"role": "system", "content": system}] + history
    messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n(" + LANG_INSTRUCTION[lang] + ")"}

    engines = []
    if GROQ_KEY:
        engines.append(lambda: _stream_groq(messages))
    if GEMINI_KEY:
        engines.append(lambda: _stream_gemini(system, history))
    if _ollama_ok():
        engines.append(lambda: _stream_ollama(messages))

    for make in engines:
        started = False
        try:
            for piece in make():
                started = True
                yield piece
            if started:
                return
        except Exception:
            if started:          # already showing an answer - stop cleanly instead of repeating it
                return
            continue
    yield _offline_answer(refs)


def get_chatbot_reply(conversation_history, patient_context=""):
    """Non-streaming version (kept for older callers). Returns (reply_or_None, error_or_None)."""
    text = "".join(stream_chatbot_reply(conversation_history, patient_context)).strip()
    return (text, None) if text else (None, "NephroBot could not answer right now. Please try again.")