"""
Tests: language detection, reply language, reply completeness, no gender assumptions.
Simulates 4 consecutive questions in each language (fresh conversation each time).
"""
import sys, os, time, re
sys.path.insert(0, 'src')

from chatbot import _detect_language, stream_chatbot_reply

# --- 4 consecutive questions per language, simulating real conversation ---
SESSIONS = {
    "Marathi": [
        "किडनी डिजीज काय असतात?",
        "त्यावर घरगुती उपाय काय आहेत?",
        "डॉक्टरकडे कधी जावे?",
        "किडनी साठी चांगले अन्न कोणते आहे?",
    ],
    "Hindi": [
        "किडनी की बीमारी क्या होती है?",
        "किडनी की पथरी के लिए घरेलू उपाय बताएं",
        "डॉक्टर के पास कब जाना चाहिए?",
        "किडनी के लिए अच्छा खाना क्या है?",
    ],
    "English": [
        "What is kidney disease?",
        "What are home remedies for kidney stones?",
        "When should I see a doctor?",
        "What foods are good for kidney health?",
    ],
}

all_pass = True
for lang_label, questions in SESSIONS.items():
    print(f"\n{'='*55}")
    print(f"  Session: {lang_label} — {len(questions)} consecutive questions")
    print(f"{'='*55}")
    history = []
    for qi, q in enumerate(questions):
        detected = _detect_language(q)
        history.append({"role": "user", "content": q})
        t0 = time.time()
        try:
            reply = "".join(stream_chatbot_reply(history, ""))
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  Q{qi+1} ERROR: {e}")
            all_pass = False
            history.append({"role": "assistant", "content": "[error]"})
            continue

        history.append({"role": "assistant", "content": reply})
        words   = len(reply.split())
        has_dev = bool(re.search(r'[\u0900-\u097F]', reply))
        exp_lang = lang_label.lower()
        lang_ok = detected == exp_lang
        scr_ok  = (exp_lang in ('marathi','hindi') and has_dev) or (exp_lang=='english' and not has_dev)
        comp_ok = words >= 40
        # Check no gender assumption for first question
        gender_words = ['ताई','बहन','sister','lady','ma\'am','madam','she','her','महिला','स्त्री']
        gender_ok = not any(gw.lower() in reply.lower() for gw in gender_words)

        status = 'PASS' if (lang_ok and scr_ok and comp_ok) else 'FAIL'
        if not (lang_ok and scr_ok and comp_ok):
            all_pass = False

        print(f"  Q{qi+1}: {status}  detect={detected} script={'Dev' if has_dev else 'Lat'} words={words} {elapsed:.1f}s gender={'ok' if gender_ok else 'WARN-gender-assumed'}")
        print(f"       Q: {q}")
        print(f"       A: {reply[:180]}")
        print()

print("\n" + "="*55)
print("RESULT:", "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
