"""Quick diagnosis: test language quality and spot naturalness issues."""
import sys, os, time, re
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()

from chatbot import _detect_language, stream_chatbot_reply

tests = [
    ('Marathi', 'मला पाठदुखी होत आहे, काय करावे?'),
    ('Marathi', 'किडनी स्टोन साठी घरगुती उपाय सांगा'),
    ('Hindi',   'मुझे पीठ में दर्द है, क्या करूं?'),
    ('Hindi',   'किडनी की पथरी के लिए घरेलू उपाय बताएं'),
    ('English', 'I have back pain, what should I do?'),
]

for label, q in tests:
    detected = _detect_language(q)
    h = [{'role':'user','content':q}]
    t0 = time.time()
    reply = ''.join(stream_chatbot_reply(h, ''))
    print(f'[{label}|det={detected}] {q}')
    print(f'  {time.time()-t0:.1f}s | {len(reply.split())} words')
    print(f'  {reply[:250]}')
    print()
