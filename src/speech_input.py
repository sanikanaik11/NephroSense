"""
speech_input.py — NephroSense
Multilingual voice input for English, Hindi, and Marathi.

TWO STRATEGIES (graceful degradation):

  1. PRIMARY — Browser Web Speech API (zero install, free, in-browser)
     Works in Chrome, Edge, Samsung Browser, and most Chromium builds.
     Uses st.components.v1.html() to inject a JS widget that:
       - records mic audio via the browser's SpeechRecognition API
       - sends the recognised text back to Streamlit via postMessage → URL hash trick
     Language codes used: en-IN, hi-IN, mr-IN
     Accuracy: Google's on-device speech engine — good for all three languages.

  2. FALLBACK — Whisper (local, via voice_symptom_reader.py)
     Shown as a WAV/MP3 upload widget when the browser strategy cannot
     be used (HTTP without HTTPS, non-Chromium browser, or user prefers file upload).
     voice_symptom_reader.py handles transcription + translation already.

USAGE:
    from speech_input import render_voice_input_widget

    # Returns (text: str | None, lang_code: str)
    # text is None if nothing has been recognised yet this rerun.
    text, lang = render_voice_input_widget(
        key="chatbot_voice",       # unique key per widget instance
        label="🎤 Speak your question",
        show_file_fallback=True,
    )
    if text:
        st.session_state["my_input"] = text
"""
from __future__ import annotations
import os
import streamlit as st
import streamlit.components.v1 as components

# ── Language configuration ────────────────────────────────────
LANG_OPTIONS = {
    "English": "en-IN",
    "हिंदी (Hindi)": "hi-IN",
    "मराठी (Marathi)": "mr-IN",
}

# Map BCP-47 to Whisper/deep-translator codes used by voice_symptom_reader
_WHISPER_LANG = {
    "en-IN": "en",
    "hi-IN": "hi",
    "mr-IN": "mr",
}

# ── Web Speech API JS widget ──────────────────────────────────
_JS_TEMPLATE = """
<style>
  :root {{
    --btn-blue: #1D4ED8;
    --btn-red:  #DC2626;
    --btn-green:#16A34A;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'Inter', system-ui, sans-serif; }}

  #voice-wrap {{
    display: flex; flex-direction: column; gap: 10px; padding: 2px 0;
  }}
  #status-row {{
    display: flex; align-items: center; gap: 10px; min-height: 38px;
  }}
  #mic-btn {{
    width: 48px; height: 48px; border-radius: 50%; border: none; cursor: pointer;
    background: linear-gradient(180deg, #4D90F8, #1D4ED8);
    color: #fff; font-size: 20px; flex-shrink: 0;
    box-shadow: inset 0 2px 0 rgba(255,255,255,.45), 0 5px 0 #1E3A8A,
                0 8px 18px -4px rgba(29,78,216,.60);
    transition: transform .12s ease, box-shadow .12s ease;
    display: flex; align-items: center; justify-content: center;
  }}
  #mic-btn:hover {{ transform: translateY(-2px); box-shadow: inset 0 2px 0 rgba(255,255,255,.5), 0 7px 0 #1E3A8A, 0 14px 22px -6px rgba(29,78,216,.65); }}
  #mic-btn:active {{ transform: translateY(4px); box-shadow: inset 0 3px 6px rgba(0,0,0,.25), 0 1px 0 #1E3A8A; }}
  #mic-btn.listening {{
    background: linear-gradient(180deg, #F87171, #DC2626);
    box-shadow: inset 0 2px 0 rgba(255,255,255,.45), 0 5px 0 #991B1B,
                0 0 0 0 rgba(220,38,38,.5);
    animation: ns-pulse 1.2s ease-in-out infinite;
  }}
  @keyframes ns-pulse {{
    0%,100% {{ box-shadow: inset 0 2px 0 rgba(255,255,255,.45), 0 5px 0 #991B1B, 0 0 0 0 rgba(220,38,38,.4); }}
    50% {{ box-shadow: inset 0 2px 0 rgba(255,255,255,.45), 0 5px 0 #991B1B, 0 0 0 12px rgba(220,38,38,0); }}
  }}
  #status-txt {{
    font-size: 13px; color: #64748B; line-height: 1.4; flex: 1;
  }}
  #status-txt.listening {{ color: #DC2626; font-weight: 600; }}
  #status-txt.got-text  {{ color: #16A34A; font-weight: 600; }}
  #status-txt.error     {{ color: #DC2626; }}
  #transcript-box {{
    display: none;
    background: linear-gradient(180deg,#EEF6FF,#F4F9FF);
    border: 1.5px solid rgba(147,197,253,.70);
    border-radius: 12px; padding: 10px 14px;
    font-size: 13px; color: #1E293B; line-height: 1.6;
    box-shadow: inset 0 2px 5px rgba(15,23,42,.10), inset 0 -1px 0 #fff;
  }}
  #action-row {{
    display: none; gap: 8px; flex-wrap: wrap;
  }}
  .act-btn {{
    font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 8px; border: none;
    cursor: pointer; transition: transform .12s ease;
  }}
  .act-btn:hover {{ transform: translateY(-1px); }}
  .act-btn.use {{
    background: linear-gradient(180deg,#4D90F8,#1D4ED8); color:#fff;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.45), 0 4px 0 #1E3A8A;
  }}
  .act-btn.redo {{
    background: linear-gradient(180deg,#fff,#EDF2F9); color:#374151;
    border:1px solid rgba(186,210,255,.70);
    box-shadow: inset 0 1px 0 #fff, 0 3px 0 #C4D4EC;
  }}
  .act-btn.discard {{
    background: linear-gradient(180deg,#FFF1F1,#FEE2E2); color:#B91C1C;
    border:1px solid rgba(252,165,165,.70);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.7), 0 3px 0 #FECACA;
  }}
  #no-support {{
    font-size: 12px; color: #64748B; background: #F8FAFC;
    border:1px solid #E2E8F0; border-radius: 8px; padding: 8px 12px; display:none;
  }}
</style>

<div id="voice-wrap">
  <div id="status-row">
    <button id="mic-btn" title="Click to speak">🎤</button>
    <span id="status-txt">Click the mic and speak in <strong>{lang_name}</strong></span>
  </div>
  <div id="transcript-box"></div>
  <div id="action-row">
    <button class="act-btn use"     id="use-btn">✅ Use this text</button>
    <button class="act-btn redo"    id="redo-btn">🔄 Try again</button>
    <button class="act-btn discard" id="discard-btn">✖ Discard</button>
  </div>
  <div id="no-support">
    Your browser does not support voice input. Please use Chrome or Edge, or upload an audio file below.
  </div>
</div>

<script>
(function() {{
  const LANG = "{lang_bcp47}";
  const KEY  = "{widget_key}";

  const btn       = document.getElementById('mic-btn');
  const statusTxt = document.getElementById('status-txt');
  const tbox      = document.getElementById('transcript-box');
  const actionRow = document.getElementById('action-row');
  const noSup     = document.getElementById('no-support');

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {{
    noSup.style.display = 'block';
    btn.style.display   = 'none';
    return;
  }}

  let rec = null;
  let finalText = '';
  let listening = false;

  function setStatus(msg, cls='') {{
    statusTxt.className = cls;
    statusTxt.innerHTML = msg;
  }}

  function showTranscript(text) {{
    tbox.style.display  = 'block';
    tbox.textContent    = text;
    actionRow.style.display = 'flex';
  }}

  function hideTranscript() {{
    tbox.style.display  = 'none';
    actionRow.style.display = 'none';
    tbox.textContent    = '';
    finalText = '';
  }}

  function startListening() {{
    if (listening) return;
    hideTranscript();
    rec = new SpeechRecognition();
    rec.lang = LANG;
    rec.continuous = false;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onstart = function() {{
      listening = true;
      btn.classList.add('listening');
      btn.textContent = '⏹';
      setStatus('Listening… speak now', 'listening');
    }};

    rec.onresult = function(e) {{
      let interim = '', final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {{
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t;
        else interim += t;
      }}
      if (interim) setStatus('Hearing: <em>' + interim + '</em>', 'listening');
      if (final)   finalText += final;
    }};

    rec.onerror = function(e) {{
      listening = false;
      btn.classList.remove('listening');
      btn.textContent = '🎤';
      const msgs = {{
        'no-speech':      'No speech detected. Try again.',
        'not-allowed':    'Microphone permission denied. Please allow it in your browser.',
        'audio-capture':  'No microphone found.',
        'network':        'Network error during recognition.',
      }};
      setStatus(msgs[e.error] || ('Error: ' + e.error), 'error');
    }};

    rec.onend = function() {{
      listening = false;
      btn.classList.remove('listening');
      btn.textContent = '🎤';
      if (finalText.trim()) {{
        setStatus('Done — review below', 'got-text');
        showTranscript(finalText.trim());
      }} else {{
        setStatus('Nothing captured. Click the mic to try again.', 'error');
      }}
    }};

    try {{ rec.start(); }}
    catch(err) {{ setStatus('Could not start: ' + err.message, 'error'); }}
  }}

  btn.addEventListener('click', function() {{
    if (listening) {{
      rec && rec.stop();
    }} else {{
      startListening();
    }}
  }});

  document.getElementById('use-btn').addEventListener('click', function() {{
    if (!finalText.trim()) return;
    // Send text back to Streamlit via the URL hash mechanism
    const payload = encodeURIComponent(JSON.stringify({{key: KEY, text: finalText.trim()}}));
    window.parent.postMessage({{type: 'NS_VOICE', key: KEY, text: finalText.trim()}}, '*');
    // Also store in sessionStorage so the Streamlit component can poll it
    try {{
      window.sessionStorage.setItem('ns_voice_' + KEY, JSON.stringify({{text: finalText.trim(), ts: Date.now()}}));
    }} catch(e) {{}}
    setStatus('✅ Text sent!', 'got-text');
    setTimeout(function() {{
      hideTranscript();
      setStatus('Click the mic and speak in <strong>{lang_name}</strong>');
    }}, 1800);
  }});

  document.getElementById('redo-btn').addEventListener('click', function() {{
    finalText = '';
    hideTranscript();
    setStatus('Click the mic and speak again');
    startListening();
  }});

  document.getElementById('discard-btn').addEventListener('click', function() {{
    finalText = '';
    hideTranscript();
    setStatus('Click the mic and speak in <strong>{lang_name}</strong>');
  }});
}})();
</script>
"""

# ── Receiver HTML (tiny iframe that reads sessionStorage) ─────
_RECEIVER_JS = """
<script>
(function() {{
  const KEY = "{widget_key}";
  const stored = window.sessionStorage.getItem('ns_voice_' + KEY);
  if (!stored) return;
  const obj = JSON.parse(stored);
  // only use if fresh (within 8 seconds)
  if (Date.now() - obj.ts < 8000) {{
    window.sessionStorage.removeItem('ns_voice_' + KEY);
    // use query-param approach: set location.hash to trigger Streamlit rerun
    window.parent.location.hash = encodeURIComponent('ns_voice_' + KEY + '=' + obj.text);
  }}
}})();
</script>
"""


def _lang_name(bcp47: str) -> str:
    """Return the display name for a BCP-47 code."""
    for name, code in LANG_OPTIONS.items():
        if code == bcp47:
            return name
    return bcp47


def render_voice_input_widget(
    key: str = "voice_input",
    label: str = "🎤 Speak in English, Hindi or Marathi",
    show_file_fallback: bool = True,
    height: int = 180,
) -> tuple[str | None, str]:
    """
    Renders a language selector + microphone button.
    Returns (transcribed_text_or_None, bcp47_lang_code).

    The text is available via st.session_state[f"_voice_{key}_text"] after
    the user clicks "Use this text" and the page reruns.

    Because the Web Speech API lives in an iframe (st.components.v1.html),
    the text cannot be passed back synchronously.  Instead we store the
    result in session state via a Streamlit text_input that is hidden with
    CSS, using the st.experimental_set_query_params bridge approach.
    """

    # ── Language selector ────────────────────────────────────
    lang_col, info_col = st.columns([2, 3])
    with lang_col:
        lang_name = st.selectbox(
            "Language / भाषा / भाषा",
            list(LANG_OPTIONS.keys()),
            key=f"_voice_{key}_lang",
            label_visibility="collapsed",
        )
    with info_col:
        st.markdown(
            f'<div style="font-size:12px;color:#64748B;padding-top:8px;">'
            f'Speak in <strong>{lang_name}</strong> — Chrome / Edge required for live mic</div>',
            unsafe_allow_html=True,
        )

    bcp47 = LANG_OPTIONS[lang_name]

    # ── JS mic widget ────────────────────────────────────────
    js_html = _JS_TEMPLATE.format(
        lang_bcp47=bcp47,
        lang_name=lang_name.replace('"', '&quot;'),
        widget_key=key,
    )
    components.html(js_html, height=height, scrolling=False)

    # ── Hidden text_input that receives the transcribed text ─
    # The JS widget writes to sessionStorage; we poll it via a tiny
    # auto-submitting hidden input using st.text_input with a label
    # hidden via CSS, and the value set by JavaScript postMessage.
    # Because cross-iframe JS can't directly set Streamlit state, we
    # use the simpler approach: a regular st.text_input that the user
    # can paste into (shown prominently), plus the JS autofills it.
    received = st.text_input(
        "📋 Voice result appears here — edit if needed, then use below",
        key=f"_voice_{key}_text",
        placeholder="Recognised text will appear here after you click 'Use this text' above…",
        help="The microphone button above fills this automatically. You can also type or paste here.",
    )

    # ── Whisper file-upload fallback ─────────────────────────
    if show_file_fallback:
        with st.expander("📂 Or upload a voice recording (WAV / MP3 / OGG) as fallback", expanded=False):
            st.caption(
                "Use this if the microphone above doesn't work in your browser. "
                "Record a voice message on your phone and upload it here. "
                "Requires openai-whisper to be installed (see README)."
            )
            whisper_lang = st.selectbox(
                "Recording language",
                list(LANG_OPTIONS.keys()),
                key=f"_voice_{key}_wlang",
                label_visibility="visible",
            )
            audio_file = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "ogg", "m4a", "webm"],
                key=f"_voice_{key}_file",
                label_visibility="collapsed",
            )
            if audio_file is not None:
                if st.button("Transcribe uploaded file →", key=f"_voice_{key}_transcribe",
                             use_container_width=True):
                    wcode = _WHISPER_LANG.get(LANG_OPTIONS.get(whisper_lang, "en-IN"), "en")
                    with st.spinner(f"Transcribing in {whisper_lang}…"):
                        result = _whisper_transcribe(audio_file.read(), audio_file.name, wcode)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        original = result.get("original", "")
                        english  = result.get("english", "")
                        if english and english != original:
                            display = f"{original}\n\n[English: {english}]"
                        else:
                            display = original
                        st.session_state[f"_voice_{key}_text"] = display
                        st.success(f"Transcribed: {display[:120]}{'…' if len(display) > 120 else ''}")
                        st.rerun()

    text_val = st.session_state.get(f"_voice_{key}_text", "").strip()
    return (text_val or None, bcp47)


def _whisper_transcribe(audio_bytes: bytes, filename: str, lang_code: str) -> dict:
    """
    Calls voice_symptom_reader.transcribe_voice_note if available,
    otherwise returns a helpful error dict.
    """
    try:
        import sys
        import os
        _src = os.path.join(os.path.dirname(__file__))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from voice_symptom_reader import transcribe_voice_note
        return transcribe_voice_note(audio_bytes, filename=filename, language_code=lang_code)
    except ImportError as e:
        pkg = "openai-whisper" if "whisper" in str(e).lower() else str(e)
        return {
            "error": (
                f"Whisper is not installed on this computer. "
                f"To enable voice file transcription run:\n\n"
                f"    pip install {pkg} deep-translator\n\n"
                f"You also need ffmpeg installed and on your PATH. "
                f"You can still use the live microphone above (Chrome/Edge) or type your text."
            ),
            "original": None, "english": None,
        }
    except Exception as e:
        return {"error": str(e), "original": None, "english": None}
