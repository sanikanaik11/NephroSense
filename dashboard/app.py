"""
NephroSense — Phase 4
Kidney Health Screening Platform
Run: python -m streamlit run dashboard/app.py
"""
import os
import sys
import time
import json
import html
import re
from datetime import datetime, date

import streamlit as st

# API keys may be placed in .streamlit/secrets.toml instead of environment variables
for _k in ("GEMINI_API_KEY", "GROQ_API_KEY", "GEOAPIFY_API_KEY"):
    try:
        if not os.environ.get(_k) and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from database import (
    register_user, login_user, get_user_history, save_assessment,
    get_latest_component_scores, get_latest_clinical_notes, get_latest_egfr,
    get_latest_lifestyle_answers,
)
from lifestyle_model import (
    QUESTIONS, FEAT_LABELS, CLINICAL_QUESTIONS, visible_clinical_questions, predict_lifestyle_risk,
)
from egfr import calculate_egfr_ckdepi
from clinical_risk_score import calculate_clinical_risk_score
from risk_engine import assess_risk, age_text
from disease_predictor import predict_diseases
from history_extractor import extract_history

# The training data's "bp" is DIASTOLIC (the lower number, ~80). Make the form say so.
for _q in QUESTIONS:
    if _q["key"] == "bp":
        _q["label"] = "Diastolic blood pressure (the LOWER number, e.g. 80 in 120/80)"
        _q["unit"] = "mm Hg"
FEAT_LABELS["bp"] = "Diastolic blood pressure"
_GRADE_LABELS = {0: "Nil / Negative", 1: "Trace", 2: "+ (1+)", 3: "++ (2+)", 4: "+++ (3+)", 5: "++++ (4+)"}
for _q in QUESTIONS:
    if _q["key"] == "al":
        _q["label"] = "Albumin (protein) in urine — as printed on your report"
    if _q["key"] == "su":
        _q["label"] = "Sugar (glucose) in urine — as printed on your report"

try:
    from eye_model import analyze_eye_image, generate_gradcam, eye_label
    EYE_AVAILABLE, EYE_IMPORT_ERROR = True, None
except Exception as e:
    EYE_AVAILABLE, EYE_IMPORT_ERROR = False, str(e)

    def analyze_eye_image(*_a, **_k):
        return {"eye_score": None, "note": "Visual Screening isn't set up on this computer yet."}

    def generate_gradcam(*_a, **_k):
        return None

    def eye_label(score):
        return ""

try:
    from report_reader import (extract_values_from_files, read_text_from_files, uses_online_ai, reader_ready,
                               last_stats, warm_up, REPORT_HINTS)
    REPORT_READER_AVAILABLE = True
except Exception as e:
    REPORT_READER_AVAILABLE = False
    REPORT_HINTS = {}

    def extract_values_from_files(*_a, **_k):
        return {}, None, "Report scanning isn't available right now. Please type the values in by hand."

    def read_text_from_files(*_a, **_k):
        return "", "Report scanning isn't available right now. Please add the scan by hand."

    def uses_online_ai():
        return False

    def reader_ready():
        return False

    def last_stats():
        return {"seconds": 0.0, "pages": 0, "engine": ""}

    def warm_up():
        return None

try:
    from imaging_reader import extract_imaging, FINDING_LABELS, MAJOR_KEYS
except Exception as e:
    FINDING_LABELS, MAJOR_KEYS = {"normal_study": "Reported as normal"}, set()

    def extract_imaging(*_a, **_k):
        return {"modality": "Other", "impression": "", "findings": [], "normal": False}

try:
    from chatbot import get_chatbot_reply, chatbot_status, stream_chatbot_reply, warm_up as chatbot_warm_up
    CHATBOT_AVAILABLE = True
except Exception as e:
    CHATBOT_AVAILABLE = False

    def get_chatbot_reply(*_a, **_k):
        return None, "NephroBot isn't set up on this computer yet."

    def chatbot_status():
        return False, "NephroBot isn't set up on this computer yet."

    def stream_chatbot_reply(*_a, **_k):
        yield "NephroBot isn't set up on this computer yet."

    def chatbot_warm_up():
        return None

@st.cache_resource
def _start_reader_warmup():
    warm_up()
    try:
        chatbot_warm_up()
    except Exception:
        pass
    return True


_start_reader_warmup()

st.set_page_config(
    page_title="NephroSense | Kidney Health",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_STATE = {
    "logged_in": False,
    "user": None,
    "page": "home",
    "auth_tab": "login",
    "reg_step": 1,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go(page):
    st.session_state.page = page
    st.rerun()


@st.cache_resource
def _load_risk_model():
    import joblib
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model = joblib.load(os.path.join(model_dir, "ckd_model.pkl"))
    meta = joblib.load(os.path.join(model_dir, "ckd_meta.pkl"))
    return model, meta


def _get_top_risk_factors(n=5):
    """Global feature importance of the ML model (not personalised)."""
    try:
        model, meta = _load_risk_model()
        feature_cols = meta["feature_cols"]
        est = model.named_estimators_["xgb"] if hasattr(model, "named_estimators_") and "xgb" in model.named_estimators_ else model
        importances = est.feature_importances_
        pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:n]
        return [(FEAT_LABELS.get(k, k), float(v)) for k, v in pairs]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_find_hosp(city_in, budget):
    return find_hosp(city_in, budget)


def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.rerun()


def _parse_notes(raw):
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


EYE_MIN_CONFIDENCE = 0.55   # below this the photo model is "inconclusive" and nothing is saved


def _hb_status(u):
    """Compares against the user's own blood report. Returns (status, hb) with status in
    'normal' / 'low' / None (no blood report on file)."""
    try:
        notes = _parse_notes(get_latest_clinical_notes(u["id"]))
        ans = _parse_notes(get_latest_lifestyle_answers(u["id"]))
        if not notes.get("_have_blood") or ans.get("hemo") is None:
            return None, None
        hb = float(ans["hemo"])
        cut = 13.0 if str(u.get("gender", "")).strip().lower().startswith("m") else 12.0
        return ("normal" if hb >= cut else "low"), hb
    except Exception:
        return None, None


_DATE_NUM = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
_DATE_TXT = re.compile(r"\b(\d{1,2})[ \-]([A-Za-z]{3,9})[ ,\-]+(\d{4})\b")
_DATE_KEYS = [("collect", 5), ("sample", 4), ("receiv", 3), ("regist", 3), ("report", 2), ("date", 1)]


def _iso_to_date(v):
    try:
        return date.fromisoformat(str(v)) if v else None
    except ValueError:
        return None


def _guess_report_date(text):
    """Best guess of the sample/collection date printed on a lab report (patient must confirm)."""
    best = None
    for line in (text or "").splitlines():
        low = line.lower()
        if any(w in low for w in ("dob", "birth", "born", "age")):
            continue
        prio = max([p for k, p in _DATE_KEYS if k in low] or [0])
        if prio == 0:
            continue
        cands = []
        for d_, m_, y_ in _DATE_NUM.findall(line):
            cands.append((d_, m_, y_))
        for d_, mon, y_ in _DATE_TXT.findall(line):
            for fmt in ("%b", "%B"):
                try:
                    cands.append((d_, str(datetime.strptime(mon, fmt).month), y_))
                    break
                except ValueError:
                    continue
        for d_, m_, y_ in cands:
            try:
                dt = date(int(y_), int(m_), int(d_))
            except ValueError:
                continue
            if date(2000, 1, 1) <= dt <= date.today():
                if best is None or prio > best[0] or (prio == best[0] and dt < best[1]):
                    best = (prio, dt)
    return best[1] if best else None


def _date_input(label, key, help_text):
    kw = dict(min_value=date(2000, 1, 1), max_value=date.today(), format="DD/MM/YYYY",
              key=key, help=help_text)
    if key not in st.session_state:
        kw["value"] = None
    try:
        return st.date_input(label, **kw)
    except Exception:
        kw["value"] = date.today()
        kw.pop("key", None)
        return st.date_input(label, **kw)


def _render_predictions(preds):
    if not preds:
        st.info(
            "No specific kidney-condition pattern was detected from your answers. "
            "If you have symptoms, please still see a doctor."
        )
        return
    for p in preds:
        cls = "hi" if p["likelihood"].startswith("Higher") else "po"
        ev = "".join(f"<li>{html.escape(e)}</li>" for e in p["evidence"])
        st.markdown(
            f'<div class="ns-dis"><div class="ns-dis-h"><div class="ns-dis-n">{html.escape(p["name"])}</div>'
            f'<span class="ns-dis-b {cls}">{html.escape(p["likelihood"])}</span></div>'
            f'<ul>{ev}</ul>'
            f'<div class="ns-dis-t">🔬 To confirm: {html.escape(p["tests"])}<br>'
            f'👨‍⚕️ See: {html.escape(p["specialist"])}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Pattern-based suggestions from your answers and test values using standard nephrology "
        "associations (KDIGO/NICE). They show what your doctor may want to rule out — they are "
        "NOT a diagnosis."
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root, .stApp{
  --background-color:#F1F5F9!important;
  --secondary-background-color:#FFFFFF!important;
  --text-color:#0F172A!important;
  color-scheme:light!important;
}

*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.stApp{background:#F1F5F9!important;}

.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] li{
  color:#0F172A!important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption{
  color:#64748B!important;
}
[data-testid="stMetricValue"]{color:#0F172A!important;}
[data-testid="stMetricLabel"]{color:#64748B!important;}
[data-testid="stMetricDelta"]{color:#64748B!important;}

div.block-container{padding-top:0!important;padding-bottom:0!important;
  padding-left:0!important;padding-right:0!important;max-width:100%!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important;}
[data-testid="stHeader"]{
  background:transparent!important;
  height:2.75rem!important;
  min-height:2.75rem!important;
}
[data-testid="stDecoration"]{display:none!important;height:0!important;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stMainMenu"]{display:none!important;}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{
  display:flex!important;
  visibility:visible!important;
  opacity:1!important;
  position:fixed!important;
  top:10px!important;
  left:10px!important;
  z-index:999999!important;
  background:#fff!important;
  border:1.5px solid #E2E8F0!important;
  border-radius:8px!important;
  padding:4px!important;
  box-shadow:0 2px 6px rgba(0,0,0,.08)!important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg{
  fill:#1D4ED8!important;
}
section[data-testid="stSidebar"]{
  min-width:260px!important;
}

[data-testid="stHorizontalBlock"]{
  gap:0!important;
  flex-wrap:nowrap!important;
  align-items:stretch!important;
}
[data-testid="column"]{
  padding:0!important;
  min-width:0!important;
}
[data-testid="column"] > div:first-child{
  padding:0!important;
  margin:0!important;
  height:100%!important;
}
.element-container{padding:0!important;margin:0!important;}
.stMarkdown{padding:0!important;margin:0!important;}

.ns-bar{position:sticky;top:0;z-index:9999;background:#fff;
  border-bottom:1.5px solid #E2E8F0;padding:0 20px 0 36px;height:60px;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.ns-brand{display:flex;align-items:center;gap:9px;}
.ns-brand-ic{width:34px;height:34px;border-radius:9px;
  background:linear-gradient(135deg,#1D4ED8,#3B82F6);
  display:flex;align-items:center;justify-content:center;font-size:17px;}
.ns-brand-nm{font-size:18px;font-weight:700;color:#0F172A;letter-spacing:-0.3px;}
.ns-brand-nm b{color:#1D4ED8;}
.ns-av{width:34px;height:34px;border-radius:50%;
  background:linear-gradient(135deg,#1D4ED8,#3B82F6);
  color:#fff;font-size:13px;font-weight:700;
  display:flex;align-items:center;justify-content:center;}

.stat-row{display:flex;gap:0;max-width:380px;
  border:1.5px solid #E2E8F0;border-radius:12px;overflow:hidden;margin-top:32px;}
.stat-box{flex:1;padding:14px 10px;text-align:center;
  background:#F8FAFC;border-right:1.5px solid #E2E8F0;}
.stat-box:last-child{border-right:none;}
.stat-num{font-size:20px;font-weight:800;color:#0F172A;line-height:1;margin-bottom:3px;}
.stat-lbl{font-size:10px;color:#64748B;line-height:1.3;}

.auth-title{font-size:20px;font-weight:700;color:#0F172A!important;margin-bottom:4px;}
.auth-sub{font-size:13px;color:#64748B!important;margin-bottom:18px;}
.divline{height:1px;background:#E2E8F0;margin:14px 0;}

div.st-key-dash_wrap,
div.st-key-content_wrap,
div.st-key-lifestyle_wrap{
  background:#F1F5F9!important;padding:8px 32px 24px!important;
}

div.st-key-capture_card{
  background:#fff!important;border:1.5px solid #E2E8F0!important;border-radius:16px!important;
  padding:20px 22px 24px!important;margin:16px 0 14px!important;
  box-shadow:0 1px 3px rgba(15,23,42,.04)!important;
}
div.st-key-capture_card [data-baseweb="tab-list"]{
  gap:6px!important;border-bottom:1.5px solid #E2E8F0!important;margin-bottom:16px!important;
}
div.st-key-capture_card [data-baseweb="tab"]{
  font-size:14px!important;font-weight:600!important;color:#64748B!important;
  padding:8px 4px!important;background:transparent!important;
}
div.st-key-capture_card [data-baseweb="tab"][aria-selected="true"]{
  color:#1D4ED8!important;
}
div.st-key-capture_card [data-baseweb="tab-highlight"]{
  background-color:#1D4ED8!important;height:3px!important;border-radius:3px!important;
}
div.st-key-capture_card [data-testid="stCameraInput"] video,
div.st-key-capture_card [data-testid="stCameraInput"] img{
  border-radius:12px!important;
}
div.st-key-capture_card button[kind="primary"]{
  background:#1D4ED8!important;border-radius:8px!important;
}

.ns-welc{background:linear-gradient(135deg,#1E3A8A,#1D4ED8 60%,#3B82F6);
  border-radius:14px;padding:22px 26px;color:#fff;
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:14px;flex-wrap:wrap;gap:12px;}
.ns-welc *{color:#fff!important;}
.nw-g{font-size:11px;font-weight:600;opacity:.7;letter-spacing:1px;
  text-transform:uppercase;margin-bottom:3px;}
.nw-n{font-size:22px;font-weight:800;letter-spacing:-0.5px;margin-bottom:3px;}
.nw-m{font-size:12px;opacity:.65;}
.nw-box{background:rgba(255,255,255,.15);border:1.5px solid rgba(255,255,255,.25);
  border-radius:10px;padding:12px 20px;text-align:center;flex-shrink:0;}
.nw-v{font-size:24px;font-weight:800;line-height:1;margin-bottom:3px;}
.nw-l{font-size:10px;opacity:.75;font-weight:500;letter-spacing:.5px;text-transform:uppercase;}
.ns-kc{background:#fff;border:1px solid #E2E8F0;border-radius:14px;
  padding:16px 18px;text-align:left;box-shadow:0 1px 2px rgba(15,23,42,.04);}
.ns-kc-i{font-size:20px;margin-bottom:6px;}
.ns-kc-v{font-size:20px;font-weight:800;color:#0F172A!important;letter-spacing:-.3px;margin-bottom:2px;}
.ns-kc-v.re{color:#DC2626!important;}
.ns-kc-v.am{color:#D97706!important;}
.ns-kc-v.gr{color:#059669!important;}
.ns-kc-v.bl{color:#1D4ED8!important;}
.ns-kc-l{font-size:11.5px;color:#64748B!important;font-weight:500;}

.ns-lbl{font-size:11px;font-weight:600;color:#94A3B8!important;text-transform:uppercase;
  letter-spacing:1px;margin-bottom:10px;margin-top:4px;}
.ns-mods{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;}
.ns-mod{background:#fff;border:1.5px solid #E2E8F0;border-radius:12px;padding:18px;transition:all .15s;}
.ns-mod *{color:#0F172A!important;}
.ns-mod.op{cursor:pointer;}
.ns-mod.op:hover{border-color:#93C5FD;box-shadow:0 4px 14px rgba(29,78,216,.09);transform:translateY(-1px);}
.ns-mod.lk{opacity:.5;cursor:not-allowed;}
.ns-mp{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:600;
  padding:3px 9px;border-radius:8px;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px;}
.ns-mp.av{background:#EFF6FF;color:#1D4ED8!important;}.ns-mp.sn{background:#F1F5F9;color:#94A3B8!important;}
.ns-mi{font-size:24px;margin-bottom:8px;}
.ns-mt{font-size:13px;font-weight:600;color:#0F172A!important;margin-bottom:3px;}
.ns-md{font-size:11px;color:#64748B!important;line-height:1.5;}

.ns-hf-top{background:#fff;border:1px solid #E2E8F0;border-radius:16px;
  padding:22px 26px;margin-bottom:18px;display:flex;align-items:center;gap:16px;
  box-shadow:0 1px 3px rgba(15,23,42,.04), 0 8px 24px -12px rgba(29,78,216,.10);}
.ns-hft{font-size:18px;font-weight:700;color:#0F172A!important;margin-bottom:3px;letter-spacing:-.2px;}
.ns-hfs{font-size:12.5px;color:#64748B!important;line-height:1.5;}
.ns-hcard{background:#fff;border:1px solid #E2E8F0;border-radius:14px;
  padding:18px 20px;margin-bottom:10px;display:flex;gap:12px;
  box-shadow:0 1px 2px rgba(15,23,42,.04);transition:all .18s;}
.ns-hcard:hover{border-color:#93C5FD;box-shadow:0 4px 16px -4px rgba(29,78,216,.14);transform:translateY(-1px);}
.ns-hrk{width:28px;height:28px;border-radius:50%;background:#EFF6FF;color:#1D4ED8!important;
  font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;margin-top:2px;}
.ns-hn{font-size:14px;font-weight:700;color:#0F172A!important;margin-bottom:2px;}
.ns-ht{font-size:11px;font-weight:600;color:#1D4ED8!important;margin-bottom:5px;}
.ns-ha{font-size:12px;color:#64748B!important;margin-bottom:4px;}
.ns-hp{font-size:12px;color:#374151!important;margin-bottom:6px;}
.ns-htags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;}
.ns-htag{font-size:10px;font-weight:500;padding:2px 7px;border-radius:5px;
  background:#F1F5F9;color:#475569!important;}
.ns-hlinks{display:flex;gap:7px;}
.ns-hl{font-size:11px;font-weight:600;padding:5px 11px;border-radius:6px;text-decoration:none;}
.ns-hl.m{background:#EFF6FF;color:#1D4ED8!important;}.ns-hl.c{background:#F0FDF4;color:#059669!important;}
.ns-bb{font-size:11px;font-weight:700;padding:4px 9px;border-radius:6px;
  flex-shrink:0;align-self:flex-start;white-space:nowrap;}
.bb-l{background:#F0FDF4;color:#059669!important;}.bb-m{background:#FFFBEB;color:#D97706!important;}
.bb-h{background:#FEF2F2;color:#DC2626!important;}
.ns-open{display:inline-block;font-size:11.5px;font-weight:700;padding:3px 10px;border-radius:7px;margin:2px 0 6px;}
.ns-open.open{background:#F0FDF4;color:#047857!important;}
.ns-open.closed{background:#FEF2F2;color:#B91C1C!important;}
.ns-open.unk{background:#F1F5F9;color:#64748B!important;}
.ns-hnote{font-size:11.5px;color:#475569!important;line-height:1.55;margin-top:8px;background:#F8FAFC;
  border-radius:8px;padding:7px 10px;}
.ns-hl.b{background:#FFF7ED;color:#C2410C!important;}

.ns-dis{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:14px 18px;margin-bottom:10px;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
.ns-dis-h{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:6px;}
.ns-dis-n{font-size:14.5px;font-weight:700;color:#0F172A!important;}
.ns-dis-b{font-size:10.5px;font-weight:700;padding:3px 9px;border-radius:6px;white-space:nowrap;}
.ns-dis-b.hi{background:#FEF2F2;color:#DC2626!important;}
.ns-dis-b.po{background:#FFFBEB;color:#D97706!important;}
.ns-dis ul{margin:0 0 6px;padding-left:18px;font-size:12.5px;line-height:1.7;}
.ns-dis ul li{color:#374151!important;}
.ns-dis-t{font-size:11.5px;color:#64748B!important;line-height:1.6;}


div.st-key-navpanel{padding:8px 0 24px 24px!important;}
div.st-key-topbar_row{background:#fff!important;border-bottom:1.5px solid #E2E8F0;box-shadow:0 1px 4px rgba(0,0,0,.06);}
div.st-key-topbar_row .ns-bar{border-bottom:none!important;box-shadow:none!important;}
div.st-key-topbar_row [data-testid="stHorizontalBlock"]{align-items:center!important;}
div.st-key-topbar_row [data-testid="stColumn"]:last-child,div.st-key-topbar_row [data-testid="column"]:last-child{padding-right:20px!important;}
div.st-key-topbar_row div.stButton{padding-bottom:0!important;}
div.st-key-home_wrap{padding:0 48px 32px!important;}
/* 3D depth: consistent layered shadows + hover lift on existing cards */
.ns-kc,.ns-dis,.ns-hcard,.ns-hf-top,.ns-mod,.ns-sb-card,.stat-row,div.st-key-capture_card{
  box-shadow:0 1px 2px rgba(15,23,42,.06),0 6px 16px -6px rgba(15,23,42,.14),0 14px 28px -14px rgba(29,78,216,.16)!important;
  transition:transform .18s ease,box-shadow .18s ease!important;}
.ns-kc:hover,.ns-dis:hover,.ns-hcard:hover,.ns-mod:hover{
  transform:translateY(-3px)!important;
  box-shadow:0 2px 4px rgba(15,23,42,.08),0 12px 24px -8px rgba(15,23,42,.18),0 22px 36px -16px rgba(29,78,216,.24)!important;}
.ns-welc{box-shadow:0 2px 4px rgba(15,23,42,.10),0 12px 28px -8px rgba(30,58,138,.40)!important;}
.ns-bar,div.st-key-topbar_row{box-shadow:0 2px 6px rgba(15,23,42,.08),0 8px 18px -10px rgba(15,23,42,.18)!important;}
div.st-key-topbar_row .ns-bar{box-shadow:none!important;}
@media (prefers-reduced-motion: reduce){.ns-kc:hover,.ns-dis:hover,.ns-hcard:hover,.ns-mod:hover{transform:none!important;}}
@media (max-width: 800px){div.st-key-home_wrap{padding:0 16px 24px!important;}}
@media (max-width: 800px){
  div.st-key-shell [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;}
  div.st-key-shell [data-testid="column"],
  div.st-key-shell [data-testid="stColumn"]{flex:1 1 100%!important;min-width:100%!important;width:100%!important;}
  div.st-key-navpanel{padding:12px 16px 0!important;}
}

div[data-testid="stDateInput"] label p{font-size:13.5px!important;font-weight:600!important;color:#1E293B!important;}
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="base-input"],
div[data-testid="stDateInput"] input{
  background:#F8FAFC!important;color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;
  border-color:#E2E8F0!important;border-radius:8px!important;font-size:13px!important;}
div[data-testid="stDateInput"] input::placeholder{color:#94A3B8!important;-webkit-text-fill-color:#94A3B8!important;}
div[data-testid="stDateInput"] svg{fill:#64748B!important;}
div[data-testid="stMultiSelect"] label p{font-size:13.5px!important;font-weight:600!important;color:#1E293B!important;}
div[data-testid="stMultiSelect"] [data-baseweb="select"]>div{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;border-radius:8px!important;}
div[data-testid="stMultiSelect"] input{color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;}
div[data-testid="stMultiSelect"] svg{fill:#64748B!important;}
span[data-baseweb="tag"]{background:#DBEAFE!important;max-width:100%!important;}
span[data-baseweb="tag"] *{color:#1E40AF!important;}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label{
  font-size:12px!important;font-weight:600!important;color:#374151!important;}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;
  border-radius:8px!important;color:#0F172A!important;
  font-size:13px!important;padding:9px 12px!important;}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus{
  border-color:#1D4ED8!important;background:#fff!important;
  box-shadow:0 0 0 3px rgba(29,78,216,.07)!important;outline:none!important;}
div[data-testid="stTextInput"] input::placeholder{color:#94A3B8!important;}
div[data-testid="stTextArea"] label{
  font-size:12px!important;font-weight:600!important;color:#374151!important;}
div[data-testid="stTextArea"] textarea{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;
  border-radius:8px!important;color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;
  font-size:14px!important;padding:9px 12px!important;}
div[data-testid="stTextArea"] textarea:focus{
  border-color:#1D4ED8!important;background:#fff!important;
  box-shadow:0 0 0 3px rgba(29,78,216,.07)!important;outline:none!important;}
div[data-testid="stTextArea"] textarea::placeholder{color:#94A3B8!important;-webkit-text-fill-color:#94A3B8!important;}
div[data-testid="stSelectbox"]>div>div{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;
  border-radius:8px!important;color:#0F172A!important;font-size:13px!important;}
div[data-testid="stSelectbox"]>div>div *{color:#0F172A!important;}

div[data-testid="stDateInput"] label,
div[data-testid="stMultiSelect"] label{
  font-size:12px!important;font-weight:600!important;color:#374151!important;}
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="base-input"]{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;border-radius:8px!important;}
div[data-testid="stDateInput"] input{
  background:#F8FAFC!important;color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;
  font-size:13.5px!important;font-weight:500!important;}
div[data-testid="stDateInput"] input::placeholder{color:#94A3B8!important;-webkit-text-fill-color:#94A3B8!important;}
div[data-testid="stDateInput"] svg{fill:#64748B!important;}
div[data-testid="stMultiSelect"] [data-baseweb="select"]>div{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;border-radius:8px!important;}
div[data-testid="stMultiSelect"] input{color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;}
div[data-baseweb="calendar"],div[data-baseweb="calendar"] *{
  background-color:#FFFFFF!important;color:#0F172A!important;}
div[data-baseweb="calendar"] [aria-selected="true"],
div[data-baseweb="calendar"] [aria-selected="true"] *{
  background-color:#1D4ED8!important;color:#FFFFFF!important;border-radius:6px!important;}
div[data-baseweb="calendar"] [aria-disabled="true"],
div[data-baseweb="calendar"] [aria-disabled="true"] *{color:#CBD5E1!important;}

div[data-testid="stCheckbox"]{padding:3px 0!important;}
div[data-testid="stCheckbox"] input[type="checkbox"]{
  width:18px!important;height:18px!important;
  accent-color:#1D4ED8!important;
  cursor:pointer!important;
  opacity:1!important;
  -webkit-appearance:checkbox!important;
  appearance:checkbox!important;
}
div[data-testid="stCheckbox"] label{
  display:flex!important;align-items:center!important;gap:8px!important;cursor:pointer!important;}
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label span,
div[data-testid="stCheckbox"] label div{
  color:#111827!important;font-size:13px!important;font-weight:500!important;
  background:transparent!important;}

div[data-testid="stFormSubmitButton"] button,
div.stButton>button[kind="primary"]{
  background:#1D4ED8!important;color:#fff!important;
  font-size:14px!important;font-weight:600!important;
  border:none!important;border-radius:10px!important;
  padding:11px 20px!important;width:100%!important;
  box-shadow:0 2px 8px rgba(29,78,216,.28)!important;
  transition:all .18s!important;}
div[data-testid="stFormSubmitButton"] button *,
div.stButton>button[kind="primary"] *{color:#fff!important;}
div[data-testid="stFormSubmitButton"] button:hover,
div.stButton>button[kind="primary"]:hover{
  background:#1E40AF!important;
  box-shadow:0 6px 18px -2px rgba(29,78,216,.4)!important;
  transform:translateY(-1px)!important;}
div[data-testid="stFormSubmitButton"] button:active,
div.stButton>button[kind="primary"]:active{
  transform:translateY(0)!important;}
div.stButton>button[kind="secondary"]{
  background:#F1F5F9!important;color:#374151!important;
  font-size:14px!important;font-weight:600!important;
  border:1.5px solid #E2E8F0!important;border-radius:8px!important;
  padding:10px 20px!important;width:100%!important;}
div.stButton>button[kind="secondary"] *{color:#374151!important;}
div.stButton>button[kind="secondary"]:hover{background:#E2E8F0!important;color:#0F172A!important;}

div[data-testid="stSuccess"]{background:#F0FDF4!important;border:1.5px solid #86EFAC!important;border-radius:8px!important;color:#166534!important;}
div[data-testid="stSuccess"] *{color:#166534!important;}
div[data-testid="stError"]{background:#FEF2F2!important;border:1.5px solid #FECACA!important;border-radius:8px!important;color:#991B1B!important;}
div[data-testid="stError"] *{color:#991B1B!important;}
div[data-testid="stWarning"]{background:#FFFBEB!important;border:1.5px solid #FDE68A!important;border-radius:8px!important;color:#92400E!important;}
div[data-testid="stWarning"] *{color:#92400E!important;}
div[data-testid="stInfo"]{background:#EFF6FF!important;border:1px solid #BFDBFE!important;border-radius:12px!important;color:#1E40AF!important;padding:14px 18px!important;margin-bottom:2px!important;box-shadow:0 1px 2px rgba(29,78,216,.04)!important;}
div[data-testid="stInfo"] *{color:#1E40AF!important;line-height:1.55!important;}
div[data-testid="stInfo"] p{margin-bottom:6px!important;}
div[data-testid="stInfo"] p:last-child{margin-bottom:0!important;}
section[data-testid="stSidebar"]{background:#fff!important;border-right:1.5px solid #E2E8F0!important;}
section[data-testid="stSidebar"] *{color:#374151!important;}

.ns-sb-card{background:linear-gradient(135deg,#1E3A8A,#1D4ED8 65%,#3B82F6);
  border-radius:14px;padding:16px;margin:2px 0 4px;color:#fff!important;}
.ns-sb-card *{color:#fff!important;}
.stApp .ns-sb-card, .stApp .ns-sb-card *{color:#fff!important;}
.ns-sb-top{display:flex;align-items:center;gap:10px;margin-bottom:12px;}
.ns-sb-av{width:38px;height:38px;border-radius:50%;background:rgba(255,255,255,.22);
  border:1.5px solid rgba(255,255,255,.35);display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:15px;flex-shrink:0;}
.ns-sb-name{font-size:14.5px;font-weight:700;letter-spacing:-.2px;line-height:1.25;}
.ns-sb-role{font-size:10.5px;opacity:.75;font-weight:500;text-transform:uppercase;letter-spacing:.4px;}
.ns-sb-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;}
.ns-sb-item{background:rgba(255,255,255,.12);border-radius:8px;padding:6px 9px;}
.ns-sb-item span{display:block;font-size:9.5px;opacity:.7;text-transform:uppercase;letter-spacing:.4px;margin-bottom:1px;}
.ns-sb-item b{font-size:12.5px;font-weight:700;}
.ns-sb-date{font-size:11px;opacity:.7;border-top:1px solid rgba(255,255,255,.2);padding-top:8px;}

div[data-testid="stExpander"],
div[class*="stExpander"]{background:#fff!important;border:1px solid #E2E8F0!important;border-radius:12px!important;overflow:hidden;margin-top:12px!important;box-shadow:0 1px 2px rgba(15,23,42,.04)!important;}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] details{background:#fff!important;color:#0F172A!important;font-weight:600!important;padding:12px 16px!important;}
div[data-testid="stExpander"] summary:hover{background:#F8FAFC!important;}
div[data-testid="stExpander"] summary svg{fill:#0F172A!important;}
div[data-testid="stExpander"] *{color:#0F172A!important;}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"]{background:#fff!important;}

div[data-testid="stFileUploader"]{background:transparent!important;}
div[data-testid="stFileUploader"] section,
div[data-testid^="stFileUploaderDropzone"],
div[class*="stFileUploaderDropzone"]{background:#F8FAFC!important;border:1.5px dashed #CBD5E1!important;border-radius:10px!important;}
div[data-testid="stFileUploader"] section *,
div[data-testid^="stFileUploaderDropzone"] *{color:#374151!important;}
div[data-testid="stFileUploader"] svg{fill:#94A3B8!important;}
div[data-testid="stFileUploader"] button{background:#1D4ED8!important;color:#fff!important;border:none!important;border-radius:8px!important;}
div[data-testid="stFileUploader"] button *{color:#fff!important;}
div[data-testid="stFileUploader"] button:hover{background:#1E40AF!important;}
div[data-testid="stFileUploaderFile"],
div[class*="stFileUploaderFile"]{background:#F0F4F9!important;border-radius:8px!important;}
div[data-testid="stFileUploaderFile"] *{color:#0F172A!important;}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span{color:#0F172A!important;}
div[data-testid="stRadio"] > label > div:first-child p{
  font-size:12px!important;font-weight:600!important;color:#374151!important;
}
div[data-testid="stRadio"] label [role="radio"]{
  border-color:#94A3B8!important;
  background:#fff!important;
}
div[data-testid="stRadio"] label [role="radio"][aria-checked="true"]{
  border-color:#1D4ED8!important;
  background:#1D4ED8!important;
}

.ns-success{
  background:#fff;border:1px solid #E2E8F0;border-radius:20px;
  padding:52px 36px;text-align:center;max-width:560px;margin:20px auto 8px;
  box-shadow:0 1px 3px rgba(15,23,42,.04), 0 16px 40px -18px rgba(34,197,94,.20);
  animation:ns-pop .35s ease-out;
}
@keyframes ns-pop{
  0%{opacity:0;transform:translateY(8px) scale(.98);}
  100%{opacity:1;transform:translateY(0) scale(1);}
}
.ns-success-ic{
  width:76px;height:76px;border-radius:50%;
  background:linear-gradient(135deg,#22C55E,#16A34A);
  color:#fff!important;font-size:36px;font-weight:800;line-height:76px;
  display:block;margin:0 auto 22px;
  box-shadow:0 10px 24px -8px rgba(22,163,74,.5);
}
.ns-success-title{
  font-size:23px;font-weight:800;color:#0F172A!important;
  margin-bottom:12px;letter-spacing:-.3px;
}
.ns-success-sub{
  font-size:13.5px;color:#64748B!important;line-height:1.7;
  max-width:420px;margin:0 auto;
}

.ns-emergency{
  background:#FEF2F2;border:2px solid #FCA5A5;border-radius:14px;
  padding:16px 20px;margin-bottom:16px;
}
.ns-emergency *{color:#991B1B!important;}
.ns-emergency b{font-size:13.5px;}
.ns-emergency p{font-size:12.5px;margin:4px 0 0;line-height:1.6;}

.ns-qsection{
  display:inline-flex;align-items:center;
  font-size:12px;font-weight:700;color:#1D4ED8;letter-spacing:.2px;
  text-transform:uppercase;margin:26px 0 12px;padding:6px 14px;
  background:#EFF6FF;border-radius:20px;border:1px solid #DBEAFE;
}
.ns-qsection:first-child{margin-top:0;}

div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stNumberInput"]),
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stSelectbox"]),
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]){
  background:#fff!important;border:1px solid #E7ECF3!important;border-radius:14px!important;
  padding:13px 17px 15px!important;margin-bottom:9px!important;
  box-shadow:0 1px 2px rgba(15,23,42,.035)!important;
  transition:border-color .18s, box-shadow .18s, transform .18s!important;
}
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stNumberInput"]):hover,
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stSelectbox"]):hover,
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]):hover{
  border-color:#93C5FD!important;
  box-shadow:0 4px 14px -4px rgba(29,78,216,.16)!important;
}
div[data-testid="stNumberInput"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stRadio"] label p{
  font-size:13.5px!important;font-weight:600!important;color:#1E293B!important;
}
div.st-key-lifestyle_wrap div[data-testid="stHorizontalBlock"]{
  gap:14px!important;align-items:flex-start!important;
}

div[data-testid="stNumberInput"] input{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;
  border-radius:8px 0 0 8px!important;color:#0F172A!important;
  font-size:13px!important;height:40px!important;
}
div[data-testid="stNumberInput"] input:focus{
  border-color:#1D4ED8!important;background:#fff!important;
  box-shadow:0 0 0 3px rgba(29,78,216,.07)!important;
}
div[data-testid="stNumberInput"] button{
  background:#F1F5F9!important;border:1.5px solid #E2E8F0!important;
  color:#374151!important;width:34px!important;height:40px!important;
  min-width:34px!important;box-shadow:none!important;
}
div[data-testid="stNumberInput"] button:hover{background:#E2E8F0!important;}
div[data-testid="stNumberInput"] button svg{fill:#374151!important;width:14px!important;height:14px!important;}
div[data-testid="stNumberInput"]{max-width:220px!important;}

/* date picker, multiselect and dropdown pop-ups: always light, readable (they turned dark in dark-mode browsers) */
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="base-input"],
div[data-testid="stDateInput"] input{
  background:#F8FAFC!important;color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;
  caret-color:#0F172A!important;}
div[data-testid="stDateInput"] [data-baseweb="input"]{
  border:1.5px solid #E2E8F0!important;border-radius:8px!important;}
div[data-testid="stDateInput"] input::placeholder{color:#94A3B8!important;-webkit-text-fill-color:#94A3B8!important;}
div[data-testid="stDateInput"] svg{fill:#64748B!important;}
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div{
  background:#F8FAFC!important;border:1.5px solid #E2E8F0!important;border-radius:8px!important;}
div[data-testid="stMultiSelect"] input{color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;}
div[data-testid="stMultiSelect"] svg{fill:#64748B!important;}
span[data-baseweb="tag"]{background:#DBEAFE!important;border-radius:6px!important;}
span[data-baseweb="tag"] *{color:#1E3A8A!important;-webkit-text-fill-color:#1E3A8A!important;}
[data-baseweb="popover"],[data-baseweb="popover"] > div,[data-baseweb="menu"],
[data-baseweb="calendar"],[data-baseweb="calendar"] *{
  background:#FFFFFF!important;color:#0F172A!important;-webkit-text-fill-color:#0F172A!important;}
[data-baseweb="menu"] li,[role="listbox"] li,[role="option"]{background:#FFFFFF!important;color:#0F172A!important;}
[data-baseweb="menu"] li:hover,[role="option"]:hover,[role="option"][aria-selected="true"]{background:#EFF6FF!important;}
[data-baseweb="calendar"] [aria-selected="true"],
[data-baseweb="calendar"] [aria-selected="true"] *{background:#1D4ED8!important;color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;}
[data-baseweb="calendar"] svg{fill:#0F172A!important;}

/* ═════════════ 3-D DEPTH LAYER (added last so it overrides the flat styles above) ═════════════ */

/* soft lit background with depth */
.stApp{
  background:
    radial-gradient(1100px 520px at 8% -8%, #DCE8FF 0%, rgba(220,232,255,0) 62%),
    radial-gradient(900px 480px at 100% 0%, #D9F1FF 0%, rgba(217,241,255,0) 58%),
    linear-gradient(180deg,#EEF3FB 0%,#E6EDF7 100%) !important;
  background-attachment:fixed !important;
}

/* top bar floats above the page */
.ns-bar{
  background:linear-gradient(180deg,#FFFFFF 0%,#F4F7FC 100%) !important;
  border-bottom:1px solid #D9E2F0 !important;
  box-shadow:0 1px 0 #FFFFFF inset, 0 10px 24px -12px rgba(15,23,42,.35), 0 2px 4px rgba(15,23,42,.06) !important;
}
.ns-brand-ic{
  background:radial-gradient(circle at 30% 22%, #93C5FD 0%, #3B82F6 38%, #1D4ED8 100%) !important;
  box-shadow:0 8px 14px -4px rgba(29,78,216,.55), inset 0 2px 3px rgba(255,255,255,.55), inset 0 -3px 5px rgba(0,0,0,.28) !important;
}

/* raised cards: layered shadows + light edge on top, subtle shade at the bottom */
.ns-hf-top, .ns-kc, .ns-mod, .ns-hcard, .ns-dis, .ns-success,
div[data-testid="stExpander"], div[class*="stExpander"], div.st-key-capture_card,
div[data-testid="stMetric"], div[data-testid="stChatMessage"]{
  background:linear-gradient(180deg,#FFFFFF 0%,#F6F9FD 100%) !important;
  border:1px solid #DCE5F2 !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,1),
    inset 0 -2px 0 rgba(15,23,42,.035),
    0 1px 1px rgba(15,23,42,.06),
    0 4px 8px -2px rgba(15,23,42,.08),
    0 18px 32px -14px rgba(30,64,175,.30) !important;
}
.ns-kc, .ns-mod, .ns-hcard, .ns-dis{
  transition:transform .22s ease, box-shadow .22s ease !important;
  -webkit-transform:perspective(900px) rotateX(0deg) translateY(0);
  transform:perspective(900px) rotateX(0deg) translateY(0);
}
.ns-kc:hover, .ns-mod:hover, .ns-hcard:hover, .ns-dis:hover{
  -webkit-transform:perspective(900px) rotateX(2.2deg) translateY(-5px);
  transform:perspective(900px) rotateX(2.2deg) translateY(-5px);
  box-shadow:
    inset 0 1px 0 #FFFFFF, inset 0 -2px 0 rgba(15,23,42,.04),
    0 2px 2px rgba(15,23,42,.06), 0 12px 18px -6px rgba(15,23,42,.14),
    0 30px 44px -18px rgba(29,78,216,.42) !important;
  border-color:#B6CCF5 !important;
}

/* hero banner: glossy 3-D slab */
.ns-welc{
  position:relative;
  background:linear-gradient(135deg,#1E3A8A 0%,#1D4ED8 55%,#4C8DF6 100%) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.35), inset 0 -3px 0 rgba(0,0,0,.18),
    0 4px 6px rgba(15,23,42,.12), 0 26px 44px -18px rgba(30,58,138,.65) !important;
  overflow:hidden;
}
.ns-welc:before{
  content:""; position:absolute; left:0; right:0; top:0; height:48%;
  background:linear-gradient(180deg,rgba(255,255,255,.20),rgba(255,255,255,0));
  border-radius:14px 14px 0 0; pointer-events:none;
}
.nw-box{
  background:linear-gradient(180deg,rgba(255,255,255,.28),rgba(255,255,255,.10)) !important;
  border:1px solid rgba(255,255,255,.38) !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55), inset 0 -2px 4px rgba(0,0,0,.18), 0 10px 18px -8px rgba(0,0,0,.45) !important;
}

/* round "sphere" badges */
.ns-av, .ns-sb-av{
  background:radial-gradient(circle at 30% 24%, #A5C8FF 0%, #3B82F6 36%, #1D4ED8 100%) !important;
  box-shadow:0 8px 14px -4px rgba(29,78,216,.6), inset 0 2px 3px rgba(255,255,255,.55), inset 0 -3px 6px rgba(0,0,0,.30) !important;
}
.ns-hrk{
  background:radial-gradient(circle at 30% 24%, #FFFFFF 0%, #DCE9FF 55%, #BBD3FA 100%) !important;
  box-shadow:0 5px 9px -3px rgba(29,78,216,.40), inset 0 -2px 3px rgba(29,78,216,.18) !important;
}
.ns-success-ic{
  background:radial-gradient(circle at 30% 24%, #86EFAC 0%, #22C55E 40%, #15803D 100%) !important;
  box-shadow:0 14px 26px -8px rgba(21,128,61,.65), inset 0 3px 4px rgba(255,255,255,.55), inset 0 -5px 8px rgba(0,0,0,.28) !important;
}

/* patient card (left panel) */
.ns-sb-card{
  background:linear-gradient(145deg,#2347B8 0%,#1D4ED8 50%,#4C8DF6 100%) !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.35), inset 0 -3px 0 rgba(0,0,0,.18),
             0 4px 6px rgba(15,23,42,.12), 0 22px 36px -16px rgba(30,58,138,.65) !important;
}
.ns-sb-item{
  background:linear-gradient(180deg,rgba(255,255,255,.20),rgba(255,255,255,.08)) !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.30), inset 0 -2px 3px rgba(0,0,0,.15) !important;
}

/* big raised buttons that physically press down */
div.stButton, div[data-testid="stFormSubmitButton"]{ padding-bottom:6px !important; }
div[data-testid="stFormSubmitButton"] button,
div.stButton>button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(180deg,#4D90F8 0%,#2563EB 48%,#1D4ED8 100%) !important;
  border:1px solid #1E40AF !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.45), 0 5px 0 #1E3A8A, 0 12px 18px -6px rgba(29,78,216,.60) !important;
  transition:transform .12s ease, box-shadow .12s ease, filter .12s ease !important;
  text-shadow:0 1px 0 rgba(0,0,0,.25);
}
div[data-testid="stFormSubmitButton"] button:hover,
div.stButton>button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  -webkit-transform:translateY(-2px); transform:translateY(-2px); filter:brightness(1.06);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.5), 0 7px 0 #1E3A8A, 0 18px 24px -8px rgba(29,78,216,.65) !important;
}
div[data-testid="stFormSubmitButton"] button:active,
div.stButton>button[kind="primary"]:active,
button[data-testid="stBaseButton-primary"]:active,
button[data-testid="stBaseButton-primaryFormSubmit"]:active{
  -webkit-transform:translateY(4px); transform:translateY(4px);
  box-shadow:inset 0 2px 4px rgba(0,0,0,.25), 0 1px 0 #1E3A8A, 0 3px 6px -2px rgba(29,78,216,.5) !important;
}
div.stButton>button[kind="secondary"],
button[data-testid="stBaseButton-secondary"]{
  background:linear-gradient(180deg,#FFFFFF 0%,#EDF2F9 100%) !important;
  border:1px solid #D2DCEB !important;
  box-shadow:inset 0 1px 0 #FFFFFF, 0 4px 0 #C9D4E5, 0 10px 16px -8px rgba(15,23,42,.35) !important;
  transition:transform .12s ease, box-shadow .12s ease !important;
}
div.stButton>button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover{
  -webkit-transform:translateY(-2px); transform:translateY(-2px);
  background:linear-gradient(180deg,#FFFFFF 0%,#E6EEFA 100%) !important;
  box-shadow:inset 0 1px 0 #FFFFFF, 0 6px 0 #C2CEE2, 0 16px 22px -10px rgba(29,78,216,.40) !important;
}
div.stButton>button[kind="secondary"]:active,
button[data-testid="stBaseButton-secondary"]:active{
  -webkit-transform:translateY(3px); transform:translateY(3px);
  box-shadow:inset 0 2px 4px rgba(15,23,42,.18), 0 1px 0 #C9D4E5 !important;
}
/* the page you are on looks pressed in */
div.st-key-navpanel div.stButton>button[kind="primary"]{
  -webkit-transform:translateY(3px); transform:translateY(3px);
  box-shadow:inset 0 3px 7px rgba(0,0,0,.35), 0 1px 0 #1E3A8A, 0 0 0 3px rgba(59,130,246,.28) !important;
}

/* sunken (inset) input fields */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"]>div>div,
div[data-testid="stDateInput"] input, div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stMultiSelect"] [data-baseweb="select"]>div{
  background:linear-gradient(180deg,#EEF2F8 0%,#F9FBFE 100%) !important;
  box-shadow:inset 0 2px 5px rgba(15,23,42,.13), inset 0 -1px 0 #FFFFFF, 0 1px 0 rgba(255,255,255,.95) !important;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus{
  background:#FFFFFF !important;
  box-shadow:inset 0 2px 4px rgba(15,23,42,.08), 0 0 0 4px rgba(59,130,246,.22) !important;
}
div[data-testid="stFileUploader"] section, div[data-testid^="stFileUploaderDropzone"]{
  background:linear-gradient(180deg,#EDF1F8,#F8FAFD) !important;
  box-shadow:inset 0 3px 8px rgba(15,23,42,.10), 0 1px 0 #fff !important;
}

/* question tiles in the assessment form */
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stNumberInput"]),
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stSelectbox"]),
div.st-key-lifestyle_wrap div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]){
  background:linear-gradient(180deg,#FFFFFF,#F6F9FD) !important;
  box-shadow:inset 0 1px 0 #fff, inset 0 -2px 0 rgba(15,23,42,.03), 0 2px 3px rgba(15,23,42,.06),
             0 12px 20px -12px rgba(30,64,175,.30) !important;
}

/* chips, tags, links look like little raised keys */
.ns-chip, .ns-htag, .ns-qsection{
  box-shadow:inset 0 1px 0 rgba(255,255,255,.9), 0 2px 0 rgba(15,23,42,.10), 0 6px 10px -6px rgba(15,23,42,.30) !important;
}
.ns-qsection{ background:linear-gradient(180deg,#F3F8FF,#E3EEFF) !important; }
.ns-hl{ box-shadow:inset 0 1px 0 rgba(255,255,255,.85), 0 3px 0 rgba(15,23,42,.14), 0 8px 12px -6px rgba(15,23,42,.35) !important;
        transition:transform .12s ease, box-shadow .12s ease !important; }
.ns-hl:hover{ -webkit-transform:translateY(-2px); transform:translateY(-2px); }
.ns-hl:active{ -webkit-transform:translateY(2px); transform:translateY(2px); box-shadow:inset 0 2px 3px rgba(0,0,0,.2) !important; }
.ns-bb, .ns-dis-b{ box-shadow:inset 0 1px 0 rgba(255,255,255,.85), 0 2px 0 rgba(15,23,42,.12), 0 6px 10px -6px rgba(15,23,42,.35) !important; }

/* progress bars: glass tube with a glossy fill */
div[data-testid="stProgress"] > div > div{
  background:linear-gradient(180deg,#DCE3EE,#EEF2F8) !important;
  box-shadow:inset 0 2px 4px rgba(15,23,42,.20) !important; border-radius:999px !important;
}
div[data-testid="stProgress"] > div > div > div{
  background:linear-gradient(180deg,#7DB1FF 0%,#2563EB 55%,#1D4ED8 100%) !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.6), 0 2px 6px rgba(29,78,216,.5) !important; border-radius:999px !important;
}

/* chat bubbles */
div[data-testid="stChatMessage"]{ border-radius:16px !important; margin-bottom:12px !important; }
div[data-testid="stChatInput"]{
  box-shadow:inset 0 2px 6px rgba(0,0,0,.35), 0 10px 24px -10px rgba(15,23,42,.45) !important; border-radius:14px !important;
}

/* alerts become raised notes */
div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"], div[data-testid="stError"]{
  box-shadow:inset 0 1px 0 rgba(255,255,255,.75), inset 0 -2px 0 rgba(15,23,42,.05), 0 2px 3px rgba(15,23,42,.06),
             0 12px 20px -12px rgba(15,23,42,.35) !important;
}

/* camera / photos float */
div[data-testid="stImage"] img{ border-radius:12px !important; box-shadow:0 4px 8px rgba(15,23,42,.12), 0 18px 30px -14px rgba(15,23,42,.45) !important; }

/* entrance: cards rise gently */
@-webkit-keyframes ns-rise{from{-webkit-transform:translateY(12px)}to{-webkit-transform:translateY(0)}}
@keyframes ns-rise{from{transform:translateY(12px)}to{transform:translateY(0)}}
.ns-hf-top, .ns-welc, .ns-kc, .ns-mod{ -webkit-animation:ns-rise .45s ease both; animation:ns-rise .45s ease both; }

/* accessibility: no motion for people who prefer less */
@media (prefers-reduced-motion: reduce){
  *{ transition:none !important; animation:none !important; }
  .ns-kc:hover, .ns-mod:hover, .ns-hcard:hover, .ns-dis:hover{ transform:none !important; }
}

/* ===== Professional 3D theme (final override block) ===== */
:root{
  --ns-bg:linear-gradient(180deg,#FFFFFF 0%,#F6F9FF 100%);
  --ns-edge:#D9E2F0;
  --ns-sh:0 1px 0 rgba(255,255,255,.95) inset,0 2px 0 var(--ns-edge),0 12px 24px -12px rgba(15,23,42,.20),0 26px 44px -26px rgba(29,78,216,.28);
  --ns-sh-hover:0 1px 0 rgba(255,255,255,.95) inset,0 3px 0 var(--ns-edge),0 18px 32px -12px rgba(15,23,42,.24),0 34px 52px -26px rgba(29,78,216,.36);
}
.stApp{background:linear-gradient(180deg,#EAF1FF 0%,#F1F5F9 360px)!important;}

/* alignment: one 24px page margin everywhere */
.ns-bar{padding-left:24px!important;}
div.st-key-navpanel{padding:8px 0 24px 24px!important;}
div.st-key-dash_wrap,div.st-key-content_wrap,div.st-key-lifestyle_wrap{padding:8px 24px 24px 20px!important;}
div.st-key-topbar_row{background:#fff!important;border-bottom:1.5px solid #E2E8F0;
  box-shadow:0 2px 6px rgba(15,23,42,.08),0 10px 20px -12px rgba(15,23,42,.20)!important;}
div.st-key-topbar_row .ns-bar{box-shadow:none!important;border-bottom:none!important;}

/* left menu: left-aligned labels so icons and text line up */
div.st-key-navpanel div.stButton>button{justify-content:flex-start!important;text-align:left!important;padding:12px 18px!important;}
div.st-key-navpanel div.stButton>button *{text-align:left!important;justify-content:flex-start!important;}
.ns-sb-card{box-shadow:0 2px 0 #1E3A8A,0 14px 26px -10px rgba(30,58,138,.55)!important;}

/* cards: raised, edged, tilt on hover */
.ns-kc,.ns-dis,.ns-hcard,.ns-hf-top,.ns-mod,div.st-key-capture_card,.stat-row,
div[data-testid="stExpander"],div[data-testid="stMetric"]{
  background:var(--ns-bg)!important;border:1px solid #E2E8F0!important;border-radius:16px!important;
  box-shadow:var(--ns-sh)!important;transition:transform .18s ease,box-shadow .18s ease!important;}
.ns-kc:hover,.ns-dis:hover,.ns-hcard:hover,.ns-mod:hover{
  transform:perspective(900px) translateY(-4px) rotateX(1.5deg)!important;box-shadow:var(--ns-sh-hover)!important;}
.ns-welc{box-shadow:0 3px 0 #1E3A8A,0 18px 34px -10px rgba(30,58,138,.55)!important;}

/* expanders: same height, spacing and padding as the other cards */
div[data-testid="stExpander"]{margin-bottom:14px!important;overflow:hidden!important;}
div[data-testid="stExpander"] details{border:none!important;background:transparent!important;}
div[data-testid="stExpander"] summary{padding:16px 20px!important;}

/* 3D buttons */
div.stButton>button[kind="primary"],div[data-testid="stFormSubmitButton"] button{
  background:linear-gradient(180deg,#3B82F6 0%,#1D4ED8 100%)!important;border:none!important;border-radius:12px!important;
  box-shadow:0 1px 0 rgba(255,255,255,.35) inset,0 4px 0 #1E3A8A,0 12px 20px -6px rgba(29,78,216,.50)!important;
  transition:transform .12s ease,box-shadow .12s ease!important;}
div.stButton>button[kind="primary"]:hover,div[data-testid="stFormSubmitButton"] button:hover{transform:translateY(-1px)!important;}
div.stButton>button[kind="primary"]:active,div[data-testid="stFormSubmitButton"] button:active{
  transform:translateY(3px)!important;box-shadow:0 1px 0 #1E3A8A,0 4px 8px -2px rgba(29,78,216,.40)!important;}
div.stButton>button[kind="secondary"]{
  background:linear-gradient(180deg,#FFFFFF 0%,#F1F5F9 100%)!important;border:1px solid #E2E8F0!important;border-radius:12px!important;
  box-shadow:0 1px 0 #fff inset,0 3px 0 #D5DEEB,0 10px 16px -8px rgba(15,23,42,.20)!important;
  transition:transform .12s ease,box-shadow .12s ease!important;}
div.stButton>button[kind="secondary"]:hover{transform:translateY(-1px)!important;}
div.stButton>button[kind="secondary"]:active{transform:translateY(2px)!important;box-shadow:0 1px 0 #D5DEEB!important;}

/* front page: hero, headline, stat cards, auth card */
div.st-key-home_wrap{padding:36px 56px 40px!important;
  background:radial-gradient(circle at 12% 18%,rgba(59,130,246,.14),transparent 42%),
             radial-gradient(circle at 92% 8%,rgba(14,165,233,.14),transparent 38%)!important;}
div.st-key-home_wrap > div > div[data-testid="stHorizontalBlock"]{gap:56px!important;align-items:flex-start!important;}
div.st-key-home_wrap div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"],
div.st-key-home_wrap div[data-testid="column"] div[data-testid="stHorizontalBlock"]{gap:14px!important;}
div.st-key-home_wrap h2{font-size:44px!important;line-height:1.12!important;font-weight:800!important;letter-spacing:-1px!important;}
div[data-testid="stMetric"]{padding:16px 18px!important;text-align:center!important;}
div[data-testid="stMetric"] *{justify-content:center!important;text-align:center!important;}
div[data-testid="stMetricValue"]{font-size:32px!important;font-weight:800!important;color:#1D4ED8!important;}
div.st-key-auth_card{background:var(--ns-bg)!important;border:1px solid #E2E8F0!important;border-radius:22px!important;
  padding:28px 30px 20px!important;box-shadow:var(--ns-sh-hover)!important;}
div.st-key-auth_card div[data-testid="stForm"]{border:none!important;padding:0!important;}

/* inputs: remove the black password-eye block */
div[data-testid="stTextInput"] [data-baseweb="input"]{background:#fff!important;border-radius:10px!important;overflow:hidden!important;}
div[data-testid="stTextInput"] [data-baseweb="input"] > div{background:#fff!important;}
div[data-testid="stTextInput"] [data-baseweb="input"] button{background:transparent!important;}
div[data-testid="stTextInput"] [data-baseweb="input"] svg{fill:#64748B!important;color:#64748B!important;}

/* ═══════════════════════════════════════════════════
   RESPONSIVE — TABLET (≤900px)
   ═══════════════════════════════════════════════════ */
@media (max-width:900px){
  div.st-key-home_wrap{padding:20px 16px 28px!important;}
  div.st-key-home_wrap h2{font-size:32px!important;}
}

/* ═══════════════════════════════════════════════════
   RESPONSIVE — MOBILE (≤640px)
   Covers both the login/register home page AND the
   logged-in shell (nav + content columns).
   ═══════════════════════════════════════════════════ */
@media (max-width:640px){

  /* ── 0. Global overflow guard ── */
  html, body,
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMainBlockContainer"],
  [data-testid="stVerticalBlock"]{
    overflow-x:hidden!important;
    max-width:100vw!important;
  }

  /* ── 1. Topbar ── */
  .ns-bar{padding:0 12px!important;height:52px!important;}
  .ns-brand-nm{font-size:15px!important;}

  /* ── 2. HOME PAGE — hero + auth card ──────────────────────
     The outer home_wrap container */
  div.st-key-home_wrap{
    padding:12px 10px 20px!important;
    overflow-x:hidden!important;
  }

  /* Stack the left-hero and right-auth columns vertically.
     Streamlit sets width AND flex via INLINE styles on the column div.
     We must use !important on every axis to override them. */
  div.st-key-home_wrap [data-testid="stHorizontalBlock"]{
    flex-direction:column!important;
    flex-wrap:wrap!important;
    gap:0!important;
    align-items:stretch!important;
    width:100%!important;
    max-width:100%!important;
  }
  /* target both the outer column wrapper and its immediate first child
     (Streamlit wraps column content in an extra div) */
  div.st-key-home_wrap [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"],
  div.st-key-home_wrap [data-testid="stHorizontalBlock"]
    > [data-testid="column"],
  div.st-key-home_wrap [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"] > div,
  div.st-key-home_wrap [data-testid="stHorizontalBlock"]
    > [data-testid="column"] > div{
    flex:1 1 100%!important;
    width:100%!important;
    min-width:0!important;
    max-width:100%!important;
    box-sizing:border-box!important;
  }

  /* Hero text */
  div.st-key-home_wrap h2{
    font-size:26px!important;
    line-height:1.2!important;
    letter-spacing:-.5px!important;
  }
  div.st-key-home_wrap p{font-size:14px!important;line-height:1.55!important;}

  /* Hide the 3-column metric stats row on mobile */
  div.st-key-home_wrap [data-testid="stMetric"]{display:none!important;}
  /* Also hide the stHorizontalBlock that ONLY contains metrics
     (the stats row — 3rd stHorizontalBlock inside the left col) */
  div.st-key-home_wrap [data-testid="stColumn"]:first-child
    [data-testid="stHorizontalBlock"]:last-child,
  div.st-key-home_wrap [data-testid="column"]:first-child
    [data-testid="stHorizontalBlock"]:last-child{
    display:none!important;
  }

  /* ── 3. AUTH CARD (login / register form) ── */
  div.st-key-auth_card{
    padding:16px 12px 14px!important;
    border-radius:14px!important;
    width:100%!important;
    box-sizing:border-box!important;
    margin-top:8px!important;
  }

  /* Sign In / Register toggle buttons row — keep side by side, fit screen */
  div.st-key-auth_card > div > [data-testid="stHorizontalBlock"]:first-child,
  div.st-key-auth_card > div > [data-testid="stHorizontalBlock"]:first-child{
    flex-direction:row!important;
    gap:8px!important;
    flex-wrap:nowrap!important;
    width:100%!important;
  }
  div.st-key-auth_card > div > [data-testid="stHorizontalBlock"]:first-child
    > [data-testid="stColumn"],
  div.st-key-auth_card > div > [data-testid="stHorizontalBlock"]:first-child
    > [data-testid="column"]{
    flex:1 1 0!important;
    min-width:0!important;
    max-width:50%!important;
  }

  /* All buttons inside auth card */
  div.st-key-auth_card div.stButton > button{
    width:100%!important;
    min-width:0!important;
    font-size:13px!important;
    padding:10px 4px!important;
    white-space:nowrap!important;
    overflow:hidden!important;
    text-overflow:ellipsis!important;
  }

  /* Form inputs and selects */
  div.st-key-auth_card [data-testid="stTextInput"],
  div.st-key-auth_card [data-testid="stTextInput"] input,
  div.st-key-auth_card [data-testid="stSelectbox"],
  div.st-key-auth_card [data-baseweb="input"]{
    width:100%!important;
    max-width:100%!important;
    box-sizing:border-box!important;
  }

  /* Submit buttons (Sign in →, Continue →, Create account →) */
  div.st-key-auth_card [data-testid="stFormSubmitButton"] button{
    width:100%!important;
    min-width:0!important;
    font-size:14px!important;
  }

  /* Reg step-2: back + continue cols → stack vertically */
  div.st-key-auth_card [data-testid="stHorizontalBlock"]:not(:first-child){
    flex-wrap:wrap!important;
  }
  div.st-key-auth_card [data-testid="stHorizontalBlock"]:not(:first-child)
    > [data-testid="stColumn"],
  div.st-key-auth_card [data-testid="stHorizontalBlock"]:not(:first-child)
    > [data-testid="column"]{
    flex:1 1 100%!important;
    min-width:0!important;
    max-width:100%!important;
  }

  /* ── 4. LOGGED-IN SHELL — nav + content ──────────────────
     st.columns([1, 4.2]) creates a narrow nav + wide content.
     On mobile we stack them: nav on top, content below.
     Must override Streamlit's inline width/flex styles. */
  div.st-key-shell > div > [data-testid="stHorizontalBlock"]{
    flex-direction:column!important;
    flex-wrap:wrap!important;
    gap:0!important;
    width:100%!important;
    max-width:100%!important;
  }
  div.st-key-shell > div > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"],
  div.st-key-shell > div > [data-testid="stHorizontalBlock"]
    > [data-testid="column"],
  div.st-key-shell > div > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"] > div,
  div.st-key-shell > div > [data-testid="stHorizontalBlock"]
    > [data-testid="column"] > div{
    flex:1 1 100%!important;
    width:100%!important;
    min-width:0!important;
    max-width:100%!important;
    box-sizing:border-box!important;
  }

  /* Nav panel: compact, full width */
  div.st-key-navpanel{
    padding:8px 10px 12px!important;
    width:100%!important;
  }
  div.st-key-navpanel div.stButton > button{
    font-size:14px!important;
    padding:10px 14px!important;
  }

  /* Content/dashboard wrap */
  div.st-key-dash_wrap,
  div.st-key-content_wrap,
  div.st-key-lifestyle_wrap{
    padding:8px 10px 20px!important;
    width:100%!important;
  }

  /* ns-sb-card inside navpanel — show as horizontal strip */
  .ns-sb-card{
    border-radius:12px!important;
    padding:12px!important;
    margin-bottom:8px!important;
  }
  .ns-sb-grid{
    grid-template-columns:repeat(4,1fr)!important;
    gap:6px!important;
  }

  /* Welcome banner in dashboard — stack instead of space-between */
  .ns-welc{
    flex-direction:column!important;
    align-items:flex-start!important;
    gap:10px!important;
    padding:16px 14px!important;
  }
  .nw-box{align-self:stretch!important;text-align:left!important;padding:10px 14px!important;}

  /* Topbar logout button area */
  div.st-key-topbar_row [data-testid="stHorizontalBlock"]{
    align-items:center!important;
  }
  div.st-key-topbar_row [data-testid="stColumn"]:last-child,
  div.st-key-topbar_row [data-testid="column"]:last-child{
    min-width:80px!important;
    max-width:80px!important;
    flex:0 0 80px!important;
  }
}
@media (prefers-reduced-motion: reduce){
  .ns-kc:hover,.ns-dis:hover,.ns-hcard:hover,.ns-mod:hover,div.stButton>button:hover{transform:none!important;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ── HOSPITAL DATABASE (fallback only — live search is primary) ──
HOSP = {
    "mumbai": [
        {"name": "KEM Hospital", "type": "Government", "address": "Acharya Donde Marg, Parel, Mumbai 400012", "phone": "022-24107000", "budget": "low", "tags": ["Nephrology", "Urology", "Dialysis"], "rating": 4.1},
        {"name": "Sion Hospital (LTMGH)", "type": "Government", "address": "Babasaheb Ambedkar Rd, Sion, Mumbai 400022", "phone": "022-24076000", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.9},
        {"name": "Hinduja Hospital", "type": "Private", "address": "Veer Savarkar Marg, Mahim, Mumbai 400016", "phone": "022-24452222", "budget": "high", "tags": ["Nephrology", "Kidney Transplant"], "rating": 4.5},
        {"name": "Kokilaben Ambani Hospital", "type": "Private", "address": "Andheri West, Mumbai 400053", "phone": "022-30661000", "budget": "high", "tags": ["Kidney Transplant", "Nephrology"], "rating": 4.6},
        {"name": "Lilavati Hospital", "type": "Private", "address": "Bandra Reclamation, Bandra West, Mumbai 400050", "phone": "022-26751000", "budget": "high", "tags": ["Nephrology", "Dialysis"], "rating": 4.6},
    ],
    "pune": [
        {"name": "Sassoon General Hospital", "type": "Government", "address": "Pune Station Road, Pune 411001", "phone": "020-26126000", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.8},
        {"name": "Ruby Hall Clinic", "type": "Private", "address": "40, Sassoon Road, Pune 411001", "phone": "020-66455555", "budget": "high", "tags": ["Nephrology", "Dialysis", "Transplant"], "rating": 4.5},
    ],
    "delhi": [
        {"name": "AIIMS New Delhi", "type": "Government", "address": "Ansari Nagar, New Delhi 110029", "phone": "011-26588500", "budget": "low", "tags": ["Nephrology", "Transplant", "Urology"], "rating": 4.7},
        {"name": "Max Hospital Saket", "type": "Private", "address": "Press Enclave Road, Saket, Delhi 110017", "phone": "011-26515050", "budget": "high", "tags": ["Nephrology", "Dialysis"], "rating": 4.5},
    ],
    "bengaluru": [
        {"name": "Victoria Hospital", "type": "Government", "address": "Fort Road, Bengaluru 560002", "phone": "080-26701150", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.7},
        {"name": "Manipal Hospital", "type": "Private", "address": "HAL Airport Road, Bengaluru 560017", "phone": "080-25024444", "budget": "high", "tags": ["Nephrology", "Transplant"], "rating": 4.5},
    ],
    "hyderabad": [
        {"name": "Osmania General Hospital", "type": "Government", "address": "Afzalgunj, Hyderabad 500012", "phone": "040-24535555", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.8},
        {"name": "Apollo Hospitals Hyderabad", "type": "Private", "address": "Jubilee Hills, Hyderabad 500033", "phone": "040-23607777", "budget": "high", "tags": ["Nephrology", "Transplant", "Dialysis"], "rating": 4.5},
    ],
    "chennai": [
        {"name": "Government Stanley Medical College", "type": "Government", "address": "Old Jail Rd, Chennai 600001", "phone": "044-25281151", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.9},
        {"name": "Apollo Hospital Chennai", "type": "Private", "address": "21, Greams Lane, Chennai 600006", "phone": "044-28293333", "budget": "high", "tags": ["Nephrology", "Kidney Transplant"], "rating": 4.6},
    ],
    "nashik": [
        {"name": "Civil Hospital Nashik", "type": "Government", "address": "Dashak Chowk, Nashik 422001", "phone": "0253-2579000", "budget": "low", "tags": ["Nephrology", "Medicine"], "rating": 3.7},
        {"name": "Wockhardt Hospital Nashik", "type": "Private", "address": "Pathardi Phata, Nashik 422010", "phone": "0253-6619999", "budget": "high", "tags": ["Nephrology", "Dialysis"], "rating": 4.3},
    ],
}

ALIAS = {
    "bombay": "mumbai", "greater mumbai": "mumbai",
    "bangalore": "bengaluru", "blr": "bengaluru",
    "new delhi": "delhi", "noida": "delhi", "gurgaon": "delhi", "gurugram": "delhi",
}

BUDGET_MAP = {
    "Low / Government (Free or very low cost)": "low",
    "Mid-range (Rs 500-2000 per visit)": "mid",
    "Premium Private": "high",
    "Show all budgets": None,
}


def _hosp_defaults(h):
    for k, v in (("kind", "Hospital"), ("hours", ""), ("website", ""), ("emergency", False), ("nephro", False),
                 ("distance_km", None), ("directions_url", ""), ("lat", None), ("lon", None)):
        h.setdefault(k, v)
    return h


def _ensure_tiers(results, pool, limit=12):
    """Every search shows at least one Government / free-low-cost hospital AND one Premium hospital
    (when one exists in `pool`), whatever budget was chosen."""
    def key(h):
        d = h.get("distance_km")
        return (0 if h.get("nephro") else 1, 0 if h.get("kind", "Hospital") == "Hospital" else 1,
                d if d is not None else 9999)

    out = list(results)[:limit]
    names = {h.get("name") for h in out}
    for tier in ("low", "high"):
        if any(h.get("budget") == tier for h in out):
            continue
        cand = next((h for h in pool if h.get("budget") == tier and h.get("name") not in names), None)
        if cand is None:
            continue
        if len(out) >= limit:
            for idx in range(len(out) - 1, -1, -1):
                if out[idx].get("budget") not in ("low", "high"):
                    out.pop(idx)
                    break
            else:
                out.pop()
        out.append(cand)
        names.add(cand.get("name"))
    out.sort(key=key)
    return out


def find_hosp(city_in, budget):
    """Live search (Geoapify, OSM fallback). Returns (results, found, matched). If nothing exists in the chosen
    budget, ALL nearby hospitals are returned and matched == "__all__" so the page can say so.
    Results always include at least one Government/low-cost and one Premium hospital when one can be found."""
    from places_api import search_hospitals_live

    bkey = BUDGET_MAP.get(budget)
    live_results = search_hospitals_live(city_in, max_results=60)
    if live_results is not None:
        results, matched = live_results, city_in
        if bkey and live_results:
            filtered = [h for h in live_results if h["budget"] == bkey]
            if filtered:
                results = filtered
            else:
                matched = "__all__"
        pool = list(live_results)
        if live_results:
            for tier in ("low", "high"):
                if not any(h["budget"] == tier for h in pool):      # none nearby -> look wider (150 km)
                    try:
                        wide = search_hospitals_live(city_in, max_results=60, radii=(150000,)) or []
                    except Exception:
                        wide = []
                    have = {h["name"] for h in pool}
                    pool += [h for h in wide if h["budget"] == tier and h["name"] not in have]
        results = _ensure_tiers(results, pool)
        return results, bool(live_results), matched

    k = city_in.strip().lower()
    k = ALIAS.get(k, k)
    if k not in HOSP:
        for hk in HOSP:
            if hk in k or k in hk:
                k = hk
                break
    base = [_hosp_defaults(dict(h)) for h in HOSP.get(k, [])]
    results, matched = base, k
    if bkey and base:
        filtered = [h for h in base if h["budget"] == bkey]
        if filtered:
            results = filtered
        else:
            matched = "__all__"
    results = _ensure_tiers(results, base)
    return results, bool(base), matched


def _status_of(h):
    """(state, text) — 'open' / 'closed' / None — computed fresh every time the page is drawn."""
    try:
        from places_api import open_status
        return open_status(h.get("hours"))
    except Exception:
        return None, "Timing not listed here — check on Google or call to confirm"


def _hospital_card_html(i, h, city):
    esc = lambda v: html.escape(str(v or ""))
    state, status_text = _status_of(h)
    cls = {"open": "open", "closed": "closed"}.get(state, "unk")
    icon = {"open": "🟢", "closed": "🔴"}.get(state, "⏰")
    phone = str(h.get("phone") or "")
    tel = re.sub(r"[^\d+]", "", phone)
    dist = f" · {h['distance_km']} km away" if h.get("distance_km") is not None else ""
    emerg = ' <span class="ns-htag">🚑 Emergency</span>' if h.get("emergency") else ""
    tags = "".join(f'<span class="ns-htag">{esc(t)}</span>' for t in h.get("tags", []) if t != "Emergency") + emerg
    murl = h.get("directions_url") or h.get("maps_url") or (
        "https://www.google.com/maps/search/" + str(h.get("name", "")).replace(" ", "+") + "+" + str(city).replace(" ", "+"))
    links = f'<a class="ns-hl m" href="{esc(murl)}" target="_blank">📍 Directions</a>'
    if tel:
        links += f'<a class="ns-hl c" href="tel:{tel}">📞 Call to book appointment</a>'
    if h.get("website"):
        w = str(h["website"])
        w = w if w.startswith("http") else "https://" + w
        links += f'<a class="ns-hl b" href="{esc(w)}" target="_blank">🌐 Website / book online</a>'
    import urllib.parse as _up
    _gq = _up.quote_plus(f"{h.get('name', '')} {h.get('address', '') or city}")
    links += (f'<a class="ns-hl b" href="https://www.google.com/maps/search/?api=1&query={_gq}" '
              f'target="_blank">🕒 Timings on Google</a>')
    try:
        from places_api import hours_summary
        week = hours_summary(h.get("hours"))
    except Exception:
        week = ""
    htype = str(h.get("type", ""))
    if htype == "Government":
        note = ("Appointment: OPD registration (case paper) is usually done at the hospital counter in the morning, and "
                "the kidney doctor sits only on fixed OPD days — call to confirm the day. Some government hospitals also "
                "offer online registration (ors.gov.in).")
    elif htype.startswith("Charitable"):
        note = ("Appointment: call the hospital desk first — the kidney doctor may visit only on certain days, so book "
                "at least 1 day before if you can. Charges are usually lower here.")
    else:
        note = ("Appointment: kidney-specialist visits are usually by prior appointment — book at least 1 day before "
                "by phone or the hospital website; same-day booking at the reception may not be possible.")
    note += " Timings and doctor availability can change — please call first."
    labels = {"low": "Govt / low cost", "mid": "Mid-range (est.)", "high": "Premium (est.)"}
    classes = {"low": "bb-l", "mid": "bb-m", "high": "bb-h"}
    b = h.get("budget")
    return (
        f'<div class="ns-hcard"><div class="ns-hrk">{i}</div><div style="flex:1">'
        f'<div class="ns-hn">{esc(h.get("name"))}</div>'
        f'<div class="ns-ht">{esc(htype)} {esc(h.get("kind", "Hospital"))}{dist}</div>'
        f'<div class="ns-open {cls}">{icon} {esc(status_text)}</div>'
        + (f'<div class="ns-hp">🕒 {esc(week)}</div>' if week else "")
        + f'<div class="ns-ha">📍 {esc(h.get("address"))}</div>'
        + (f'<div class="ns-hp">📞 {esc(phone)}</div>' if phone else "")
        + f'<div class="ns-htags">{tags}</div><div class="ns-hlinks">{links}</div>'
        f'<div class="ns-hnote">{esc(note)}</div></div>'
        f'<div><div class="ns-bb {classes.get(b, "bb-m")}">{labels.get(b, "")}</div></div></div>'
    )


def topbar(show_user=False):
    u = st.session_state.user
    if show_user and u:
        session_time = st.session_state.get("_session_login_time", "")
        av = (
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="text-align:right;line-height:1.3;">'
            f'<div style="font-size:10px;color:#94A3B8;font-weight:600;text-transform:uppercase;letter-spacing:.4px;">This session</div>'
            f'<div style="font-size:12px;color:#0F172A;font-weight:600;">{session_time}</div>'
            f'</div>'
            f'<div class="ns-av">{u["full_name"][0].upper()}</div>'
            f'</div>'
        )
    else:
        av = ""
    with st.container(key="topbar_row"):
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(
                f"""
            <div class="ns-bar">
                <div class="ns-brand">
                    <div class="ns-brand-ic">🩺</div>
                    <div class="ns-brand-nm">Nephro<b>Sense</b></div>
                </div>
                {av}
            </div>""",
                unsafe_allow_html=True,
            )
        if show_user and u:
            with col2:
                if st.button("Logout", key=f"topbar_logout_{st.session_state.page}"):
                    logout()


def render_nav():
    """Left panel: patient card + vertical navigation (always visible, no st.sidebar needed)."""
    u = st.session_state.user
    today = datetime.now().strftime("%d %b %Y")
    with st.container(key="navpanel"):
        st.markdown(
            f"""
        <div class="ns-sb-card">
            <div class="ns-sb-top">
                <div class="ns-sb-av">{html.escape(u['full_name'][0].upper())}</div>
                <div>
                    <div class="ns-sb-name">{html.escape(u['full_name'].title())}</div>
                    <div class="ns-sb-role">{'Doctor / Clinical Staff' if u.get('role') == 'doctor' else 'Patient'}</div>
                </div>
            </div>
            <div class="ns-sb-grid">
                <div class="ns-sb-item"><span>Age</span><b>{u.get('age', '—')} yrs</b></div>
                <div class="ns-sb-item"><span>Gender</span><b>{html.escape(str(u.get('gender', '—')))}</b></div>
                <div class="ns-sb-item"><span>City</span><b>{html.escape(str(u.get('city') or '—'))}</b></div>
                <div class="ns-sb-item"><span>Blood Group</span><b>{html.escape(str(u.get('blood_group') or '—'))}</b></div>
            </div>
            <div class="ns-sb-date">📅 Last login: {u.get('last_login') or today}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        items = [("🏠 Dashboard", "dashboard"), ("📋 Health Assessment", "lifestyle"),
                 ("👁️ Visual Screening", "eye"), ("💬 NephroBot", "chatbot"),
                 ("🏥 Find Hospital", "find_hospital")]
        for label, page in items:
            if st.button(label, key=f"nav_{page}", use_container_width=True,
                         type="primary" if st.session_state.page == page else "secondary"):
                go(page)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Sign out", key="nav_out", use_container_width=True):
            logout()


def page_home():
    topbar()
    active = st.session_state.auth_tab

    with st.container(key="home_wrap"):
        col_l, col_r = st.columns([1, 1], gap="small")

        with col_l:
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
            st.markdown("🟢 &nbsp; **KIDNEY HEALTH SCREENING PLATFORM**")
            st.markdown("## Your kidney health, screened *early.*")
            st.markdown(
                """
NephroSense provides a personalised kidney health assessment
without any lab visit or blood draw.
Get a detailed result in minutes.
                """
            )
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Global Impact", "850M", help="People with CKD worldwide")
            with c2:
                st.metric("CKD Stages", "5", help="All stages detected")
            with c3:
                st.metric("Research Sources", "3", help="UCI, Bangladesh & NHANES public datasets")

        with col_r:
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

            with st.container(key="auth_card"):
                st.markdown(
                    '<div class="auth-tabs-start"></div>',
                            unsafe_allow_html=True
                        )
                t1, t2 = st.columns([1, 1], gap="small")

                with t1:
                    if st.button("Sign In", key="tl", use_container_width=True,
                                 type="primary" if active == "login" else "secondary"):
                        st.session_state.auth_tab = "login"
                        st.rerun()
                with t2:
                    if st.button("Register", key="tr", use_container_width=True,
                                 type="primary" if active == "register" else "secondary"):
                        st.session_state.auth_tab = "register"
                        st.rerun()

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
       
                if active == "login":
                    _login()
                else:
                    _register()

                st.caption(
                    "NephroSense is a health screening research platform. "
                    "Not a substitute for professional medical diagnosis."
                )


def _login():
    st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Sign in to your account</div>', unsafe_allow_html=True)

    with st.form("lf"):
        email = st.text_input("Email address", placeholder="you@example.com")
        pw = st.text_input("Password", type="password", placeholder="Enter your password")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        sub = st.form_submit_button("Sign in →", use_container_width=True)

    if sub:
        if not email or not pw:
            st.error("Please fill in both fields.")
        else:
            ok, res = login_user(email.strip(), pw)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user = res
                st.session_state["_session_login_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
                st.success(f"Welcome back, {res['full_name'].split()[0]}!")
                time.sleep(0.3)
                go("dashboard" if get_user_history(res["id"]) else "lifestyle")
            else:
                st.error("Incorrect email or password.")

    st.markdown('<div class="divline"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#64748B;margin-bottom:8px;'>New to NephroSense?</div>",
        unsafe_allow_html=True,
    )
    if st.button("Create a free account →", key="sw", use_container_width=True):
        st.session_state.auth_tab = "register"
        st.rerun()


def _register():
    step = st.session_state.reg_step
    d1 = "on" if step == 1 else "done"
    d2 = "on" if step == 2 else ""
    lbl = "Step 1 of 2 — Personal Details" if step == 1 else "Step 2 of 2 — Medical Background"

    st.markdown(
        f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
        <div class="ns-sdot {d1}"></div>
        <div class="ns-sdot {d2}"></div>
        <div style="font-size:12px;color:#64748B!important;font-weight:500;">{lbl}</div>
    </div>
    <style>
    .ns-sdot{{width:8px;height:8px;border-radius:50%;background:#CBD5E1;flex-shrink:0;display:inline-block;}}
    .ns-sdot.on{{background:#1D4ED8;}}.ns-sdot.done{{background:#22C55E;}}
    </style>
    """,
        unsafe_allow_html=True,
    )

    if step == 1:
        _reg1()
    else:
        _reg2()


def _reg1():
    st.markdown('<div class="auth-title">Create account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Personal details — under 2 minutes</div>', unsafe_allow_html=True)

    with st.form("r1"):
        name = st.text_input("Full name *", placeholder="Your full name")
        email = st.text_input("Email address *", placeholder="you@example.com")
        phone = st.text_input("Mobile number *", placeholder="+91 98765 43210")

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender *", ["Select", "Female", "Male", "Other", "Prefer not to say"])
        with c2:
            dob = st.text_input("Date of birth *", placeholder="DD/MM/YYYY")

        c3, c4 = st.columns(2)
        with c3:
            pw = st.text_input("Password *", type="password", placeholder="Min 8 characters")
        with c4:
            pw2 = st.text_input("Confirm *", type="password", placeholder="Repeat password")

        city = st.text_input("City / Village / Town *", placeholder="e.g. Boisar, Wada, Thane, Pune")
        role = st.selectbox("I am", ["Patient", "Doctor / Clinical Staff", "Caregiver (for family)"])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        sub = st.form_submit_button("Continue →", use_container_width=True)

    if sub:
        errs = []
        if not name.strip():
            errs.append("Full name is required.")
        if not email.strip():
            errs.append("Email address is required.")
        if not phone.strip():
            errs.append("Mobile number is required.")
        if gender == "Select":
            errs.append("Please select your gender.")
        if not dob.strip():
            errs.append("Date of birth is required.")
        if len(pw) < 8:
            errs.append("Password must be at least 8 characters.")
        if pw != pw2:
            errs.append("Passwords do not match.")
        if not city.strip():
            errs.append("City / Town is required.")

        if errs:
            for e in errs:
                st.error(e)
        else:
            try:
                age = datetime.now().year - int(dob.strip().split("/")[-1])
            except Exception:
                age = 0

            st.session_state["_r"] = dict(
                name=name.strip(), email=email.strip(), phone=phone.strip(), gender=gender,
                dob=dob.strip(), age=age, pw=pw, city=city.strip(),
                role="doctor" if "Doctor" in role else "patient",
            )
            st.session_state.reg_step = 2
            st.rerun()


def _reg2():
    st.markdown('<div class="auth-title">Medical background</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Personalises your screening results</div>', unsafe_allow_html=True)
    st.markdown('<div class="divline"></div>', unsafe_allow_html=True)

    blood = st.selectbox("Blood group", ["Don't know", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    smk = st.selectbox("Smoking status", ["Non-smoker", "Former smoker", "Current smoker"])
    wtr = st.selectbox("Daily water intake", ["Less than 1 litre", "1-2 litres", "2-3 litres", "More than 3 litres"])

    st.markdown(
        "<div style='font-size:13px;font-weight:700;color:#0F172A!important;margin:10px 0 4px;'>"
        "Existing health conditions — select all that apply</div>",
        unsafe_allow_html=True,
    )
    dm = st.checkbox("Diabetes (Type 1 or Type 2)", key="cb_dm")
    htn = st.checkbox("High blood pressure (Hypertension)", key="cb_htn")
    hrt = st.checkbox("Heart disease or cardiovascular condition", key="cb_hrt")
    ckh = st.checkbox("Kidney disease (personal or family history)", key="cb_ckh")
    uti = st.checkbox("Frequent urinary tract infections (UTIs)", key="cb_uti")
    agree = st.checkbox("I confirm this is a research screening tool, not a clinical diagnosis.", key="cb_agree")

    st.markdown('<div class="divline"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True, key="back2"):
            st.session_state.reg_step = 1
            st.rerun()
    with col2:
        if st.button("Create account →", use_container_width=True, key="submit2", type="primary"):
            if not agree:
                st.warning("Please accept the disclaimer to continue.")
            else:
                r = st.session_state.get("_r", {})
                ok, msg = register_user(
                    full_name=r.get("name", ""), email=r.get("email", ""), password=r.get("pw", ""),
                    gender=r.get("gender", ""), age=r.get("age", 0), role=r.get("role", "patient"),
                    phone=r.get("phone", ""), dob=r.get("dob", ""), blood_group=blood,
                    city=r.get("city", ""), diabetes=dm, hypertension=htn, heart_disease=hrt,
                    kidney_history=ckh, uti_history=uti, smoking=smk, water_intake=wtr,
                )
                if ok:
                    st.success("Account created! Please sign in.")
                    st.session_state.reg_step = 1
                    st.session_state.auth_tab = "login"
                    st.session_state.pop("_r", None)
                    time.sleep(0.6)
                    st.rerun()
                else:
                    st.error(msg)


def page_dashboard():
    u = st.session_state.user
    hist = get_user_history(u["id"])
    scored = [r for r in hist if r.get("risk_score") is not None]
    latest = scored[0] if scored else None

    hr = datetime.now().hour
    g = "Good morning" if hr < 12 else ("Good afternoon" if hr < 17 else "Good evening")

    lr = latest["risk_level"] if latest else None
    sc = latest["risk_score"] if latest else None
    sc_d = f"{sc:.0f}%" if sc is not None else "—"

    components = get_latest_component_scores(u["id"])
    ls, es = components.get("lifestyle_score"), components.get("eye_score")
    done_mods = []
    if ls is not None:
        done_mods.append("Health Assessment")
    if es is not None:
        done_mods.append("Visual Screening")
    modules_done_text = " + ".join(done_mods) if done_mods else "Not started yet"

    notes = _parse_notes(get_latest_clinical_notes(u["id"]))
    preds = notes.get("_predictions") or []
    conf = notes.get("_confidence")

    with st.container(key="dash_wrap"):
        st.markdown(
            f"""
        <div class="ns-welc">
            <div>
                <div class="nw-g">{g}</div>
                <div class="nw-n">{u['full_name']}</div>
                <div class="nw-m">{u['gender']} · {u['age']} yrs · {'Doctor' if u['role'] == 'doctor' else 'Patient'}</div>
                <div class="nw-m">Based on: {modules_done_text}</div>
            </div>
            <div class="nw-box"><div class="nw-v">{sc_d}</div><div class="nw-l">Overall Risk{' · ' + lr if lr else ''}</div></div>
        </div>""",
            unsafe_allow_html=True,
        )

        if lr:
            REC = {
                "HIGH": ("🔴", "Your latest result suggests visiting a nephrologist or hospital soon.", True),
                "MEDIUM": ("🟠", "Consider consulting a doctor in the next few weeks to review these results.", True),
                "LOW": ("🟢", "Your latest result is in the low-risk range — routine checkups are still a good idea.", False),
            }
            icon, msg, show_hosp = REC.get(str(lr).upper(), ("🔵", "Review your results with a doctor.", False))
            st.info(f"{icon} **{msg}** This is a screening estimate, not a diagnosis."
                    + (f"  \nConfidence of this result: **{conf}**" if conf else ""))
            for _nm, _hk, _dk in (("blood", "_have_blood", "_blood_date"), ("urine", "_have_urine", "_urine_date")):
                _d = _iso_to_date(notes.get(_dk))
                if _d and notes.get(_hk):
                    _days = (date.today() - _d).days
                    if _days > 180:
                        st.warning(f"Your {_nm} report is now {age_text(_days)} old (dated {_d.strftime('%d %b %Y')}). "
                                   f"It may no longer reflect your current kidney health — please repeat the test "
                                   f"and update your Health Assessment.")
                    else:
                        st.caption(f"Based on your {_nm} report dated {_d.strftime('%d %b %Y')} ({age_text(_days)} ago).")

            if show_hosp:
                nearby, found, _ = _cached_find_hosp(u.get("city", ""), "Show all budgets")
                if nearby:
                    st.markdown('<div class="ns-lbl">Nearby options</div>', unsafe_allow_html=True)
                    for hosp in nearby[:3]:
                        _stt, _stx = _status_of(hosp)
                        _dist = f" · {hosp['distance_km']} km" if hosp.get("distance_km") is not None else ""
                        st.markdown(
                            f"**{hosp['name']}** · {hosp['type']}{_dist}  \n"
                            f"{ {'open': '🟢', 'closed': '🔴'}.get(_stt, '⏰') } {_stx}  \n"
                            f"📍 {hosp['address']}" + (f"  \n📞 {hosp['phone']}" if hosp.get('phone') else "")
                        )
                        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if st.button("🏥 View all nearby hospitals →", use_container_width=True, key="dash_hosp_btn"):
                    go("find_hospital")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        def _mini_card(icon, label, text, cls=""):
            return (
                f'<div class="ns-kc"><div class="ns-kc-i">{icon}</div>'
                f'<div class="ns-kc-v {cls}">{text}</div>'
                f'<div class="ns-kc-l">{label}</div></div>'
            )

        hb_state, hb_val = _hb_status(u)
        if es is None:
            eye_card = _mini_card("👁️", "Visual Screening (reference only)", "Not done yet")
        elif hb_state == "normal":
            eye_card = _mini_card("👁️", f"Visual Screening — ignore: your blood test shows normal haemoglobin ({hb_val:.1f} g/dL)", "Not reliable")
        else:
            eye_card = _mini_card("👁️", f"Visual Screening · paleness index {es:.0f}/100 (reference only, not in score)",
                                  eye_label(es) or f"{es:.0f}/100", "gr" if es < 33 else "am")
        ls_card = _mini_card("📋", "Health Assessment", f"{ls:.0f}/100" if ls is not None else "Not done yet",
                             "gr" if ls is not None else "")
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:14px;">'
            + ls_card + eye_card + "</div>",
            unsafe_allow_html=True,
        )

        if sc is not None:
            st.markdown('<div class="ns-lbl">Possible kidney conditions — to discuss with your doctor</div>',
                        unsafe_allow_html=True)
            _render_predictions(preds)

            reasons = notes.get("_reasons") or []
            if reasons:
                with st.expander("🔍 Why this result — what your score is based on", expanded=False):
                    for r in reasons:
                        st.markdown(f"- {r}")

            with st.expander("ℹ️ How is the Overall Risk calculated?"):
                st.markdown(
                    "- **Kidney function (main):** your eGFR (CKD-EPI 2021 formula from creatinine, age and sex) "
                    "combined with urine albumin using the international **KDIGO** risk grid — the same "
                    "method doctors use to stage CKD.\n"
                    "- **Risk factors:** diabetes, high BP, family history, painkiller use, abnormal blood values.\n"
                    "- **AI pattern model (20%):** an ensemble trained on public datasets (UCI, Bangladesh, NHANES) "
                    "gives a minor second opinion; it cannot turn healthy test results into a high score.\n"
                    "- **No blood test?** The score uses history and symptoms only, is capped, and is marked "
                    "**Limited confidence**.\n"
                    "- **Visual Screening** is shown separately and is **not** in the score.\n\n"
                    "This is a research screening estimate — not a medical diagnosis."
                )

            top_factors = _get_top_risk_factors()
            if top_factors:
                with st.expander("📊 What the AI pattern model looks at in general"):
                    st.caption(
                        "Global importance across all patients — not a personalised explanation of your result."
                    )
                    max_imp = top_factors[0][1] if top_factors else 1
                    for label, imp in top_factors:
                        pct = (imp / max_imp) if max_imp else 0
                        st.markdown(f"**{label}**")
                        st.progress(min(1.0, max(0.0, pct)))

            latest_egfr = get_latest_egfr(u["id"])
            if latest_egfr:
                with st.expander("🧮 Estimated eGFR (CKD-EPI 2021)"):
                    st.markdown(f"**{latest_egfr['egfr']} mL/min/1.73m²** — {latest_egfr['egfr_stage']}")
                    _bd = _iso_to_date(notes.get("_blood_date"))
                    if _bd:
                        st.caption(f"From your blood report dated {_bd.strftime('%d %b %Y')}.")
                    st.caption(
                        "Calculated using the standard CKD-EPI 2021 formula from your age, sex, "
                        "and serum creatinine. A single low reading does not by itself confirm CKD; "
                        "diagnosis needs persistence over 3+ months or other markers of kidney damage."
                    )

            if notes:
                label_map = {cq["key"]: cq["label"] for cq in CLINICAL_QUESTIONS}
                default_map = {cq["key"]: cq.get("default", "") for cq in CLINICAL_QUESTIONS}
                skip_values = ("", "no", "No change", "Not applicable", "Never / rarely",
                               "Normal (pale yellow)", "none", "never", "unknown", "normal",
                               "notpresent", "not diabetic", "not hypertensive", "no change")

                def _is_default(k, v):
                    d = default_map.get(k)
                    if d is None:
                        return False
                    try:
                        return float(v) == float(d)
                    except (TypeError, ValueError):
                        return str(v).strip().lower() == str(d).strip().lower()

                notable = {
                    k: v for k, v in notes.items()
                    if not k.startswith(("_", "hx_", "img_")) and k != "own_description" and v not in skip_values and not _is_default(k, v)
                }
                own_notes = (notes.get("own_description") or "").strip()
                hx_flags = notes.get("_history_flags") or []
                img_recs = notes.get("_imaging") or []
                if notable or own_notes or hx_flags or img_recs:
                    with st.expander("📝 Symptoms & Medical History — for your doctor's reference"):
                        st.caption("From your most recent Health Assessment — saved for a doctor visit.")
                        for k, v in notable.items():
                            st.markdown(f"**{label_map.get(k, k)}:** {v}")
                        if hx_flags:
                            st.markdown("**Points from your own notes:**")
                            for f in hx_flags:
                                st.markdown(f"- {f['label']}")
                        if img_recs:
                            st.markdown("**Scan reports:**")
                            for _r in img_recs:
                                _dt = _iso_to_date(_r.get("date"))
                                _fl = ", ".join(FINDING_LABELS.get(k, k) for k in _r.get("keys", [])) or "no findings ticked"
                                st.markdown(f"- {_r.get('modality', 'Scan')}"
                                            + (f" ({_dt.strftime('%d %b %Y')})" if _dt else "") + f": {_fl}")
                                if _r.get("impression"):
                                    st.caption("Kidney / urinary part of the report: " + _r["impression"][:300])
                        if own_notes:
                            st.markdown("**Your notes for the doctor (as you wrote them):**")
                            st.info(own_notes)

        with st.expander("🚻 CKD Symptoms — Men vs Women"):
            st.caption(
                "Symptoms doctors report seeing more often in one sex — not a diagnosis. "
                "Core symptoms (fatigue, swelling, urination changes, nausea, appetite loss) "
                "occur in both sexes. Please consult a doctor for any symptom."
            )
            user_gender = (u.get("gender") or "").strip().lower()
            men_tag = " (you)" if user_gender.startswith("m") else ""
            women_tag = " (you)" if user_gender.startswith("f") else ""
            st.markdown(
                f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px;">
                <div style="background:#EFF6FF;border:1.5px solid #BFDBFE;border-radius:12px;padding:16px;">
                    <div style="font-weight:700;color:#1D4ED8;margin-bottom:8px;">♂ Symptoms more often reported by men{men_tag}</div>
                    <ul style="margin:0;padding-left:18px;font-size:12.5px;color:#1E293B;line-height:1.9;">
                        <li>Difficulty starting or maintaining a urine stream</li>
                        <li>Weak or interrupted urine flow</li>
                        <li>Erectile dysfunction</li>
                        <li>Blood visible in urine, noticed more often</li>
                    </ul>
                </div>
                <div style="background:#FDF2F8;border:1.5px solid #FBCFE8;border-radius:12px;padding:16px;">
                    <div style="font-weight:700;color:#BE185D;margin-bottom:8px;">♀ Symptoms more often reported by women{women_tag}</div>
                    <ul style="margin:0;padding-left:18px;font-size:12.5px;color:#1E293B;line-height:1.9;">
                        <li>Burning or urgent urination linked to recurrent UTIs</li>
                        <li>Swelling in the face/around the eyes noticed with menstrual or pregnancy-related fluid changes</li>
                        <li>Swelling or high blood pressure appearing during pregnancy (pre-eclampsia pattern)</li>
                        <li>Unexplained change in menstrual cycle regularity</li>
                    </ul>
                </div>
            </div>
            <div style="margin-top:10px;font-size:11.5px;color:#64748B;line-height:1.6;">
                Source: National Kidney Foundation; <i>Nature Reviews Nephrology</i> (Carrero et al., Ricardo et al.).
                This is general population data, not a personal diagnosis.
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class="ns-mods" style="grid-template-columns:repeat(2,1fr);">
            <div class="ns-mod op">
                <div class="ns-mp av">● Available</div>
                <div class="ns-mi">📋</div>
                <div class="ns-mt">Health Assessment</div>
                <div class="ns-md">Answer short health questions and add your lab values if you have them. Get your kidney risk and possible conditions.</div>
            </div>
            <div class="ns-mod op">
                <div class="ns-mp av">● Available</div>
                <div class="ns-mi">👁️</div>
                <div class="ns-mt">Visual Screening</div>
                <div class="ns-md">Upload an eyelid photo for a model-based check — a supporting signal, shown for reference only.</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📋 Start Health Assessment", use_container_width=True):
                go("lifestyle")
        with b2:
            if st.button("👁️ Start Visual Screening", use_container_width=True):
                go("eye")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        b3, b4 = st.columns(2)
        with b3:
            if st.button("💬 Ask NephroBot →", use_container_width=True, key="dash_bot_btn"):
                go("chatbot")
        with b4:
            if st.button("🏥  Find Nephrologist / Hospital →", use_container_width=True):
                go("find_hospital")



_URINE_KEYS = ("sg", "al", "su")
_BLOOD_KEYS = ("bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc")
_REPORT_ALIAS = {"wc": "wbcc", "rc": "rbcc", "rbc": "rbc_urine", "pc": "pc_urine",
                 "pcc": "pcc_urine", "ba": "ba_urine"}


def _apply_extracted_values(found):
    found = {_REPORT_ALIAS.get(k, k): v for k, v in found.items()}
    filled_keys = []
    for q in list(QUESTIONS) + list(CLINICAL_QUESTIONS):
        key = q["key"]
        if key not in found or q.get("type") not in ("number", "select"):
            continue
        raw = found[key]
        widget_key = f"q_{key}"
        try:
            if q["type"] == "number":
                val = max(float(q["min"]), min(float(q["max"]), float(raw)))
                st.session_state[widget_key] = val
                filled_keys.append(key)
            else:
                options = q["options"]
                if key in ("pcc_urine", "ba_urine"):
                    mapped = "present" if raw in (1, "present") else "notpresent"
                    if mapped in options:
                        st.session_state[widget_key] = mapped
                        filled_keys.append(key)
                elif isinstance(raw, str):
                    if raw in options:
                        st.session_state[widget_key] = raw
                        filled_keys.append(key)
                elif all(isinstance(o, (int, float)) for o in options):
                    st.session_state[widget_key] = min(options, key=lambda o: abs(o - float(raw)))
                    filled_keys.append(key)
        except (ValueError, TypeError, KeyError):
            continue
    if any(k in filled_keys for k in _URINE_KEYS):
        st.session_state["have_urine"] = True
    if any(k in filled_keys for k in _BLOOD_KEYS):
        st.session_state["have_blood"] = True
    return filled_keys


_FIELD_NAMES = {"hemo": "Haemoglobin", "pcv": "PCV", "wbcc": "WBC count", "rbcc": "RBC count", "sc": "Creatinine",
                "bu": "Urea", "sod": "Sodium", "pot": "Potassium", "bgr": "Blood sugar", "sg": "Specific gravity",
                "al": "Urine albumin", "su": "Urine sugar", "pc_urine": "Pus cells", "rbc_urine": "Urine RBC",
                "pcc_urine": "Pus cell clumps", "ba_urine": "Bacteria"}
_IMG_MODALITIES = ["Ultrasound (sonography)", "MRI", "CT scan", "X-ray", "Other"]


_READ_ROWS = [
    ("Blood report", [("hemo", "Haemoglobin", "g/dL"), ("pcv", "PCV / haematocrit", "%"),
                      ("wbcc", "WBC count", "per cumm"), ("rbcc", "RBC count", "million/cumm"),
                      ("sc", "Creatinine", "mg/dL"), ("bu", "Blood urea", "mg/dL"),
                      ("sod", "Sodium", "mmol/L"), ("pot", "Potassium", "mmol/L"),
                      ("bgr", "Random blood sugar", "mg/dL")]),
    ("Urine report", [("sg", "Specific gravity", ""), ("al", "Albumin / protein", ""), ("su", "Sugar / glucose", ""),
                      ("pc_urine", "Pus cells", ""), ("rbc_urine", "Red blood cells", ""),
                      ("ba_urine", "Bacteria", "")]),
]


def _fmt_read(key, v):
    try:
        if key in ("al", "su"):
            return _GRADE_LABELS.get(int(v), str(v))
        if key in ("pc_urine", "rbc_urine"):
            return "Normal" if v == "normal" else "Raised"
        if key in ("ba_urine", "pcc_urine"):
            return "Present" if v == "present" else "Absent"
        if key == "wbcc":
            return f"{float(v):,.0f}"
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _render_read_table(found):
    """Neat table of what was read from the uploaded report(s), plus what was not found."""
    if not found:
        return
    td = 'style="padding:6px 10px;border:1px solid #E2E8F0;"'
    missing, blocks = [], []
    for section, rows in _READ_ROWS:
        body = ""
        for key, label, unit in rows:
            if key in found:
                body += (f"<tr><td {td}>{label}</td><td {td}><b>{html.escape(_fmt_read(key, found[key]))}</b></td>"
                         f"<td {td}>{unit}</td></tr>")
            else:
                missing.append(label)
        if body:
            blocks.append(
                f'<div style="font-size:12px;font-weight:700;color:#1D4ED8;margin:10px 0 4px;">{section}</div>'
                f'<table style="width:100%;border-collapse:collapse;font-size:12.5px;">'
                f'<tr style="background:#F1F5F9;text-align:left;"><th {td}>Test</th><th {td}>Value read</th>'
                f'<th {td}>Unit</th></tr>{body}</table>')
    st.markdown("".join(blocks), unsafe_allow_html=True)
    if missing:
        st.caption("Not found in your report: " + ", ".join(missing) +
                   ". These stay at typical normal values in the form below and are not your real results.")


def _consistency_warning(found):
    try:
        if "hemo" in found and "pcv" in found and abs(float(found["pcv"]) - 3 * float(found["hemo"])) > 7:
            return "Haemoglobin and PCV do not match well, so one of them may be misread — please check both."
    except (TypeError, ValueError):
        pass
    return ""


def _imaging_block():
    """Scan reports (ultrasound / MRI / CT): upload printed report or add by hand. Returns the confirmed list."""
    u = st.session_state.user
    if st.session_state.get("_img_user") != u["id"]:
        prior = _parse_notes(get_latest_clinical_notes(u["id"])).get("_imaging") or []
        st.session_state["imaging_reports"] = [dict(r, id=i) for i, r in enumerate(prior)]
        st.session_state["_img_next_id"] = len(prior)
        st.session_state["_img_user"] = u["id"]
    recs = st.session_state["imaging_reports"]

    with st.expander("🩻 Ultrasound / MRI / CT scan reports — upload the printed report or add it by hand",
                     expanded=bool(recs)):
        _f = st.session_state.pop("_img_flash", None)
        if _f:
            getattr(st, _f[0])(_f[1])
        st.caption("Upload all pages of ONE scan report. Add other scans one by one. Scan findings are counted in the "
                   "risk result and shown to your doctor.")
        n = st.session_state.get("_img_upload_n", 0)
        files = st.file_uploader("Upload scan report", type=["jpg", "jpeg", "png", "pdf"],
                                 accept_multiple_files=True, key=f"imaging_upload_{n}",
                                 label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if files and st.button("Read scan report →", use_container_width=True, key="read_img_btn"):
                with st.spinner("Reading your scan report…"):
                    text, err = read_text_from_files([(f.getvalue(), f.name) for f in files])
                if err or not text:
                    st.error(err or "Couldn't read this scan report. Please add it by hand.")
                else:
                    info = extract_imaging(text)
                    keys = [x["key"] for x in info["findings"]] or (["normal_study"] if info["normal"] else [])
                    d = _guess_report_date(text)
                    rid = st.session_state.get("_img_next_id", len(recs))
                    st.session_state["_img_next_id"] = rid + 1
                    recs.append({"id": rid, "modality": info["modality"], "date": d.isoformat() if d else None,
                                 "keys": keys, "impression": info["impression"], "source": files[0].name})
                    st.session_state["_img_upload_n"] = n + 1
                    st.session_state["_img_flash"] = (
                        "success", "Scan report read. Please check the findings below against your report.")
                    st.rerun()
        with c2:
            if st.button("➕ Add a scan by hand", use_container_width=True, key="add_img_btn"):
                rid = st.session_state.get("_img_next_id", len(recs))
                st.session_state["_img_next_id"] = rid + 1
                recs.append({"id": rid, "modality": "Ultrasound (sonography)", "date": None, "keys": [],
                             "impression": "", "source": ""})
                st.rerun()

        final = []
        for rec in list(recs):
            rid = rec["id"]
            k_mod, k_date, k_keys = f"img_mod_{rid}", f"img_date_{rid}", f"img_keys_{rid}"
            if k_mod not in st.session_state:
                st.session_state[k_mod] = rec["modality"] if rec.get("modality") in _IMG_MODALITIES else "Other"
            if k_keys not in st.session_state:
                st.session_state[k_keys] = [k for k in rec.get("keys", []) if k in FINDING_LABELS]
            d0 = _iso_to_date(rec.get("date"))
            if k_date not in st.session_state and d0:
                st.session_state[k_date] = d0
            st.markdown("---")
            if rec.get("source"):
                st.caption(f"From file: {rec['source']}")
            cc1, cc2 = st.columns(2)
            with cc1:
                mod = st.selectbox("Type of scan", _IMG_MODALITIES, key=k_mod)
            with cc2:
                dd = _date_input("📅 Date of this scan", k_date, "Use the date printed on the scan report.")
            sel = st.multiselect("Findings in this report — please tick everything the report mentions",
                                 options=list(FINDING_LABELS.keys()), format_func=lambda k: FINDING_LABELS[k],
                                 key=k_keys)
            if rec.get("impression"):
                st.caption("What the report says about your kidneys / urinary tract: " + rec["impression"][:500])
            if st.button("🗑️ Remove this scan", key=f"img_rm_{rid}"):
                recs.remove(rec)
                for k in (k_mod, k_date, k_keys):
                    st.session_state.pop(k, None)
                st.rerun()
            rec.update(modality=mod, date=dd.isoformat() if dd else None, keys=list(sel))
            final.append(rec)
        return final


def page_lifestyle():
    u = st.session_state.user

    prior_raw = get_latest_lifestyle_answers(u["id"])
    if prior_raw:
        prior_answers = _parse_notes(prior_raw)
        for q in QUESTIONS:
            wk = f"q_{q['key']}"
            if wk not in st.session_state and q["key"] in prior_answers and prior_answers[q["key"]] is not None:
                st.session_state[wk] = prior_answers[q["key"]]

    prior_clinical = _parse_notes(get_latest_clinical_notes(u["id"]))
    if prior_clinical:
        for cq in CLINICAL_QUESTIONS:
            wk = f"q_{cq['key']}"
            if wk not in st.session_state and prior_clinical.get(cq["key"]):
                st.session_state[wk] = prior_clinical[cq["key"]]
        if "have_blood" not in st.session_state:
            st.session_state["have_blood"] = bool(prior_clinical.get("_have_blood"))
        if "have_urine" not in st.session_state:
            st.session_state["have_urine"] = bool(prior_clinical.get("_have_urine"))
        for _k, _nk in (("blood_date", "_blood_date"), ("urine_date", "_urine_date")):
            _d = _iso_to_date(prior_clinical.get(_nk))
            if _k not in st.session_state and _d:
                st.session_state[_k] = _d

    with st.container(key="lifestyle_wrap"):
        if prior_raw:
            st.info("📝 Pre-filled from your last submission — edit anything that's changed, or just review and resubmit.")
        st.markdown(
            """
        <div class="ns-hf-top">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1D4ED8,#3B82F6);
                        border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">📋</div>
            <div>
                <div class="ns-hft">Health Assessment</div>
                <div class="ns-hfs">Fill what you know. For the most accurate result, add your blood and urine report values and tick the “I have a report” boxes.</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        _ocr_open = bool(st.session_state.get("_ocr_flash") or st.session_state.get("_last_ocr_text"))
        with st.expander("📤 Upload your blood / urine report (photo or PDF) to fill the values below", expanded=_ocr_open):
            _flash = st.session_state.pop("_ocr_flash", None)
            if _flash:
                getattr(st, _flash[0])(_flash[1])
            st.caption("Upload every page of your printed report (photos or PDF). Please check the filled values before submitting.")
            if uses_online_ai():
                st.caption("🔒 Reports are read by an online AI service (Google Gemini).")
            report_files = st.file_uploader(
                "Upload report", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True,
                key="report_upload", label_visibility="collapsed",
            )
            if report_files:
                if st.button("Read report & fill values →", use_container_width=True, key="extract_btn"):
                    _setup = "" if reader_ready() else " — one-time reader setup, this can take 1–3 minutes"
                    try:
                        _bar = st.progress(0.0, text=f"Reading your report{_setup}…")
                    except TypeError:
                        _bar = st.progress(0)

                    def _prog(done, total):
                        try:
                            _bar.progress(done / total, text=f"Read page {done} of {total}…")
                        except Exception:
                            pass

                    found, raw_text, err = extract_values_from_files(
                        [(f.getvalue(), f.name) for f in report_files], progress=_prog)
                    _bar.empty()
                    _st = last_stats()
                    if _st.get("pages"):
                        st.session_state["_read_timing"] = (
                            f"Read {_st['pages']} page(s) in {_st['seconds']:.0f} seconds"
                            + (f" using {_st['engine']}." if _st.get("engine") else "."))
                    st.session_state["_last_ocr_text"] = raw_text or ""
                    st.session_state["_last_found"] = found or {}
                    if err:
                        st.error(err)
                    elif not found:
                        st.warning("Couldn't read any values clearly from this report. Please type them in below.")
                    else:
                        filled = _apply_extracted_values(found)
                        guessed = _guess_report_date(raw_text)
                        if guessed:
                            st.session_state["blood_date"] = guessed
                            st.session_state["urine_date"] = guessed
                            st.session_state["_date_guess_note"] = guessed.strftime("%d/%m/%Y")
                        if filled:
                            warn = _consistency_warning(found)
                            msg = (f"Filled {len(filled)} value(s): "
                                   + ", ".join(_FIELD_NAMES.get(k, k) for k in filled)
                                   + ". Please check them below before submitting.")
                            st.session_state["_ocr_flash"] = ("warning" if warn else "success", msg + (" " + warn if warn else ""))
                            st.rerun()
                        else:
                            st.warning("Couldn't match the text to the fields. Please type the values in below.")
            if st.session_state.get("_last_found"):
                st.markdown("**What was read from your report**")
                if st.session_state.get("_read_timing"):
                    st.caption(st.session_state["_read_timing"])
                _render_read_table(st.session_state["_last_found"])
            if st.session_state.get("_last_ocr_text"):
                if st.checkbox("Show the full text that was read", key="show_ocr_text"):
                    st.text(st.session_state["_last_ocr_text"][:6000])

        imaging_final = _imaging_block()

        if st.session_state.get("_date_guess_note"):
            st.info(f"📅 Report date read from your upload: **{st.session_state['_date_guess_note']}**. "
                    "Please check the date fields below and correct them if needed.")

        hx_confirmed = []
        with st.expander(
            "📝 Anything else your doctor should know? (family history, past illnesses, operations, recent fever…)",
            expanded=bool(st.session_state.get("q_own_description")),
        ):
            st.caption(
                "Type in Marathi, Hindi, or English — family illnesses, past illnesses, operations, recent fever. "
                "Everything is saved for your doctor."
            )
            import streamlit.components.v1 as _cv1_mic
            # Input bar styled like NephroBot; the native text area below is hidden and kept in sync with it.
            st.markdown(
                "<style>div.st-key-q_own_description{position:absolute!important;left:-9999px!important;"
                "height:0!important;overflow:hidden!important;opacity:0!important;pointer-events:none!important;}</style>",
                unsafe_allow_html=True,
            )
            _NOTES_BAR_HTML = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{background:transparent;font-family:'Inter',system-ui,sans-serif}
.wrap{display:flex;flex-direction:column;gap:8px}
.langs{display:flex;gap:6px;align-items:center}
.lb{font-size:11.5px;color:#64748B;margin-right:2px}
.pill{border:1.5px solid #E2E8F0;background:#fff;color:#475569;border-radius:20px;padding:4px 13px;font-size:12px;cursor:pointer;transition:all .15s;font-family:inherit}
.pill:hover{border-color:#93C5FD;color:#1D4ED8}
.pill.on{background:#1D4ED8;border-color:#1D4ED8;color:#fff}
.bar{display:flex;align-items:flex-end;gap:0;background:#1C1C1E;border-radius:22px;padding:14px 12px 14px 20px;box-shadow:0 2px 18px rgba(0,0,0,.22);transition:box-shadow .2s}
.bar:focus-within{box-shadow:0 2px 18px rgba(0,0,0,.28),0 0 0 2px rgba(59,130,246,.45)}
.bar textarea{flex:1;min-width:0;background:transparent;border:none;outline:none;resize:none;font-size:15px;color:#F1F5F9;line-height:1.55;min-height:72px;max-height:120px;font-family:'Inter',system-ui,sans-serif;overflow-y:auto;caret-color:#F1F5F9}
.bar textarea::placeholder{color:#52525B}
.btns{display:flex;align-items:center;flex-shrink:0;padding-bottom:2px}
#mic{width:38px;height:38px;border-radius:50%;border:none;background:transparent;color:#71717A;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:color .15s,transform .12s}
#mic:hover{color:#E4E4E7;transform:scale(1.1)}
#mic.active{color:#EF4444;animation:pulse 1s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.18)}}
#status{font-size:11.5px;color:#64748B;padding-left:6px;min-height:15px}
#status.active{color:#EF4444;font-weight:600}
#status.ok{color:#22C55E;font-weight:600}
#status.err{color:#F87171}
</style></head><body>
<div class="wrap">
<div class="langs"><span class="lb">Speak in:</span>
<button class="pill" data-l="en-IN">English</button>
<button class="pill" data-l="hi-IN">हिंदी</button>
<button class="pill" data-l="mr-IN">मराठी</button></div>
<div class="bar">
<textarea id="txt" rows="3" placeholder="Write here… / लिहा… / लिखें…"></textarea>
<div class="btns"><button id="mic" title="Speak in English, Hindi or Marathi">
<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-6.5 10.5a.75.75 0 0 1 .75.75 5.75 5.75 0 0 0 11.5 0 .75.75 0 0 1 1.5 0 7.25 7.25 0 0 1-6.5 7.21V21h2a.75.75 0 0 1 0 1.5h-5.5a.75.75 0 0 1 0-1.5h2v-1.54a7.25 7.25 0 0 1-6.5-7.21.75.75 0 0 1 .75-.75z"/></svg></button></div>
</div>
<div id="status"></div>
</div>
<script>(function(){
var txt=document.getElementById('txt'),mic=document.getElementById('mic'),st=document.getElementById('status');
var pills=document.querySelectorAll('.pill');
var NAMES={'en-IN':'English','hi-IN':'हिंदी','mr-IN':'मराठी'};
var cur='mr-IN';try{cur=sessionStorage.getItem('ns_doc_lang')||'mr-IN';}catch(e){}
function paint(){for(var i=0;i<pills.length;i++){pills[i].classList.toggle('on',pills[i].getAttribute('data-l')===cur);}}
paint();
function findTA(){try{var d=window.parent.document;return d.querySelector('.st-key-q_own_description textarea')||d.querySelector('textarea[aria-label="Your notes"]');}catch(e){return null;}}
function resize(){txt.style.height='auto';txt.style.height=Math.min(Math.max(txt.scrollHeight,72),120)+'px';}
function push(){var ta=findTA();if(!ta||ta.value===txt.value)return;
 try{var P=window.parent;var set=Object.getOwnPropertyDescriptor(P.HTMLTextAreaElement.prototype,'value').set;
 set.call(ta,txt.value);ta.dispatchEvent(new P.Event('input',{bubbles:true}));
 ta.dispatchEvent(new P.FocusEvent('focusout',{bubbles:true}));}catch(e){}}
var busy=false,rec=null,base='';
function pull(){var ta=findTA();if(!ta||busy||document.activeElement===txt)return;
 if(ta.value!==txt.value){txt.value=ta.value;resize();}}
setTimeout(pull,200);setTimeout(pull,900);setInterval(pull,1500);
txt.addEventListener('input',resize);
txt.addEventListener('blur',push);
window.addEventListener('pagehide',push);
var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
for(var i=0;i<pills.length;i++){pills[i].addEventListener('click',function(){
 cur=this.getAttribute('data-l');try{sessionStorage.setItem('ns_doc_lang',cur);}catch(e){}
 paint();if(busy&&rec){rec.stop();}else{st.className='';st.textContent='';}});}
if(!SR){mic.style.display='none';st.className='err';st.textContent='Voice input needs Chrome or Edge.';}
else{mic.addEventListener('click',function(){
 if(busy){rec&&rec.stop();return;}
 base=txt.value.replace(/\s+$/,'');
 rec=new SR();rec.lang=cur;rec.continuous=true;rec.interimResults=true;
 rec.onstart=function(){busy=true;mic.classList.add('active');st.className='active';st.textContent='🔴 Listening… ('+NAMES[cur]+') — tap the mic to stop';};
 rec.onresult=function(e){var f='',it='';
  for(var i=0;i<e.results.length;i++){if(e.results[i].isFinal){f+=e.results[i][0].transcript;}else{it+=e.results[i][0].transcript;}}
  var all=(f+it).trim();txt.value=base+(base&&all?' ':'')+all;resize();txt.scrollTop=txt.scrollHeight;};
 rec.onerror=function(e){busy=false;mic.classList.remove('active');
  var m={'no-speech':'Nothing heard — tap the mic again.','not-allowed':'Microphone is blocked in the browser.','audio-capture':'No microphone found.','language-not-supported':NAMES[cur]+' is not supported by this browser.'};
  st.className='err';st.textContent=m[e.error]||('Error: '+e.error);};
 rec.onend=function(){busy=false;mic.classList.remove('active');txt.value=txt.value.trim();resize();push();
  if(st.className!=='err'){st.className='ok';st.textContent='✅ Saved';}};
 try{rec.start();}catch(err){st.className='err';st.textContent='Cannot start: '+err.message;}});}
})();</script></body></html>'''
            _cv1_mic.html(_NOTES_BAR_HTML, height=190, scrolling=False)

            own_text = st.text_area(
                "Your notes",
                key="q_own_description",
                height=130,
                label_visibility="collapsed",
                placeholder="Write here…",
            )
            found_items = extract_history(own_text)
            if found_items:
                st.markdown("**We picked up these points — untick anything that is wrong:**")
                for it in found_items:
                    ticked = st.checkbox(f"{it['label']}  — “{it['snippet']}”", value=True, key=f"hx_{it['key']}")
                    if ticked:
                        hx_confirmed.append(it)
                st.caption(
                    "Confirmed points are used in the clinical risk-factor checklist and the “possible conditions” "
                    "list, and are shown to your doctor. Your own words are always saved exactly as typed."
                )

        with st.expander("🧫 Did a home urine dipstick test? Use it to fill in the Urine Test Results below"):
            st.caption(
                "Dipstick self-test kits (available at most pharmacies) are a real, low-cost way "
                "to get FRESH urine readings when your last lab report is old or you don't have one. "
                "This app does not automatically read the strip colour — compare your strip to your kit's "
                "colour chart yourself, tick **I have a urine test report / dipstick result** below, then enter the closest matching values in the **Urine Test Results** "
                "section below (specific gravity, albumin, sugar) — plus the urine pH field further down."
            )
            dipstick_photo = st.file_uploader(
                "Optional: upload a photo of your strip to keep next to the chart while you compare (not analyzed automatically)",
                type=["jpg", "jpeg", "png"], key="dipstick_photo_ref",
            )
            if dipstick_photo is not None:
                st.image(dipstick_photo, caption="Your uploaded strip photo (for your own reference only)", width=280)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**How to read a dipstick — general guide**")
            st.caption(
                "Colours are a rough guide only — always match against the printed chart that came with YOUR kit."
            )
            cell = 'style="padding:6px 8px;border:1px solid #E2E8F0;"'
            st.markdown(
                f"""
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <tr style="background:#F1F5F9;text-align:left;">
                    <th {cell}>Pad</th><th {cell}>What it checks</th>
                    <th {cell}>Darker/more colour usually means</th><th {cell}>Fill in above as</th>
                </tr>
                <tr><td {cell}>Protein</td><td {cell}>Albumin/protein leakage</td><td {cell}>Higher albumin level</td><td {cell}>Albumin level</td></tr>
                <tr><td {cell}>Glucose</td><td {cell}>Sugar in urine</td><td {cell}>Higher sugar level</td><td {cell}>Sugar level</td></tr>
                <tr><td {cell}>Leukocytes</td><td {cell}>White blood cells (infection)</td><td {cell}>Present</td><td {cell}>Pus cells = abnormal</td></tr>
                <tr><td {cell}>Nitrite</td><td {cell}>Bacterial infection</td><td {cell}>Positive (pink)</td><td {cell}>Bacteria = present</td></tr>
                <tr><td {cell}>Specific Gravity</td><td {cell}>Urine concentration</td><td {cell}>Read the number directly (e.g. 1.010–1.025)</td><td {cell}>Specific gravity</td></tr>
                <tr><td {cell}>pH</td><td {cell}>Acidity/alkalinity</td><td {cell}>Read the number directly (usually 4.5–8)</td><td {cell}>Urine pH field below</td></tr>
            </table>
            """,
                unsafe_allow_html=True,
            )

        SECTION_AT = {
            "age": "🧍 Personal & Vitals",
            "sg": "🧪 Urine Test Results",
            "bgr": "🩸 Blood Test Results",
            "htn": "💭 Symptoms & Lifestyle",
        }

        answers = {}
        blood_date = None
        urine_date = None

        # Ticking these (outside the form, so the page updates at once) decides which questions appear:
        # questions about test results are asked ONLY if the person actually has that test.
        st.markdown('<div class="ns-qsection">📄 Do you have any test results?</div>', unsafe_allow_html=True)
        st.caption("Tick only what you really have. Questions about test results are shown only for the tests you tick.")
        have_urine = st.checkbox(
            "I have a urine test report / dipstick result",
            key="have_urine",
        )
        if have_urine:
            urine_date = _date_input(
                "📅 Date of your urine test", "urine_date",
                "Use the sample collection date printed on the report. Old results count for less.")
        have_blood = st.checkbox(
            "I have a blood test report (needed for an accurate result)",
            key="have_blood",
        )
        if have_blood:
            blood_date = _date_input(
                "📅 Date of your blood test", "blood_date",
                "Use the sample collection date printed on the report. Old results count for less.")

        # which scored questions to show: lab questions only when that test is ticked
        _vis_q = [q for q in QUESTIONS
                  if not q.get("lab")
                  or (q["key"] in _URINE_KEYS and have_urine)
                  or (q["key"] in _BLOOD_KEYS and have_blood)]
        # symptom-section questions that are really test results (urine report or dipstick)
        _URINE_TEST_CQ = {"rbc_urine", "pc_urine", "pcc_urine", "ba_urine", "dipstick_ph"}

        def render_question(q):
            label = f"{q['label']} ({q['unit']})" if q.get("unit") else q["label"]
            if q.get("lab"):
                label = f"📄 {label}"
            help_text = REPORT_HINTS.get(q["key"]) if q.get("lab") else None

            widget_key = f"q_{q['key']}"
            already_set = widget_key in st.session_state

            if q["type"] == "number":
                kwargs = dict(
                    min_value=float(q["min"]), max_value=float(q["max"]),
                    step=float(q.get("step", 1)), key=widget_key, help=help_text,
                )
                if not already_set:
                    kwargs["value"] = float(q["default"])
                answers[q["key"]] = st.number_input(label, **kwargs)
            elif q["type"] == "select":
                kwargs = dict(key=widget_key, help=help_text)
                if q["key"] in ("al", "su"):
                    kwargs["format_func"] = lambda v: _GRADE_LABELS.get(v, str(v))
                if not already_set:
                    kwargs["index"] = q["options"].index(q["default"])
                answers[q["key"]] = st.selectbox(label, q["options"], **kwargs)
            elif q["type"] == "yesno":
                choice = st.radio(
                    label, ["no", "yes"],
                    index=0 if q["default"] == "no" else 1,
                    horizontal=True,
                    key=widget_key,
                )
                answers[q["key"]] = choice
            elif q["type"] == "text":
                kwargs = dict(key=widget_key, placeholder=q.get("placeholder", ""))
                if not already_set:
                    kwargs["value"] = ""
                answers[q["key"]] = st.text_input(label, **kwargs)

        with st.form("lifestyle_form"):
            i = 0
            n = len(_vis_q)
            while i < n:
                q = _vis_q[i]
                if q["key"] in SECTION_AT:
                    st.markdown(
                        f'<div class="ns-qsection">{SECTION_AT[q["key"]]}</div>',
                        unsafe_allow_html=True,
                    )

                next_starts_section = (i + 1 < n) and (_vis_q[i + 1]["key"] in SECTION_AT)
                if i + 1 < n and not next_starts_section:
                    col1, col2 = st.columns(2, gap="medium")
                    with col1:
                        render_question(q)
                    with col2:
                        render_question(_vis_q[i + 1])
                    i += 2
                else:
                    render_question(q)
                    i += 1

            st.markdown(
                '<div class="ns-qsection">🩺 Symptoms & Medical History (helps your doctor and the “possible conditions” list)</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "These answers feed the transparent Clinical Risk Factor Checklist and the possible-conditions "
                "panel. Some questions only appear when relevant to your age/gender, or when you have the related test."
            )
            visible_cq = [cq for cq in visible_clinical_questions(u.get("gender"), answers.get("age", u.get("age", 45)))
                          if cq["key"] not in _URINE_TEST_CQ or have_urine]
            j = 0
            m = len(visible_cq)
            while j < m:
                cq = visible_cq[j]
                if j + 1 < m:
                    col1, col2 = st.columns(2, gap="medium")
                    with col1:
                        render_question(cq)
                    with col2:
                        render_question(visible_cq[j + 1])
                    j += 2
                else:
                    render_question(cq)
                    j += 1

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Submit Health Assessment →", use_container_width=True)

        for _q in QUESTIONS:
            answers.setdefault(_q["key"], _q["default"])

        if submitted:
            _errs = []
            _today = date.today()
            for _on, _d, _nm in ((have_blood, blood_date, "blood"), (have_urine, urine_date, "urine")):
                if _on and _d is None:
                    _errs.append(f"Please enter the date of your {_nm} test (printed on the report). "
                                 "Old reports count for less, so the date is needed for an accurate result.")
                elif _on and _d > _today:
                    _errs.append(f"The {_nm} test date can't be in the future.")
            for _r in imaging_final:
                _real = [k for k in _r["keys"] if k != "normal_study"]
                _dd = _iso_to_date(_r.get("date"))
                if _real and _dd is None:
                    _errs.append(f"Please enter the date of your {_r['modality']} scan (printed on the report).")
                elif _dd and _dd > _today:
                    _errs.append(f"The {_r['modality']} scan date can't be in the future.")
            if _errs:
                for _e in _errs:
                    st.error(_e)
                st.stop()
            blood_days = (_today - blood_date).days if have_blood else None
            urine_days = (_today - urine_date).days if have_urine else None
            st.session_state.pop("_date_guess_note", None)
            if answers.get("bp", 0) >= 120:
                st.warning(
                    "Your blood pressure entry is 120 or higher. This field needs the LOWER (diastolic) number "
                    "— e.g. 80 for a reading of 120/80. Please correct it if you entered the upper number."
                )
            with st.spinner("Calculating your risk..."):
                ml_result = predict_lifestyle_risk(answers, run_lime=False) if have_blood else None
                clinical_answers = {cq["key"]: answers.get(cq["key"], "") for cq in visible_cq}
                clinical_answers["own_description"] = st.session_state.get("q_own_description", "")
                for it in hx_confirmed:
                    clinical_answers[f"hx_{it['key']}"] = "yes"
                hx_keys = {it["key"] for it in hx_confirmed}
                if "fam_kidney_disease" in hx_keys:
                    clinical_answers["family_ckd"] = "yes"
                if "self_kidney_stones_recurrent" in hx_keys:
                    clinical_answers["recurrent_kidney_stones"] = "yes"
                img_notes, imaging_days = [], None
                for _r in imaging_final:
                    _dd = _iso_to_date(_r.get("date"))
                    _days = (_today - _dd).days if _dd else None
                    if _days is not None:
                        imaging_days = _days if imaging_days is None else max(imaging_days, _days)
                    for _k in _r["keys"]:
                        if _k == "normal_study":
                            clinical_answers["img_normal_study"] = "yes"
                        elif _k in MAJOR_KEYS or _days is None or _days <= 365:
                            clinical_answers[f"img_{_k}"] = "yes"
                        else:
                            img_notes.append(
                                f"Your {_r['modality']} report is {age_text(_days)} old and mentions "
                                f"{FINDING_LABELS.get(_k, _k).lower()}. Because it is over a year old it was not counted — "
                                f"please repeat the scan if you still have symptoms.")
                lifestyle_answers_only = {q["key"]: answers.get(q["key"]) for q in QUESTIONS}

                egfr_result = None
                if have_blood:
                    egfr_result = calculate_egfr_ckdepi(
                        age=answers.get("age"), sex=u.get("gender"), scr_mg_dl=answers.get("sc")
                    )
                assessment = assess_risk(answers, clinical_answers, u, ml_result, have_blood, have_urine,
                                         blood_days=blood_days, urine_days=urine_days,
                                         imaging_days=imaging_days)
                assessment["report_notes"] = list(assessment["report_notes"]) + img_notes
                if not assessment["blood_used"]:
                    egfr_result = None
                preds = predict_diseases(
                    answers, clinical_answers, u,
                    egfr_result["egfr"] if egfr_result else None,
                    assessment["blood_used"], assessment["urine_used"],
                )
                clinical_risk = calculate_clinical_risk_score(clinical_answers)

            notes_to_save = dict(clinical_answers)
            notes_to_save.update({
                "_predictions": preds,
                "_reasons": assessment["reasons"],
                "_confidence": assessment["confidence"],
                "_have_blood": bool(have_blood),
                "_have_urine": bool(have_urine),
                "_blood_date": blood_date.isoformat() if have_blood and blood_date else None,
                "_urine_date": urine_date.isoformat() if have_urine and urine_date else None,
                "_report_notes": assessment["report_notes"],
                "_imaging": [{"modality": r["modality"], "date": r.get("date"), "keys": r["keys"],
                              "impression": (r.get("impression") or "")[:400], "source": r.get("source", "")}
                             for r in imaging_final],
                "_history_flags": [{"key": it["key"], "label": it["label"], "kind": it["kind"]} for it in hx_confirmed],
            })
            save_assessment(
                user_id=u["id"],
                risk_score=assessment["risk_score"],
                ckd_stage=assessment["ckd_stage"],
                risk_level=assessment["risk_level"],
                lifestyle_score=assessment["risk_score"],
                clinical_notes=json.dumps(notes_to_save),
                egfr=egfr_result["egfr"] if egfr_result else None,
                egfr_stage=egfr_result["stage_label"] if egfr_result else None,
                lifestyle_answers=json.dumps(lifestyle_answers_only),
            )

            st.session_state["_show_result"] = True
            st.session_state["_last_answers"] = answers
            st.session_state["_assessment"] = assessment
            st.session_state["_assessment_ml"] = ml_result
            st.session_state["_assessment_preds"] = preds
            st.session_state["_assessment_egfr"] = egfr_result
            st.session_state["_assessment_clinical_risk"] = clinical_risk
            st.session_state["_assessment_have_blood"] = bool(have_blood)

        if st.session_state.get("_show_result") and st.session_state.get("_assessment"):
            assessment = st.session_state["_assessment"]
            ml_result = st.session_state.get("_assessment_ml")
            preds = st.session_state.get("_assessment_preds") or []
            egfr_result = st.session_state.get("_assessment_egfr")
            clinical_risk = st.session_state.get("_assessment_clinical_risk")
            answers_for_lime = st.session_state.get("_last_answers", {})
            was_blood = st.session_state.get("_assessment_have_blood", False)

            rc = {"HIGH": "re", "MEDIUM": "am", "LOW": "gr"}.get(assessment["risk_level"], "bl")

            extra = ""
            if egfr_result:
                extra += (
                    f"<br><b>Estimated eGFR (CKD-EPI 2021): {egfr_result['egfr']} mL/min/1.73m² "
                    f"— {egfr_result['stage_label']}</b>"
                )
            if assessment.get("kdigo_category"):
                extra += f"<br>KDIGO category: {html.escape(assessment['kdigo_category'])}"
            conf_note = {
                "High": "High confidence (recent blood + urine results used)",
                "Moderate": "Moderate confidence (recent blood test used; urine test missing or old)",
                "Limited": "Limited confidence (no recent blood test — history and symptoms mainly)",
            }[assessment["confidence"]]
            if assessment.get("blood_days") is not None:
                extra += f"<br>Blood report age: {age_text(assessment['blood_days'])}"
            if assessment.get("urine_days") is not None:
                extra += f"<br>Urine report age: {age_text(assessment['urine_days'])}"

            st.markdown(
                f"""
            <div class="ns-success">
                <div class="ns-success-ic">✓</div>
                <div class="ns-success-title">Assessment Complete</div>
                <div class="ns-kc-v {rc}" style="font-size:32px;margin:10px 0 4px;">{assessment['risk_score']}% risk · {assessment['risk_level']}</div>
                <div class="ns-success-sub">
                    <b>{conf_note}</b>{extra}<br><br>
                    {html.escape(assessment['advice'])}<br>
                    <span style="font-size:11px;">A single result is not a diagnosis; CKD needs abnormal results for 3+ months.</span>
                </div>
            </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for _n in assessment.get("report_notes") or []:
                st.warning(_n)

            with st.expander("🔍 Why this result", expanded=True):
                for r in assessment["reasons"]:
                    st.markdown(f"- {r}")

            st.markdown('<div class="ns-lbl">Possible kidney conditions — to discuss with your doctor</div>',
                        unsafe_allow_html=True)
            _render_predictions(preds)

            if was_blood and ml_result and ml_result.get("shap_explanation"):
                with st.expander("🧠 AI pattern model — SHAP explanation (specific to your answers)"):
                    st.caption("Which of YOUR answers pushed the AI pattern model up or down (20% weight in the final score).")
                    for item in ml_result["shap_explanation"]:
                        direction = "🔺 increased risk" if item["impact"] > 0 else "🔻 decreased risk"
                        st.markdown(f"**{item['label']}** — {direction} (impact: {item['impact']:+.3f})")

            if clinical_risk and clinical_risk["triggered_factors"]:
                with st.expander(
                    f"📋 Clinical Risk Factor Checklist — {clinical_risk['clinical_risk_score']}/100 ({clinical_risk['clinical_risk_level']})",
                ):
                    st.caption(clinical_risk["note"])
                    for f in clinical_risk["triggered_factors"]:
                        st.markdown(f"**{f['factor']}** — +{f['points']} points")

            if was_blood and st.button("🔍 Also show LIME explanation (takes a few seconds)", key="lime_btn"):
                with st.spinner("Computing LIME explanation..."):
                    lime_result = predict_lifestyle_risk(answers_for_lime, run_lime=True)
                if lime_result.get("lime_explanation"):
                    with st.expander("LIME explanation (independent cross-check)", expanded=True):
                        for item in lime_result["lime_explanation"]:
                            st.markdown(f"**{item['label']}** — impact: {item['impact']:+.3f}")
                else:
                    st.warning("LIME explanation could not be generated (pip install lime).")

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("💬 Ask NephroBot →", use_container_width=True, key="goto_bot_btn"):
                    st.session_state["_show_result"] = False
                    go("chatbot")
            with b2:
                if st.button("🏠 View Dashboard →", use_container_width=True, key="goto_dash_btn"):
                    st.session_state["_show_result"] = False
                    go("dashboard")
            with b3:
                if st.button("👁️ Try Visual Screening →", use_container_width=True, key="goto_eye_btn"):
                    st.session_state["_show_result"] = False
                    go("eye")

            if st.button("✕ Start a new assessment", key="clear_result_btn"):
                st.session_state["_show_result"] = False
                st.rerun()



def page_eye():
    u = st.session_state.user

    with st.container(key="content_wrap"):
        st.markdown(
            """
        <div class="ns-hf-top">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1D4ED8,#3B82F6);
                        border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">👁️</div>
            <div>
                <div class="ns-hft">Visual Screening (Experimental — not in your risk score)</div>
                <div class="ns-hfs">A simple look at the colour of the inner eyelid. It is a hint only — it does not affect your risk score, and a haemoglobin blood test is always more reliable.</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        if not EYE_AVAILABLE:
            st.error(
                "👁️ **Visual Screening isn't set up on this computer yet.**\n\n"
                "**How to fix it — 2 steps:**\n"
                "1. Open a terminal in your project folder and run:\n"
                "   `pip install opencv-python torch torchvision`\n"
                "2. Close the app (Ctrl+C in the terminal) and start it again:\n"
                "   `python -m streamlit run dashboard/app.py`\n\n"
                "No dataset or account needed — this package is all that's missing."
            )
            st.caption(f"Technical detail (for debugging): {EYE_IMPORT_ERROR}")

        photo = None
        with st.container(key="capture_card"):
            tab_cam, tab_up = st.tabs(["📸  Take Photo", "📤  Upload Photo"])
            with tab_cam:
                photo = st.camera_input("Take a photo of the eye", disabled=not EYE_AVAILABLE, label_visibility="collapsed")
            with tab_up:
                photo_up = st.file_uploader(
                    "Upload eye photo (JPG, PNG, WEBP, or BMP)",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    disabled=not EYE_AVAILABLE,
                    label_visibility="collapsed",
                )
                if photo_up is not None:
                    st.image(photo_up, caption="Uploaded photo", width=260)
                    photo = photo_up

        if photo is not None:
            if st.button("Analyze Photo →", use_container_width=True, key="analyze_eye_btn"):
                with st.spinner("Analyzing photo..."):
                    result = analyze_eye_image(photo.getvalue())
                hb_state, hb_val = _hb_status(u)
                if result.get("crop") is not None:
                    st.image(result["crop"], caption="The part of your photo that was checked (the inner eyelid lining)",
                             width=240)
                if result["eye_score"] is None:
                    if result.get("conflict"):
                        _col = (result.get("colour") or {}).get("cls")
                        _seen = {"red": "The lining looks well-coloured (pink-red) in your photo. ",
                                 "pale": "The lining looks light in colour in your photo. "}.get(_col, "")
                        st.info("**No score shown.** " + _seen + result["note"])
                    elif result.get("quality_issue"):
                        st.warning("**Photo not usable — no score shown.** " + result["note"])
                    else:
                        st.error(result["note"])
                else:
                    conf = result.get("eye_confidence")
                    reliable = True  # photo passed the quality check -> always show the score (borderline is shown as borderline)

                    if not reliable:
                        st.warning(
                            "**Inconclusive — no result shown.** The computer model was not sure about this photo. "
                            "Try again near a window in daylight with the lower eyelid gently pulled down. "
                            "Nothing was saved. A haemoglobin blood test is the reliable check."
                        )
                    else:
                        hist = get_user_history(u["id"])
                        scored = [r for r in hist if r.get("risk_score") is not None]
                        latest = scored[0] if scored else None
                        save_assessment(
                            user_id=u["id"],
                            risk_score=latest["risk_score"] if latest else None,
                            ckd_stage=latest["ckd_stage"] if latest else None,
                            risk_level=latest["risk_level"] if latest else None,
                            eye_score=result["eye_score"],
                        )
                        score = result["eye_score"]
                        label = result.get("label") or eye_label(score)
                        src_note = " (colour + AI hint)" if result.get("used_model") else " (colour reading)"
                        risk_cat   = "Low"      if score < 30 else ("Borderline" if score < 55 else "Higher")
                        clr        = "#16A34A"  if score < 30 else ("#D97706"    if score < 55 else "#DC2626")
                        bar_pct    = max(2, min(100, int(score)))
                        _cal = result.get("calibration") or {}
                        if _cal.get("sensitivity_cv") is not None and _cal.get("specificity_cv") is not None:
                            _rel = (f"Colour scale checked on {_cal.get('n')} real photos with known haemoglobin (cross-validated): "
                                    f"it picks up about {_cal['sensitivity_cv'] * 100:.0f}% of anaemic cases and correctly clears "
                                    f"about {_cal['specificity_cv'] * 100:.0f}% of normal ones. It can miss mild anaemia.")
                        else:
                            _rel = ("This colour scale has not yet been validated against blood tests, so treat the number as a "
                                    "rough hint, not a measurement. It can miss mild anaemia.")
                        st.markdown(f"""
<div style="background:#F8FAFC;border:1.5px solid #E2E8F0;border-radius:14px;padding:18px 20px;margin-bottom:10px;">
  <div style="font-size:13px;font-weight:700;color:#0F172A;margin-bottom:10px;">
    &#128065;&#65039; Visual Anemia Screening{src_note}
  </div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
    <div style="font-size:32px;font-weight:900;color:{clr};line-height:1;">{score:.0f}<span style="font-size:13px;font-weight:400;color:#94A3B8;">/100</span></div>
    <div>
      <div style="font-size:15px;font-weight:700;color:{clr};">{label}</div>
      <div style="font-size:12px;color:#64748B;">Pallor level: <b>{risk_cat}</b> &nbsp;·&nbsp; 0 = pink &nbsp;·&nbsp; 100 = very pale</div>
    </div>
  </div>
  <div style="height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden;margin-bottom:10px;">
    <div style="height:100%;width:{bar_pct}%;background:{clr};border-radius:4px;"></div>
  </div>
  <div style="font-size:11.5px;color:#475569;margin-bottom:8px;">{_rel}</div>
  <div style="font-size:11.5px;color:#92400E;background:#FFF7ED;border:1px solid #FDE68A;border-radius:8px;padding:8px 12px;">
    &#9888;&#65039; <b>Experimental screening only</b> — cannot diagnose anemia or kidney disease.
    Always confirm with a haemoglobin (Hb) blood test. Saved to history as a reference only.
  </div>
</div>""", unsafe_allow_html=True)
                        st.info(result["note"])

                    if hb_state == "normal":
                        st.success(
                            f"Your latest blood report shows **normal haemoglobin ({hb_val:.1f} g/dL)**, so you are "
                            "not anaemic. A blood test is far more reliable than a phone photo — please rely on it."
                        )
                    elif hb_state == "low":
                        _male = str(u.get("gender", "")).strip().lower().startswith("m")
                        st.warning(
                            f"Your latest blood report shows **low haemoglobin ({hb_val:.1f} g/dL)** — below the usual "
                            f"cut-off of {13 if _male else 12} g/dL for {'men' if _male else 'women'}. Labs differ slightly, "
                            "and a wrongly typed or misread value can cause this, so please check the number against "
                            "your report and discuss it with a doctor."
                            + (" Note: your photo reading and your blood report differ — mild anaemia often looks normal in "
                               "a photo, so the blood report is the reliable one." if result["eye_score"] < 30 else "")
                        )

                    if st.button("🏠 View Dashboard →", use_container_width=True, key="dash_after_eye"):
                        go("dashboard")


def _patient_context(u):
    hist = get_user_history(u["id"])
    scored = [r for r in hist if r.get("risk_score") is not None]
    parts = [f"Age {u.get('age')}, gender {u.get('gender')}."]
    if not scored:
        return " ".join(parts) + " No assessment completed yet."
    r = scored[0]
    parts.append(f"Latest overall kidney risk {r['risk_score']:.0f}% ({r['risk_level']}).")
    if r.get("egfr"):
        parts.append(f"eGFR {r['egfr']} ({r.get('egfr_stage')}).")
    notes = _parse_notes(get_latest_clinical_notes(u["id"]))
    if notes.get("_confidence"):
        parts.append(f"Result confidence: {notes['_confidence']}.")
    if notes.get("_predictions"):
        parts.append("Possible conditions flagged by the app: "
                     + ", ".join(p["name"] for p in notes["_predictions"]) + ".")
    return " ".join(parts)


def page_chatbot():
    u = st.session_state.user
    import streamlit.components.v1 as _cv1
    import json as _json

    with st.container(key="content_wrap"):
        st.markdown(
            """
        <div class="ns-hf-top">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1D4ED8,#3B82F6);
                        border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">\U0001f4ac</div>
            <div>
                <div class="ns-hft">NephroBot \u2014 ask about kidney or any health topic</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        if "chatbot_history" not in st.session_state:
            st.session_state["chatbot_history"] = []

        # ── Chat history + per-reply speaker buttons ──────────────────────────
        # HOW THE SPEAKER BUTTON WORKS:
        #   Each assistant reply gets its own st.components.v1.html() block
        #   keyed by the message _id.  The block is a self-contained HTML page
        #   that holds the full reply text in a <textarea> element (not a JS
        #   variable) so special chars, quotes, and Devanagari never break JS.
        #   The button uses window.speechSynthesis to speak the text.
        #   Because each block has a unique key, Streamlit does not reuse it
        #   across reruns — the DOM is stable and the button stays visible.
        from chatbot import _detect_language as _dlang

        chat_box = st.container()
        with chat_box:
            for msg in st.session_state["chatbot_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                # Speaker button — only for assistant messages
                if msg["role"] == "assistant":
                    # Determine language of this reply
                    i = st.session_state["chatbot_history"].index(msg)
                    stored_lang = msg.get("_lang", "")
                    if not stored_lang:
                        # Find preceding user message
                        for prev in reversed(st.session_state["chatbot_history"][:i]):
                            if prev["role"] == "user":
                                stored_lang = _dlang(prev["content"])
                                break
                        if not stored_lang:
                            stored_lang = "english"

                    msg_uid = str(msg.get("_id", abs(hash(msg["content"][:30]))))

                    bcp_lang = {"marathi": "mr-IN", "hindi": "hi-IN"}.get(stored_lang, "en-IN")

                    # Safe-encode the reply text for embedding in HTML
                    # Use JSON encode → safe ASCII + unicode escapes
                    safe_text = _json.dumps(msg["content"], ensure_ascii=True)

                    bcp_json  = _json.dumps(bcp_lang)
                    lang_json = _json.dumps(stored_lang)

                    speaker_html = (
                        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
                        "body{margin:0;padding:0;background:transparent;font-family:'Inter',system-ui,sans-serif}"
                        "#spk{"
                        "display:inline-flex;align-items:center;gap:6px;"
                        "border:1.5px solid #E2E8F0;background:#F8FAFC;"
                        "border-radius:20px;padding:5px 14px;"
                        "font-size:12.5px;color:#475569;cursor:pointer;"
                        "transition:all .15s;white-space:nowrap;"
                        "font-family:'Inter',system-ui,sans-serif;"
                        "}"
                        "#spk:hover{background:#EFF6FF;border-color:#93C5FD;color:#1D4ED8}"
                        "#spk.playing{background:#EFF6FF;border-color:#3B82F6;color:#1D4ED8;font-weight:600}"
                        "#st{font-size:11px;color:#94A3B8;margin-left:6px;display:inline}"
                        "</style></head><body>"
                        "<button id='spk'>\U0001f50a Read aloud</button>"
                        "<span id='st'></span>"
                        "<script>"
                        "(function(){"
                        "var TEXT=" + safe_text + ";"
                        "var BCP=" + bcp_json + ";"
                        "var LANG=" + lang_json + ";"
                        "var btn=document.getElementById('spk');"
                        "var st=document.getElementById('st');"
                        "var synth=window.speechSynthesis;"
                        "var playing=false;"
                        # clean markdown
                        "function clean(s){"
                        "return s"
                        ".replace(/\\[([^\\]]+)\\]\\([^)]*\\)/g,'$1')"
                        ".replace(/https?:\\/\\/\\S+/g,'')"
                        ".replace(/[*_`#>~|]/g,'')"
                        ".replace(/^\\s*[-\\u2022]\\s*/gm,'')"
                        ".replace(/\\s{2,}/g,' ').trim();}"
                        # wait for voices
                        "function getVoices(cb){"
                        "var v=synth.getVoices();"
                        "if(v&&v.length){cb(v);return;}"
                        "if('onvoiceschanged'in synth){"
                        "var h=function(){synth.removeEventListener('voiceschanged',h);cb(synth.getVoices());};"
                        "synth.addEventListener('voiceschanged',h);"
                        "setTimeout(function(){cb(synth.getVoices()||[]);},3000);"
                        "}else{setTimeout(function(){cb(synth.getVoices()||[]);},600);}}"
                        # pick voice
                        "function pickVoice(voices){"
                        "var order=LANG==='marathi'?['mr-IN','mr','hi-IN','hi','en-IN','en']"
                        ":LANG==='hindi'?['hi-IN','hi','mr-IN','en-IN','en']"
                        ":['en-IN','en-GB','en-US','en'];"
                        "for(var i=0;i<order.length;i++){"
                        "var p=order[i];"
                        "var v=voices.find(function(x){return x.lang===p;})"
                        "||voices.find(function(x){return x.lang.startsWith(p.split('-')[0]);});"
                        "if(v)return v;}"
                        "return voices[0]||null;}"
                        # stop
                        "function stop(){"
                        "synth.cancel();playing=false;"
                        "btn.classList.remove('playing');"
                        "btn.innerHTML='\U0001f50a Read aloud';"
                        "st.textContent='';}"
                        # speak
                        "function doSpeak(){"
                        "stop();"
                        "var txt=clean(TEXT);"
                        "if(!txt)return;"
                        "playing=true;"
                        "btn.classList.add('playing');"
                        "btn.innerHTML='\u23f9\ufe0f Stop';"
                        "st.textContent='Loading\u2026';"
                        "getVoices(function(voices){"
                        "var voice=pickVoice(voices);"
                        "if(voice)st.textContent='('+ voice.lang+')';"
                        "else st.textContent='';"
                        # split on sentence boundaries to avoid Chrome 15s cutoff
                        "var parts=txt.match(/[^\\u0964\\u0965.!?]+[\\u0964\\u0965.!?]*/g)||[txt];"
                        "var idx=0;"
                        "function next(){"
                        "if(idx>=parts.length||!playing){stop();return;}"
                        "var seg=parts[idx++].trim();"
                        "if(!seg){next();return;}"
                        "var u=new SpeechSynthesisUtterance(seg);"
                        "u.lang=BCP;u.rate=0.87;u.pitch=1.0;"
                        "if(voice)u.voice=voice;"
                        "u.onend=next;"
                        "u.onerror=function(ev){stop();st.textContent='Error: '+(ev.error||'');};"
                        "synth.speak(u);}"
                        "next();});}"
                        "btn.addEventListener('click',function(){"
                        "if(playing){stop();}else{doSpeak();}});"
                        "window.addEventListener('beforeunload',function(){synth.cancel();});"
                        "})();"
                        "</script></body></html>"
                    )
                    _cv1.html(speaker_html, height=40, scrolling=False)

        if st.session_state["chatbot_history"]:
            if st.button("\U0001f5d1\ufe0f Clear chat", key="clear_chatbot_btn"):
                st.session_state["chatbot_history"] = []
                st.rerun()

        # ── Input bar with mic ────────────────────────────────────────────────
        import streamlit.components.v1 as _c

        bar_html = (
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><style>"
            "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}"
            "html,body{background:transparent;font-family:'Inter',system-ui,sans-serif}"
            ".wrap{display:flex;flex-direction:column;gap:6px}"
            ".bar{display:flex;align-items:flex-end;gap:0;background:#1C1C1E;border-radius:22px;"
            "padding:14px 12px 14px 20px;box-shadow:0 2px 18px rgba(0,0,0,.22);transition:box-shadow .2s}"
            ".bar:focus-within{box-shadow:0 2px 18px rgba(0,0,0,.28),0 0 0 2px rgba(59,130,246,.45)}"
            ".bar textarea{flex:1;min-width:0;background:transparent;border:none;outline:none;resize:none;"
            "font-size:15px;color:#F1F5F9;line-height:1.55;min-height:22px;max-height:160px;"
            "font-family:'Inter',system-ui,sans-serif;overflow-y:auto;caret-color:#F1F5F9}"
            ".bar textarea::placeholder{color:#52525B}"
            ".btns{display:flex;align-items:center;gap:2px;flex-shrink:0;padding-bottom:2px}"
            "#mic{width:38px;height:38px;border-radius:50%;border:none;background:transparent;color:#71717A;"
            "cursor:pointer;display:flex;align-items:center;justify-content:center;transition:color .15s,transform .12s;flex-shrink:0}"
            "#mic:hover{color:#E4E4E7;transform:scale(1.1)}"
            "#mic.active{color:#EF4444;animation:pulse 1s ease-in-out infinite}"
            "@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.18)}}"
            "#send{width:36px;height:36px;border-radius:50%;border:none;background:#2563EB;color:#fff;"
            "cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,transform .12s;flex-shrink:0}"
            "#send:hover{background:#1D4ED8;transform:scale(1.08)}"
            "#send:active{transform:scale(.92)}"
            "#send:disabled{background:#27272A;cursor:not-allowed}"
            "#send:disabled svg{opacity:.35}"
            "#status{font-size:11.5px;color:#52525B;padding-left:6px;min-height:15px;transition:color .2s}"
            "#status.active{color:#EF4444;font-weight:600}"
            "#status.ok{color:#22C55E;font-weight:600}"
            "#status.err{color:#F87171}"
            "#nosup{font-size:12px;color:#A1A1AA;padding:0 6px}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='bar'>"
            "<textarea id='txt' rows='1' placeholder='Ask NephroBot\u2026 / \u0935\u093f\u091a\u093e\u0930\u093e\u2026 / \u092a\u0942\u091b\u0947\u0902\u2026'></textarea>"
            "<div class='btns'>"
            "<button id='mic' title='Speak in Marathi, Hindi or English'>"
            "<svg width='20' height='20' viewBox='0 0 24 24' fill='currentColor'>"
            "<path d='M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-6.5 10.5a.75.75 0 0 1 .75.75 5.75 5.75 0 0 0 11.5 0 .75.75 0 0 1 1.5 0 7.25 7.25 0 0 1-6.5 7.21V21h2a.75.75 0 0 1 0 1.5h-5.5a.75.75 0 0 1 0-1.5h2v-1.54a7.25 7.25 0 0 1-6.5-7.21.75.75 0 0 1 .75-.75z'/>"
            "</svg></button>"
            "<button id='send' disabled title='Send'>"
            "<svg width='15' height='15' viewBox='0 0 24 24' fill='currentColor'>"
            "<path d='M3.478 2.405a.75.75 0 0 0-.926.94l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.405z'/>"
            "</svg></button>"
            "</div></div>"
            "<div id='status'></div>"
            "<div id='nosup'></div>"
            "</div>"
            "<script>(function(){"
            "var txt=document.getElementById('txt');"
            "var mic=document.getElementById('mic');"
            "var send=document.getElementById('send');"
            "var status=document.getElementById('status');"
            "var nosup=document.getElementById('nosup');"
            "function resize(){txt.style.height='auto';txt.style.height=Math.min(txt.scrollHeight,160)+'px';}"
            "txt.addEventListener('input',function(){resize();send.disabled=!txt.value.trim();});"
            "function doSend(){"
            "var v=txt.value.trim();if(!v)return;"
            "try{window.sessionStorage.setItem('ns_cb_msg',JSON.stringify({t:v,ts:Date.now()}));}catch(e){}"
            "txt.value='';resize();send.disabled=true;status.className='';status.textContent='';}"
            "send.addEventListener('click',doSend);"
            "txt.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend();}});"
            "var SR=window.SpeechRecognition||window.webkitSpeechRecognition;"
            "if(!SR){mic.style.display='none';nosup.textContent='Voice input needs Chrome or Edge.';}"
            "else{"
            "var rec=null,busy=false,finalT='';"
            "function startRec(lc,retry){"
            "rec=new SR();rec.lang=lc;rec.continuous=false;rec.interimResults=true;rec.maxAlternatives=3;"
            "rec.onstart=function(){busy=true;mic.classList.add('active');status.className='active';"
            "status.textContent=lc==='mr-IN'?'\U0001f534 \u090f\u0948\u0915\u0924\u094b\u2026 (Marathi/Hindi/English)':'\U0001f534 Listening\u2026';};"
            "rec.onresult=function(e){var fin='',it='';"
            "for(var i=e.resultIndex;i<e.results.length;i++){"
            "if(e.results[i].isFinal){"
            "var best=e.results[i][0].transcript;"
            "for(var j=0;j<e.results[i].length;j++){if(/[\\u0900-\\u097F]/.test(e.results[i][j].transcript)){best=e.results[i][j].transcript;break;}}"
            "fin+=best;}else{it+=e.results[i][0].transcript;}}"
            "if(fin)finalT+=fin;txt.value=finalT+it;resize();send.disabled=!txt.value.trim();};"
            "rec.onerror=function(e){"
            "busy=false;mic.classList.remove('active');"
            "var m={'no-speech':'Nothing heard \u2014 tap mic again.','not-allowed':'Mic blocked.','audio-capture':'No mic found.'};"
            "status.className='err';status.textContent=m[e.error]||'Error: '+e.error;};"
            "rec.onend=function(){"
            "busy=false;mic.classList.remove('active');"
            "if(finalT.trim()){"
            "var hasD=/[\\u0900-\\u097F]/.test(finalT);"
            "if(!retry&&!hasD&&lc==='mr-IN'&&finalT.trim().split(/\\s+/).length<=4){"
            "finalT='';txt.value='';status.className='active';status.textContent='\U0001f504 Retrying in English\u2026';"
            "setTimeout(function(){startRec('en-IN',true);},200);return;}"
            "txt.value=finalT.trim();resize();send.disabled=false;"
            "status.className='ok';status.textContent='\u2705 Press Enter or \u27a4 to send';txt.focus();"
            "}else if(status.className!=='err'){status.className='';status.textContent='';}};"
            "try{rec.start();}catch(err){status.className='err';status.textContent='Cannot start: '+err.message;}}"
            "mic.addEventListener('click',function(){if(busy){rec&&rec.stop();return;}finalT='';startRec('mr-IN',false);});}"
            "})();</script></body></html>"
        )

        _c.html(bar_html, height=115, scrolling=False)

        # ── Polling bridge: JS sessionStorage → Streamlit ────────────────────
        _poll = (
            "<script>(function(){"
            "function poll(){"
            "try{"
            "var raw=window.sessionStorage.getItem('ns_cb_msg');"
            "if(!raw)return;"
            "var obj=JSON.parse(raw);"
            "if(Date.now()-obj.ts>8000){window.sessionStorage.removeItem('ns_cb_msg');return;}"
            "window.sessionStorage.removeItem('ns_cb_msg');"
            "var ta=window.parent.document.querySelector('textarea[data-testid=\"stChatInputTextArea\"]');"
            "if(!ta)return;"
            "var setter=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;"
            "setter.call(ta,obj.t);"
            "ta.dispatchEvent(new Event('input',{bubbles:true}));"
            "setTimeout(function(){"
            "ta.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true}));"
            "},80);"
            "}catch(e){}}"
            "setInterval(poll,300);"
            "})();</script>"
        )
        _cv1.html(_poll, height=0, scrolling=False)

        # ── Hidden native chat_input ──────────────────────────────────────────
        st.markdown("""<style>
[data-testid="stChatInput"]{
  opacity:0!important;height:0!important;min-height:0!important;
  overflow:hidden!important;padding:0!important;margin:0!important;
  pointer-events:none!important;position:absolute!important;
}
</style>""", unsafe_allow_html=True)

        user_msg = st.chat_input("Ask NephroBot\u2026")
        if user_msg:
            import time as _time
            from chatbot import _detect_language as _dl2
            st.session_state["chatbot_history"].append({"role": "user", "content": user_msg})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(user_msg)
                with st.chat_message("assistant"):
                    with st.spinner("NephroBot is typing\u2026"):
                        reply = st.write_stream(
                            stream_chatbot_reply(
                                st.session_state["chatbot_history"],
                                _patient_context(u),
                            )
                        )
                    if isinstance(reply, list):
                        reply = "".join(str(x) for x in reply)
                    reply = (reply or "").strip()
                    if reply:
                        st.session_state["chatbot_history"].append({
                            "role":    "assistant",
                            "content": reply,
                            "_id":     int(_time.time() * 1000),
                            "_lang":   _dl2(user_msg),
                        })
                        st.rerun()  # rerun so speaker button renders immediately below reply
                    else:
                        st.error("NephroBot could not answer right now. Please try again.")

def page_find_hospital():
    u = st.session_state.user

    with st.container(key="content_wrap"):
        st.markdown(
            """
        <div class="ns-hf-top">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1D4ED8,#3B82F6);
                        border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">🏥</div>
            <div>
                <div class="ns-hft">Find a Kidney Specialist Near You</div>
                <div class="ns-hfs">Live search across cities, towns and villages worldwide · Filter by budget</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([2.2, 2, 0.9])
        with c1:
            city = st.text_input(
                "City, town or village",
                value=u.get("city", "") if u else "",
                placeholder="e.g. Boisar, Wada, Santacruz, Panvel, Kothrud",
            )
        with c2:
            budget = st.selectbox("Budget preference", list(BUDGET_MAP.keys()))
        with c3:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            search = st.button("Search →", use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if search or city:
            if not city.strip():
                st.warning("Please enter your city, town or village name.")
            else:
                with st.spinner("Searching nearby hospitals..."):
                    results, found, matched = _cached_find_hosp(city.strip(), budget)

                if not found:
                    st.warning(
                        f"No hospitals found in open map data near **{city.title()}**. "
                        f"This is a live map search — coverage can be thin for very remote areas. "
                        f"Try a nearby larger town, or check with a local clinic directly."
                    )

                if matched == "__all__" and results:
                    st.info(f"No hospitals matching “{budget}” were found near {city.title()} — showing all nearby "
                            f"hospitals instead. Budget groups are estimated from the hospital's type and name.")

                if results:
                    st.markdown(
                        f'<div class="ns-lbl">Found {len(results)} result{"s" if len(results) != 1 else ""} — '
                        f'kidney-care first, then nearest</div>',
                        unsafe_allow_html=True,
                    )
                    _missing = [n for k, n in (("low", "Government / free"), ("high", "Premium"))
                                if not any(h.get("budget") == k for h in results)]
                    if _missing:
                        st.info("No " + " or ".join(_missing) + " hospital was found within 150 km in open map data. "
                                "Try a nearby larger city.")
                    for i, h in enumerate(results, 1):
                        st.markdown(_hospital_card_html(i, h, city), unsafe_allow_html=True)
                    st.caption(
                        "Opening hours come from public map data and may be missing or out of date — tap “Timings on Google” or call before "
                        "you go. Budget group is an estimate (Government / charitable = low cost; large or branded private "
                        "hospitals = premium). For emergencies call 112. Free online doctor consultation: "
                        "eSanjeevani (esanjeevani.mohfw.gov.in)."
                    )


def main():
    p = st.session_state.page

    if st.session_state.logged_in and p == "home":
        u = st.session_state.user
        p = "dashboard" if (u and get_user_history(u["id"])) else "lifestyle"
        st.session_state.page = p

    pages = {
        "dashboard": page_dashboard,
        "lifestyle": page_lifestyle,
        "eye": page_eye,
        "chatbot": page_chatbot,
        "find_hospital": page_find_hospital,
    }

    if p == "home":
        page_home()
    elif p in pages:
        if st.session_state.logged_in:
            topbar(show_user=True)
            with st.container(key="shell"):
                nav_col, main_col = st.columns([1, 4.2], gap="small")
                with nav_col:
                    render_nav()
                with main_col:
                    pages[p]()
        else:
            go("home")
    else:
        page_home()


if __name__ == "__main__":
    main()