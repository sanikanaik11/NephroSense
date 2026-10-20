"""
screening_page.py — NephroSense
Home Screening page (test-free kidney risk pre-screen), consent screen, yearly
re-screen reminder. Plugs into dashboard/app.py via page_screening(), consent_screen()
and reminder_banner(). Uses the same CSS classes as the other pages.
"""
import html
import json
import re
from datetime import date, timedelta

import streamlit as st
import streamlit.components.v1 as components

from screening_text import LANGS, BCP, FIELD, tr
from screening_db import (has_consent, record_consent, save_screening, list_screenings,
                          last_self_screening_date, delete_screening_data)
from testfree_risk import load_bundle, score, FEATURES  # noqa: F401
from home_measurements import home_bp_average, bmi as calc_bmi, valid_reading
from triage import RED_FLAGS, check_red_flags

BAND_STYLE = {"LOW": "#16A34A", "POSSIBLE": "#D97706", "HIGH": "#DC2626"}
CARD = ("background:#fff;border:1.5px solid #E2E8F0;border-radius:14px;padding:18px 20px;margin:10px 0;"
        "box-shadow:0 2px 6px rgba(15,23,42,.08),0 8px 18px -10px rgba(15,23,42,.18);")


@st.cache_resource
def _bundle():
    return load_bundle()


def _lang():
    return st.session_state.get("_lang", "en")


def _language_picker():
    codes = list(LANGS)
    cur = codes.index(_lang())
    pick = st.radio(tr("language", _lang()), [LANGS[c] for c in codes], index=cur, horizontal=True,
                    key="_lang_radio")
    code = codes[[LANGS[c] for c in codes].index(pick)]
    if code != _lang():
        st.session_state["_lang"] = code
        st.rerun()


def _header(icon, title, sub):
    st.markdown(
        f"""<div class="ns-hf-top">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#1D4ED8,#3B82F6);border-radius:11px;
        display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">{icon}</div>
        <div><div class="ns-hft">{html.escape(title)}</div><div class="ns-hfs">{html.escape(sub)}</div></div>
        </div>""", unsafe_allow_html=True)


# ───────────────────────── consent ─────────────────────────
def consent_screen(u):
    L = _lang()
    with st.container(key="content_wrap"):
        _header("🔒", tr("consent_title", L), tr("subtitle", L))
        _language_picker()
        L = _lang()
        items = "".join(f"<li style='margin-bottom:8px'>{html.escape(tr(f'consent_{i}', L))}</li>" for i in range(1, 6))
        st.markdown(f"<div style=\"{CARD}\"><ul style='margin:0;padding-left:20px;color:#0F172A;font-size:14px;"
                    f"line-height:1.55'>{items}</ul></div>", unsafe_allow_html=True)
        ok = st.checkbox(tr("consent_check", L), key="consent_ok")
        if st.button(tr("consent_btn", L), type="primary", use_container_width=True, key="consent_go"):
            if ok:
                record_consent(u["id"])
                st.rerun()
            else:
                st.warning(tr("consent_check", L))


# ───────────────────────── reminder ─────────────────────────
def _ics(due):
    d = due.strftime("%Y%m%d")
    return ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//NephroSense//EN\r\nBEGIN:VEVENT\r\n"
            f"UID:nephrosense-{d}@local\r\nDTSTAMP:{date.today().strftime('%Y%m%d')}T000000Z\r\n"
            f"DTSTART;VALUE=DATE:{d}\r\nRRULE:FREQ=YEARLY\r\n"
            "SUMMARY:NephroSense yearly kidney re-screen\r\n"
            "DESCRIPTION:Repeat the NephroSense home kidney screening.\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n")


def reminder_banner(u, go):
    """Shows a banner when the user's own last screening is a year old (or older)."""
    last = last_self_screening_date(u["id"])
    if last and (date.today() - last).days >= 365:
        L = _lang()
        st.warning(tr("due_banner", L))
        if st.button(tr("start_now", L), key="due_start"):
            go("screening")


# ───────────────────────── BP entry ─────────────────────────
def _parse_pairs(text):
    nums = re.findall(r"\d{2,3}", text or "")
    pairs, ignored = [], []
    for i in range(0, len(nums) - 1, 2):
        r = {"sys": float(nums[i]), "dia": float(nums[i + 1])}
        (pairs if valid_reading(r) else ignored).append(r if valid_reading(r) else f"{nums[i]} {nums[i + 1]}")
    if len(nums) % 2:
        ignored.append(nums[-1])
    return pairs, ignored


def _add_bp():
    pairs, ignored = _parse_pairs(st.session_state.get("scr_bp_text", ""))
    day = int(st.session_state.get("scr_bp_day", 1))
    st.session_state.setdefault("scr_bp", []).extend({"day": day, **p} for p in pairs)
    st.session_state["scr_bp_ignored"] = ignored
    st.session_state["scr_bp_text"] = ""


def _clear_bp():
    st.session_state["scr_bp"] = []
    st.session_state["scr_bp_ignored"] = []


def _mic_widget(lang):
    """Mic button that dictates into the st.text_input with key scr_bp_text (same technique as the notes box)."""
    components.html(f"""<!DOCTYPE html><html><body style="margin:0;font-family:system-ui,sans-serif">
<button id="m" style="border:1.5px solid #CBD5E1;background:#fff;border-radius:20px;padding:6px 14px;cursor:pointer;font-size:13px">🎤</button>
<span id="s" style="font-size:12px;color:#64748B;margin-left:8px"></span>
<script>(function(){{var b=document.getElementById('m'),s=document.getElementById('s');
var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SR){{b.style.display='none';s.textContent='Voice input needs Chrome or Edge.';return;}}
function field(){{try{{return window.parent.document.querySelector('.st-key-scr_bp_text input');}}catch(e){{return null;}}}}
b.onclick=function(){{var r=new SR();r.lang='{BCP[lang]}';r.interimResults=false;
r.onstart=function(){{s.textContent='🔴 …';}};
r.onerror=function(e){{s.textContent=e.error;}};
r.onresult=function(e){{var t=e.results[0][0].transcript,f=field();if(!f)return;
var P=window.parent,set=Object.getOwnPropertyDescriptor(P.HTMLInputElement.prototype,'value').set;
set.call(f,(f.value?f.value+' ':'')+t);f.dispatchEvent(new P.Event('input',{{bubbles:true}}));
f.dispatchEvent(new P.FocusEvent('blur',{{bubbles:true}}));s.textContent='✅ '+t;}};
try{{r.start();}}catch(x){{s.textContent=x.message;}}}};}})();</script></body></html>""", height=44)


def _speak_button(text, lang):
    components.html(f"""<!DOCTYPE html><html><body style="margin:0">
<button id="b" style="border:1.5px solid #CBD5E1;background:#fff;border-radius:20px;padding:6px 14px;cursor:pointer;font-size:13px">{html.escape(tr('listen', lang))}</button>
<script>document.getElementById('b').onclick=function(){{var s=window.speechSynthesis;if(!s)return;s.cancel();
var u=new SpeechSynthesisUtterance({json.dumps(text)});u.lang='{BCP[lang]}';u.rate=0.9;s.speak(u);}};</script>
</body></html>""", height=44)


# ───────────────────────── page ─────────────────────────
def page_screening(u, go):
    L = _lang()
    bundle = _bundle()
    with st.container(key="content_wrap"):
        _header("🩺", tr("title", L), tr("subtitle", L))
        _language_picker()
        L = _lang()

        if bundle is None:
            st.error("Model file not found. Train it once: `python src/nhanes_ckd_model.py` "
                     "(copies models/testfree_model.joblib into your project's models folder).")
            return

        hw = st.checkbox(tr("hw_mode", L), key="scr_hw")
        label = ""
        if hw:
            label = st.text_input(tr("person_label", L), key="scr_label").strip()

        # ── Step 1: danger signs (always first, independent of the score) ──
        st.markdown(f"**{tr('step1', L)}**")
        st.caption(tr("triage_q", L))
        yn = [tr("yes", L), tr("no", L)]
        flag_ans = {}
        for k in RED_FLAGS:
            v = st.radio(tr(k, L), yn, index=None, horizontal=True, key=f"scr_flag_{k}")
            flag_ans[k] = None if v is None else (v == yn[0])
        tri = check_red_flags(flag_ans)
        if tri["urgent"]:
            st.error(f"🚨 {tr('urgent', L)}")
            st.info(tr("urgent_no_score", L))
            if st.button(tr("hosp_btn", L), type="primary", use_container_width=True, key="scr_urgent_hosp"):
                go("find_hospital")
            return
        if tri["unanswered"]:
            st.info(tr("triage_unanswered", L))
            return

        # ── Step 2: about the person ──
        st.markdown(f"**{tr('step2', L)}**")
        c1, c2 = st.columns(2)
        prof_age = None if hw else (u.get("age") or None)
        with c1:
            age = st.number_input(tr("age", L), min_value=0, max_value=120, value=prof_age or None,
                                  step=1, key="scr_age")
        sex_opts = [tr("female", L), tr("male", L)]
        g = str(u.get("gender", "")).strip().lower()
        sex_default = None if hw else (0 if g.startswith("f") else 1 if g.startswith("m") else None)
        with c2:
            sex_pick = st.radio(tr("sex", L), sex_opts, index=sex_default, horizontal=True, key="scr_sex")
        answers = {}
        for k in ("diabetes", "hypertension", "heart_disease", "kidney_stones"):
            v = st.radio(tr(f"q_{k}", L), yn, index=None, horizontal=True, key=f"scr_{k}")
            answers[k] = None if v is None else ("yes" if v == yn[0] else "no")

        # ── Step 3: home measurements ──
        st.markdown(f"**{tr('step3', L)}**")
        st.caption(tr("bp_howto", L))
        d1, d2 = st.columns([1, 3])
        with d1:
            st.selectbox(tr("day", L), list(range(1, 8)), key="scr_bp_day",
                         format_func=lambda d: f"{tr('day', L)} {d}")
        with d2:
            st.text_input(tr("bp_input", L), key="scr_bp_text", placeholder=tr("bp_placeholder", L))
        m1, m2, m3 = st.columns([1.2, 1.2, 2])
        with m1:
            st.button(tr("add_bp", L), on_click=_add_bp, key="scr_bp_add", use_container_width=True)
        with m2:
            st.button(tr("clear_bp", L), on_click=_clear_bp, key="scr_bp_clr", use_container_width=True)
        with m3:
            _mic_widget(L)
        readings = st.session_state.get("scr_bp", [])
        bp = home_bp_average(readings)
        if st.session_state.get("scr_bp_ignored"):
            st.warning(tr("bp_ignored", L, x=", ".join(map(str, st.session_state["scr_bp_ignored"]))))
        if readings:
            st.caption(tr("bp_status", L, n=bp["n"], d=bp["days"]))
            by_day = {}
            for r in readings:
                by_day.setdefault(int(r["day"]), []).append(f"{r['sys']:.0f}/{r['dia']:.0f}")
            st.caption(" · ".join(f"{tr('day', L)} {d}: {', '.join(v)}" for d, v in sorted(by_day.items())))
        if bp["ok"]:
            st.success(tr("bp_avg", L, s=bp["sbp"], d=bp["dbp"], n=bp["n"]))

        w1, w2, w3 = st.columns(3)
        with w1:
            wt = st.number_input(tr("weight", L), min_value=0.0, max_value=300.0, value=None, step=0.5, key="scr_wt")
        with w2:
            ht = st.number_input(tr("height", L), min_value=0.0, max_value=250.0, value=None, step=0.5, key="scr_ht")
        with w3:
            waist = st.number_input(tr("waist", L), min_value=0.0, max_value=250.0, value=None, step=0.5,
                                    key="scr_waist")
        strip_opts = [tr("strip_none", L), tr("strip_neg", L), tr("strip_pos", L)]
        strip_pick = st.radio(tr("strip", L), strip_opts, index=0, horizontal=True, key="scr_strip")
        strip = {1: "negative", 2: "positive"}.get(strip_opts.index(strip_pick))

        if st.button(tr("check_btn", L), type="primary", use_container_width=True, key="scr_check"):
            inputs = {
                "age": age or None,
                "sex": None if sex_pick is None else ("female" if sex_pick == sex_opts[0] else "male"),
                "sbp": bp.get("sbp") if bp["ok"] else None,
                "bmi": calc_bmi(wt, ht) if (wt and ht) else None,
                "waist_cm": waist or None,
                **answers,
            }
            res = score(inputs, bundle, strip_result=strip)
            st.session_state["scr_result"] = {"res": res, "strip": strip, "lang_at": L}
            if res["scored"]:
                next_due = date.today() + timedelta(days=365)
                save_screening(u["id"], res["band"], {
                    "inputs": {k: v for k, v in inputs.items()},
                    "reasons": res["reasons"], "strip": strip,
                    "bp": {"n": bp.get("n"), "days": bp.get("days"), "sbp": bp.get("sbp"), "dbp": bp.get("dbp")},
                    "next_due": next_due.isoformat()}, person_label=label if hw else "")

        out = st.session_state.get("scr_result")
        if out:
            _show_result(out["res"], out["strip"], go, hw)

        _history_and_delete(u, L)


def _show_result(res, strip, go, hw):
    L = _lang()
    if not res["scored"]:
        names = [FIELD[m][L] for m in res["missing"]] if res["missing"] else []
        st.markdown(
            f"<div style=\"{CARD}border-color:#FDE68A;background:#FFFBEB\"><b>{html.escape(tr('refuse_title', L))}</b><br>"
            f"<span style='font-size:13px;color:#475569'>{html.escape(tr('refuse_note', L))}</span><ul>"
            + "".join(f"<li>{html.escape(n)}</li>" for n in names)
            + "".join(f"<li>{html.escape(tr('not_usable', L))}: {html.escape(i)}</li>" for i in res["invalid"])
            + "</ul></div>", unsafe_allow_html=True)
        return
    band, colour = res["band"], BAND_STYLE[res["band"]]
    msg = tr(f"msg_{band}", L)
    st.markdown(
        f"<div style=\"{CARD}border-left:6px solid {colour}\">"
        f"<div style='font-size:12px;color:#64748B;font-weight:700;text-transform:uppercase'>{html.escape(tr('title', L))}</div>"
        f"<div style='font-size:28px;font-weight:800;color:{colour};margin:4px 0'>{html.escape(tr(f'band_{band}', L))}</div>"
        f"<div style='font-size:14px;color:#0F172A;line-height:1.55'>{html.escape(msg)}</div></div>",
        unsafe_allow_html=True)

    st.markdown(f"**{tr('reasons_h', L)}**")
    shown = []
    for r in res["reasons"]:
        base = FIELD.get({"sbp": "sbp_val", "bmi": "bmi_val", "waist_cm": "waist_val"}.get(r["key"], r["key"]),
                         FIELD["age"])[L]
        shown.append(f"{base} ({r['value']:g})" if r.get("value") is not None else base)
    if shown:
        st.markdown("\n".join(f"- {s}" for s in shown))
    else:
        st.caption(tr("no_reasons", L))
    if strip == "positive":
        st.caption(tr("strip_note_pos", L))
    elif strip == "negative":
        st.caption(tr("strip_note_neg", L))

    rate = (res.get("band_stats") or {}).get("kidney_marker_rate")
    if rate is not None:
        st.caption(tr("group_stat", L, k=round(rate * 100)))
    st.caption(tr("not_diag", L))

    due = date.today() + timedelta(days=365)
    st.info(tr("next_due", L, d=due.strftime("%d %b %Y")))
    spoken = " ".join([tr(f"band_{band}", L) + ".", msg] + ([tr("reasons_h", L) + ": " + "; ".join(shown)] if shown else [])
                      + [tr("not_diag", L)])
    a, b, c = st.columns([1, 1.4, 1.6])
    with a:
        _speak_button(spoken, L)
    with b:
        if not hw:
            st.download_button(tr("ics_btn", L), _ics(due), file_name="nephrosense_rescreen.ics",
                               mime="text/calendar", key="scr_ics", use_container_width=True)
    with c:
        if st.button(tr("hosp_btn", L), key="scr_hosp", use_container_width=True):
            go("find_hospital")


def _history_and_delete(u, L):
    rows = list_screenings(u["id"], limit=8)
    if rows:
        with st.expander(tr("history_h", L)):
            for r in rows:
                who = f" — {r['person_label']}" if r["person_label"] else ""
                st.caption(f"{r['created_at'][:10]} · {tr('band_' + r['band'], L) if r['band'] in BAND_STYLE else r['band']}{who}")
    with st.expander(tr("delete_btn", L)):
        sure = st.checkbox(tr("delete_confirm", L), key="scr_del_ok")
        if st.button(tr("delete_btn", L), key="scr_del") and sure:
            delete_screening_data(u["id"])
            st.session_state.pop("scr_result", None)
            st.success(tr("deleted", L))
            st.rerun()
