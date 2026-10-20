"""
triage.py — NephroSense
Red-flag check. Runs BEFORE and independently of the risk score: if any flag is
present the user is told to see a doctor / go to hospital now, whatever the band.
"""
RED_FLAGS = {
    "swelling": "New swelling of face, legs or feet",
    "little_urine": "Passing very little or no urine",
    "breathless": "Breathlessness, or difficulty breathing lying flat",
    "vomiting": "Repeated vomiting / cannot keep food or water down",
}


def check_red_flags(answers):
    """answers: {flag_key: bool or 'yes'/'no'}. Unanswered flags are reported, not assumed 'no'."""
    present, unanswered = [], []
    for k in RED_FLAGS:
        v = answers.get(k)
        if isinstance(v, str):
            v = {"yes": True, "no": False}.get(v.strip().lower())
        if v is None:
            unanswered.append(k)
        elif v:
            present.append(k)
    urgent = bool(present)
    return {
        "urgent": urgent,
        "flags": [RED_FLAGS[k] for k in present],
        "unanswered": unanswered,
        "message": ("See a doctor or go to the nearest hospital NOW. Do not wait for a risk score."
                    if urgent else ""),
    }
