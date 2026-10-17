"""
places_api.py — NephroSense  (REPLACE existing)

Live hospital search with: distance, opening hours -> "Open now / Closed" (India time), emergency flag,
phone / website / directions, and a CORRECT Government / Mid-range / Premium classification.

Fixes:
  * Budget filter: the old code only ever produced "low" and "mid", so "Premium Private" always came back
    empty. Now: Government + charitable/trust = low, private = mid, premium chains / large or
    "super speciality" private hospitals = high. (Estimated from name / OSM tags — fees are never invented.)
  * "medical college" alone no longer means Government (many are private).
  * Up to 60 places are fetched BEFORE filtering, so a budget filter doesn't leave you with nothing.
  * Results are sorted: kidney-care (nephrology/dialysis/transplant) first, then hospitals before clinics,
    then nearest first.
  * Opening hours come from OpenStreetMap data (via Geoapify / Overpass). If a hospital has no hours listed
    the app says so honestly instead of guessing.

Keys: set GEOAPIFY_API_KEY (free, no card) in .streamlit/secrets.toml or the environment.
Without it (or if it fails) OpenStreetMap (Nominatim + Overpass) is used, which needs no key.
"""
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

# TODO: move this key to .streamlit/secrets.toml (GEOAPIFY_API_KEY) and regenerate it — it was pasted in code.
GEOAPIFY_KEY = os.environ.get("GEOAPIFY_API_KEY", "cc72b1f3700645a7be89ca9273bf1867")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
GEOAPIFY_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"

HEADERS = {
    "User-Agent": "NephroSense-HealthScreening/1.0 (contact: your_email@example.com)",
    "Accept": "application/json",
}

IST = timezone(timedelta(hours=5, minutes=30))

GOVT_RE = re.compile(
    r"\b(government|govt|gov\.|municipal|corporation|district hospital|civil hospital|sub[- ]?district|"
    r"rural hospital|primary health|community health|\bphc\b|\bchc\b|\bgmc\b|esic|cghs|railway|"
    r"army|military|base hospital|zilla|mahanagar|sarkari|aiims|\bpgi\b|\bnims\b|\bkem\b|sassoon|king edward|"
    r"sion hospital|\bjj hospital\b|\bsir j\.?j\b)", re.I)
TRUST_RE = re.compile(r"\b(trust|mission|charitable|charity|seva|sevashram|ngo|lions|rotary|jain seva)\b", re.I)
PREMIUM_RE = re.compile(
    r"(apollo|fortis|max (?:super|hospital|healthcare)|manipal|kokilaben|hinduja|lilavati|jaslok|medanta|narayana|"
    r"wockhardt|nanavati|breach candy|global hospital|gleneagles|asian heart|jupiter|ruby hall|aster|kims|"
    r"yashoda|columbia asia|cloudnine|bombay hospital|sahyadri|dr\.? l h hiranandani|hiranandani|"
    r"super[- ]?speciali?ty|superspeciali?ty|sir h\.? ?n\.? reliance|reliance foundation|"
    r"tata memorial|saifee|fortis|bhatia|breach|artemis|paras|sir ganga ram|bl kapur|rajiv gandhi cancer)", re.I)
NEPHRO_RE = re.compile(r"nephro|kidney|dialysis|renal|transplant|urolog", re.I)


# ───────────────────────── helpers ─────────────────────────
def _km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _classify(name, raw):
    """-> (type_label, budget) from the name and OpenStreetMap tags. Estimates only."""
    raw = raw or {}
    name = name or ""
    op_type = str(raw.get("operator:type", "")).lower()
    operator = str(raw.get("operator", ""))
    text = f"{name} {operator}"
    if op_type in ("government", "public") or GOVT_RE.search(text):
        return "Government", "low"
    if op_type in ("charitable", "community", "religious", "ngo") or TRUST_RE.search(text):
        return "Charitable / Trust", "low"
    beds = 0
    try:
        beds = int(re.sub(r"\D", "", str(raw.get("beds", ""))) or 0)
    except ValueError:
        beds = 0
    if PREMIUM_RE.search(text) or beds >= 200:
        return "Private", "high"
    return "Private", "mid"


def _is_nephro(name, raw):
    raw = raw or {}
    spec = " ".join(str(raw.get(k, "")) for k in ("healthcare:speciality", "speciality", "description"))
    return bool(NEPHRO_RE.search(f"{name} {spec}"))


def _clean_phone(p):
    if not p:
        return ""
    p = str(p).split(";")[0].strip()
    return p


def _normalise(name, address, lat, lon, raw, categories, center, phone="", website="", hours=""):
    h_type, budget = _classify(name, raw)
    raw = raw or {}
    hours = hours or raw.get("opening_hours", "") or ""
    emergency = str(raw.get("emergency", "")).lower() == "yes"
    cats = " ".join(categories or []) if not isinstance(categories, str) else categories
    kind = "Clinic" if ("clinic" in cats and "hospital" not in cats and "hospital" not in (name or "").lower()) else "Hospital"
    nephro = _is_nephro(name, raw)
    tags = []
    if nephro:
        tags.append("Nephrology / Dialysis")
    if emergency:
        tags.append("Emergency")
    if not tags:
        tags.append(kind)
    dist = _km(center[0], center[1], lat, lon) if (center and lat is not None and lon is not None) else None
    return {
        "name": name,
        "type": h_type,
        "kind": kind,
        "address": address or "Address not available",
        "phone": _clean_phone(phone or raw.get("phone") or raw.get("contact:phone")),
        "website": website or raw.get("website") or raw.get("contact:website") or "",
        "hours": hours,
        "emergency": emergency,
        "beds": raw.get("beds", ""),
        "budget": budget,
        "tags": tags,
        "nephro": nephro,
        "rating": "N/A",
        "lat": lat, "lon": lon,
        "distance_km": round(dist, 1) if dist is not None else None,
        "maps_url": f"https://www.google.com/maps?q={lat},{lon}" if lat is not None else "",
        "directions_url": (f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}" if lat is not None else ""),
    }


def _sort_key(h):
    return (0 if h["nephro"] else 1, 0 if h["kind"] == "Hospital" else 1,
            h["distance_km"] if h["distance_km"] is not None else 9999)


def _dedupe(items):
    seen, out = set(), []
    for h in items:
        k = re.sub(r"\W+", "", (h["name"] or "").lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


# ───────────────────── opening hours (OSM syntax) ─────────────────────
_DAYS = ["mo", "tu", "we", "th", "fr", "sa", "su"]
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _days_from(sel):
    sel = sel.strip().lower().replace(" ", "")
    if not sel:
        return list(range(7))
    out = []
    for part in sel.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            if a not in _DAYS or b not in _DAYS:
                raise ValueError("day")
            i, j = _DAYS.index(a), _DAYS.index(b)
            while True:
                out.append(i)
                if i == j:
                    break
                i = (i + 1) % 7
        else:
            if part not in _DAYS:
                raise ValueError("day")
            out.append(_DAYS.index(part))
    return out


def _mins(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def parse_hours(oh):
    """OSM opening_hours (common subset) -> {weekday: [(start_min, end_min)]} or None if not understood."""
    if not oh or not str(oh).strip():
        return None
    text = str(oh).strip().lower()
    if text in ("24/7", "24 / 7"):
        return {d: [(0, 1440)] for d in range(7)}
    sched = {d: [] for d in range(7)}
    seen_rule = False
    for rule in re.split(r"\s*;\s*", text):
        rule = rule.strip()
        if not rule:
            continue
        if re.match(r"^(ph|sh)\b", rule):            # public/school-holiday-only rules are ignored
            continue
        rule = re.sub(r"\b(ph|sh)\b\s*,?", "", rule).strip(" ,")
        m = re.match(r"^(?P<days>[a-z ,\-]*?)\s*(?P<times>(?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}(?:\s*,\s*)?)+|off|closed|24/7)$", rule)
        if not m:
            return None                                # unsupported syntax -> say "not listed", never guess
        try:
            days = _days_from(m.group("days"))
        except ValueError:
            return None
        times = m.group("times")
        if times in ("off", "closed"):
            spans = []
        elif times == "24/7":
            spans = [(0, 1440)]
        else:
            spans = []
            for a, b in re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", times):
                s, e = _mins(a), _mins(b)
                if e <= s:
                    e += 1440                          # runs past midnight
                spans.append((s, e))
        for d in days:
            sched[d] = list(spans)
        seen_rule = True
    return sched if seen_rule else None


def _fmt(minutes):
    minutes %= 1440
    h, m = divmod(minutes, 60)
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ap}"


def open_status(oh, now=None):
    """-> (state, text): state is 'open' | 'closed' | None (hours not listed / not understood)."""
    sched = parse_hours(oh)
    if sched is None:
        return None, "Timing not listed — please call to confirm"
    now = now or datetime.now(IST)
    d, mins = now.weekday(), now.hour * 60 + now.minute
    if all(sched[x] == [(0, 1440)] for x in range(7)):
        return "open", "Open 24 hours"
    until = None
    for s, e in sched[d]:
        if s <= mins < e:
            until = e
    for s, e in sched[(d - 1) % 7]:
        if e > 1440 and mins < e - 1440:
            until = e - 1440 if until is None else max(until, e - 1440)
    if until is not None:
        return "open", f"Open now · closes {_fmt(until)}"
    for add in range(0, 8):
        day = (d + add) % 7
        for s, e in sorted(sched[day]):
            if add > 0 or s > mins:
                when = "today" if add == 0 else ("tomorrow" if add == 1 else _DAY_NAMES[day])
                return "closed", f"Closed now · opens {when} {_fmt(s)}"
    return "closed", "Closed now"


# ───────────────────────── Geoapify ─────────────────────────
def _geoapify_geocode(city):
    if not GEOAPIFY_KEY:
        return None
    try:
        r = requests.get(GEOAPIFY_GEOCODE_URL, params={"text": city, "apiKey": GEOAPIFY_KEY, "limit": 1,
                                                       "filter": "countrycode:in"}, timeout=10)
        feats = r.json().get("features", []) if r.status_code == 200 else []
        if not feats:
            r = requests.get(GEOAPIFY_GEOCODE_URL, params={"text": city, "apiKey": GEOAPIFY_KEY, "limit": 1}, timeout=10)
            feats = r.json().get("features", []) if r.status_code == 200 else []
        if feats:
            lon, lat = feats[0]["geometry"]["coordinates"]
            return lat, lon
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return None


def _search_geoapify(city, max_results, radii=(30000, 80000)):
    if not GEOAPIFY_KEY or GEOAPIFY_KEY == "PASTE_YOUR_KEY_HERE":
        return None
    center = _geoapify_geocode(city)
    if center is None:
        return None
    lat0, lon0 = center
    results = []
    for radius in radii:
        try:
            r = requests.get(
                GEOAPIFY_PLACES_URL,
                params={"categories": "healthcare.hospital,healthcare.clinic_or_praxis",
                        "filter": f"circle:{lon0},{lat0},{radius}", "bias": f"proximity:{lon0},{lat0}",
                        "limit": 60, "apiKey": GEOAPIFY_KEY},
                timeout=20)
            if r.status_code != 200:
                return None if not results else results
            results = []
            for f in r.json().get("features", []):
                p = f.get("properties", {})
                name = p.get("name")
                if not name:
                    continue
                raw = (p.get("datasource") or {}).get("raw") or {}
                contact = p.get("contact") if isinstance(p.get("contact"), dict) else {}
                results.append(_normalise(
                    name, p.get("formatted"), p.get("lat"), p.get("lon"), raw, p.get("categories"), center,
                    phone=contact.get("phone") or "", website=p.get("website") or contact.get("website") or "",
                    hours=p.get("opening_hours") or raw.get("opening_hours") or ""))
        except (requests.RequestException, ValueError, KeyError):
            return None if not results else results
        if len(_dedupe(results)) >= 10:
            break
    out = _dedupe(results)
    out.sort(key=_sort_key)
    return out[:max_results]


# ───────────────────────── OpenStreetMap fallback ─────────────────────────
def _geocode_with_bbox(city):
    for query_text in (f"{city}, India", city):
        try:
            resp = requests.get(NOMINATIM_URL, params={"q": query_text, "format": "json", "limit": 1}, headers=HEADERS,
                                timeout=10)
            if resp.status_code == 200 and resp.json():
                r = resp.json()[0]
                return float(r["lat"]), float(r["lon"])
        except (requests.RequestException, ValueError, KeyError):
            continue
    return None


def _query_overpass(lat, lon, radius_m, max_results, center):
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});
      way["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});
      node["healthcare"~"hospital|clinic|dialysis"](around:{radius_m},{lat},{lon});
      way["healthcare"~"hospital|clinic|dialysis"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue
            items = []
            for el in resp.json().get("elements", []):
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                if "lat" in el:
                    h_lat, h_lon = el["lat"], el["lon"]
                elif "center" in el:
                    h_lat, h_lon = el["center"]["lat"], el["center"]["lon"]
                else:
                    continue
                parts = [tags.get(k) for k in ("addr:housenumber", "addr:street", "addr:suburb", "addr:city") if tags.get(k)]
                cats = tags.get("amenity", "") + " " + tags.get("healthcare", "")
                items.append(_normalise(name, ", ".join(parts), h_lat, h_lon, tags, cats, center))
            out = _dedupe(items)
            out.sort(key=_sort_key)
            return out[:max_results]
        except (requests.RequestException, ValueError):
            time.sleep(0.5)
            continue
    return None


def search_hospitals_live(city, max_results=60, radii=(30000, 80000)):
    """List of hospital dicts (sorted), [] when none found, or None when the search itself failed.
    radii (metres) are tried in order until enough places are found."""
    res = _search_geoapify(city, max_results, radii)
    if res:
        return res
    center = _geocode_with_bbox(city)
    if center is None:
        return None
    for radius in radii:
        res = _query_overpass(center[0], center[1], radius, max_results, center)
        if res:
            return res
        if res is None:
            return None
    return []