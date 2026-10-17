"""
test_osm_places.py
────────────────────
Run from your project root:
    python src/test_osm_places.py "Lonavala"

This version adds real diagnostics for the 403 case: it checks
whether Python can reach the internet at ALL (a totally unrelated
site), whether a proxy is configured, and prints the actual response
BODY on a 403 (servers often explain the block in the body text) —
instead of just the status code, which doesn't tell you why.
"""
import os
import sys
import requests

city = sys.argv[1] if len(sys.argv) > 1 else "Lonavala"
HEADERS = {
    "User-Agent": "NephroSense-StudentProject/1.0 (contact: your_email@example.com)",
    "Accept": "application/json",
    "Referer": "https://nominatim.openstreetmap.org/",
}

print("=" * 60)
print("STEP 0: Basic connectivity + proxy check")
print("=" * 60)
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    val = os.environ.get(var)
    if val:
        print(f"  ⚠ Proxy env var set: {var} = {val}")
if not any(os.environ.get(v) for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")):
    print("  No proxy environment variables set.")

try:
    r = requests.get("https://api.github.com", timeout=6)
    print(f"  Unrelated site (api.github.com) status: {r.status_code} — "
          f"{'general internet access looks fine' if r.status_code == 200 else 'even unrelated sites are failing — likely a network/firewall issue, not this API'}")
except Exception as e:
    print(f"  Could NOT reach api.github.com at all: {e}")
    print("  -> This points to a network-level block (firewall/antivirus/VPN/proxy),")
    print("     not something wrong with this project's code.")

print()
print("=" * 60)
print("STEP 1: Offline coordinate lookup (no internet needed)")
print("=" * 60)
try:
    sys.path.append(os.path.dirname(__file__))
    from places_api import _lookup_local_coords
    local = _lookup_local_coords(city)
    if local:
        print(f"  ✓ Found '{city}' in the offline table: {local}")
        print("  This means the app will work for this town even if the")
        print("  network checks below fail.")
    else:
        print(f"  '{city}' is not in the offline table yet — add it to")
        print("  COORD_LOOKUP in places_api.py for guaranteed offline coverage.")
except ImportError as e:
    print(f"  Could not import places_api.py: {e}")

print()
print("=" * 60)
print(f"STEP 2: Nominatim for '{city}, India'")
print("=" * 60)
lat, lon = None, None
try:
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": f"{city}, India", "format": "json", "limit": 1},
        headers=HEADERS, timeout=8,
    )
    print(f"  HTTP status: {resp.status_code}")
    if resp.status_code == 200:
        results = resp.json()
        if results:
            lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
            print(f"  SUCCESS via Nominatim: lat={lat}, lon={lon}")
        else:
            print("  Got 200 OK but no results — try a more specific place name.")
    else:
        print(f"  Response body (first 300 chars): {resp.text[:300]!r}")
except Exception as e:
    print(f"  Nominatim request failed entirely: {e}")

if lat is None:
    print()
    print("=" * 60)
    print("STEP 3: Photon fallback")
    print("=" * 60)
    try:
        resp = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": f"{city}, India", "limit": 1}, timeout=8,
        )
        print(f"  HTTP status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                lon, lat = features[0]["geometry"]["coordinates"]
                print(f"  SUCCESS via Photon: lat={lat}, lon={lon}")
        else:
            print(f"  Response body (first 300 chars): {resp.text[:300]!r}")
    except Exception as e:
        print(f"  Photon request failed entirely: {e}")

if lat is None:
    print()
    print("Both online geocoders failed. Read STEP 0's result above:")
    print("  - If api.github.com ALSO failed -> your network/firewall/antivirus")
    print("    is blocking Python's outbound requests generally. Try a mobile")
    print("    hotspot to confirm, then fix the proxy/firewall/antivirus setting.")
    print("  - If api.github.com worked FINE but these two didn't -> something")
    print("    is specifically blocking openstreetmap.org / komoot.io domains")
    print("    (school/office content filter, or antivirus 'web protection').")
    print("  - Either way: the app will still work for any town already listed")
    print("    in COORD_LOOKUP (places_api.py) or the curated HOSP list in app.py.")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 4: Overpass API — searching for hospitals")
print("=" * 60)
for radius_km in (20, 50, 100):
    radius_m = radius_km * 1000
    print(f"\n  Querying within {radius_km}km ...")
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query}, headers=HEADERS, timeout=20,
    )
    print(f"  HTTP status: {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        print(f"  Could not parse response. Raw text (first 300 chars): {resp.text[:300]}")
        continue

    elements = data.get("elements", [])
    named = [e for e in elements if e.get("tags", {}).get("name")]
    print(f"  Total elements: {len(elements)}, named: {len(named)}")

    if named:
        print(f"\n  SUCCESS at {radius_km}km radius:")
        for e in named[:5]:
            print(f"    - {e['tags'].get('name')} ({e['tags'].get('amenity')})")
        break
else:
    print("\n  No named hospitals found even at 100km — genuine OSM data gap for this area.")