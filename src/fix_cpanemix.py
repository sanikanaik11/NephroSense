"""
fix_cpanemix.py — NephroSense
Copies images from the already-extracted CP-AnemiC folder's Anemic/
Non-anemic subfolders into data/dataset anemia/Ghana/anemia/ and
normal/, so train_anemia_model.py picks them up.

HOW TO RUN (from nephrosense/ folder):
    python src/fix_cpanemix.py
"""
import os
import shutil
import sys
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

print(f"Scanning {DATA_DIR} for a CP-AnemiC folder ...")

cp_root = None
for name in os.listdir(DATA_DIR):
    full = os.path.join(DATA_DIR, name)
    if not os.path.isdir(full):
        continue
    low = name.lower()
    if "cp-anemi" in low or "conjunctival" in low or "cp_anemi" in low:
        cp_root = full
        print(f"  Matched folder: {name}")
        break

if cp_root is None:
    print("FAILED: no folder under data/ matched 'cp-anemi' or 'conjunctival'.")
    print(f"Folders currently in {DATA_DIR}:")
    for name in os.listdir(DATA_DIR):
        print(f"   - {name}")
    sys.exit(1)

print(f"Found CP-AnemiC folder: {cp_root}")
print(f"Contents: {os.listdir(cp_root)}")

anemic_src = None
non_anemic_src = None
for item in os.listdir(cp_root):
    full = os.path.join(cp_root, item)
    if not os.path.isdir(full):
        continue
    low = item.lower()
    if "non" in low or "normal" in low:
        non_anemic_src = full
    elif "anemi" in low:
        anemic_src = full

print(f"Anemic subfolder:     {anemic_src}")
print(f"Non-anemic subfolder: {non_anemic_src}")

if anemic_src is None and non_anemic_src is None:
    print("FAILED: could not find Anemic/Non-anemic subfolders inside it.")
    sys.exit(1)

dest_anemia = os.path.join(DATA_DIR, "dataset anemia", "Ghana", "anemia")
dest_normal = os.path.join(DATA_DIR, "dataset anemia", "Ghana", "normal")
os.makedirs(dest_anemia, exist_ok=True)
os.makedirs(dest_normal, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def copy_images(src_folder, dest_folder):
    copied, skipped = 0, 0
    if not src_folder or not os.path.isdir(src_folder):
        return 0, 0
    for root, dirs, files in os.walk(src_folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                continue
            fpath = os.path.join(root, fname)
            try:
                with PILImage.open(fpath) as img:
                    img.verify()
            except Exception:
                skipped += 1
                continue
            dest = os.path.join(dest_folder, f"ghana_{fname}")
            if not os.path.exists(dest):
                shutil.copy2(fpath, dest)
                copied += 1
    return copied, skipped


print("\nCopying images ...")
ca, sa = copy_images(anemic_src, dest_anemia)
cn, sn = copy_images(non_anemic_src, dest_normal)

print(f"\nDone.")
print(f"  Anemic copied:  {ca}  (skipped corrupt: {sa})")
print(f"  Normal copied:  {cn}  (skipped corrupt: {sn})")
print(f"\nNow run: python src/train_anemia_model.py")