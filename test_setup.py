########################################
# test_setup.py
# NephroSense — Setup Verification File
# Run this ONCE after setup to check everything works
########################################

print("🔬 NephroSense Setup Check Starting...")
print("=" * 50)

# ─── STEP 1: Check Python Version ───
import sys

print(f"✅ Python version: {sys.version}")

if sys.version_info[0] == 3 and sys.version_info[1] >= 10:
    print(" ✅ Python version is 3.10+ — Good!")
else:
    print(" ❌ Need Python 3.10 or higher")

print("\n📦 Checking Libraries...")
print("-" * 30)

# ─── STEP 2: Check all libraries ───
libraries_ok = True


def check(name, import_name=None):
    global libraries_ok
    try:
        import importlib

        mod = importlib.import_module(import_name or name)
        version = getattr(mod, "__version__", "installed")
        print(f" ✅ {name}: {version}")
    except ImportError:
        print(f" ❌ {name}: NOT FOUND — run: pip install {name}")
        libraries_ok = False


check("pandas")
check("numpy")
check("scikit-learn", "sklearn")
check("xgboost")
check("lightgbm")
check("shap")
check("torch")
check("torchvision")
check("librosa")
check("opencv-python", "cv2")
check("mediapipe")
check("imbalanced-learn", "imblearn")
check("streamlit")
check("plotly")
check("scipy")
check("matplotlib")
check("seaborn")
check("fpdf2", "fpdf")

print("\n📁 Checking Folder Structure...")
print("-" * 30)

# ─── STEP 3: Check folders exist ───
import os

folders = [
    "data",
    "notebooks",
    "models",
    "src",
    "dashboard",
    "reports",
    "outputs",
]

for f in folders:
    if os.path.exists(f):
        print(f" ✅ {f}/ folder exists")
    else:
        print(f" ❌ {f}/ folder MISSING — create it")
        os.makedirs(f, exist_ok=True)
        print(f" → Created {f}/ automatically")

print("\n📊 Checking Datasets...")
print("-" * 30)

# ─── STEP 4: Check datasets are in data/ folder ───
import pandas as pd

dataset_files = {
    "data/kaggle_ckd.csv": "Kaggle CKD Dataset",
    "data/uci_ckd.csv": "UCI CKD Dataset",
    "data/bangladesh_ckd.csv": "Bangladesh Lifestyle CKD",
    "data/nhanes_ckd.csv": "NHANES USA Dataset",
}

total_records = 0

for filepath, name in dataset_files.items():
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, nrows=5)
            full = pd.read_csv(filepath)
            records = len(full)
            total_records += records
            print(f" ✅ {name}: {records} records, {len(df.columns)} columns")
        except:
            print(f" ⚠️ {name}: File found but could not read — check format")
    else:
        print(f" ⚠️ {name}: NOT downloaded yet ({filepath})")

print(f"\n 📈 Total records available: {total_records:,}")

print("\n🧪 Quick Functionality Test...")
print("-" * 30)

# ─── STEP 5: Quick test of main libraries ───
try:
    import numpy as np
    import pandas as pd

    test_df = pd.DataFrame(
        {
            "age": [45, 32, 60],
            "bp": [120, 90, 140],
            "ckd": [1, 0, 1],
        }
    )

    print(" ✅ pandas: Can create DataFrames")
except:
    print(" ❌ pandas failed")

try:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=10)

    print(" ✅ scikit-learn: Can create ML models")
except:
    print(" ❌ scikit-learn failed")

try:
    import xgboost as xgb

    model = xgb.XGBClassifier(n_estimators=10)

    print(" ✅ XGBoost: Working")
except:
    print(" ❌ XGBoost failed")

try:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])

    print(f" ✅ PyTorch: Working (GPU: {torch.cuda.is_available()})")
except:
    print(" ❌ PyTorch failed")

try:
    import librosa
    import numpy as np

    dummy_audio = np.random.randn(22050)
    mfcc = librosa.feature.mfcc(y=dummy_audio, sr=22050, n_mfcc=13)

    print(f" ✅ librosa: MFCC extraction works — shape {mfcc.shape}")
except:
    print(" ❌ librosa failed")

try:
    import cv2

    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)

    print(" ✅ OpenCV: Image processing works")
except:
    print(" ❌ OpenCV failed")

try:
    import shap

    print(" ✅ SHAP: Explainability library ready")
except:
    print(" ❌ SHAP failed")

# ─── FINAL RESULT ───
print("\n" + "=" * 50)

if libraries_ok:
    print("🎉 ALL CHECKS PASSED! NephroSense is ready to build.")
    print(" Next step: Open notebooks/01_data_loading.ipynb")
else:
    print("⚠️ SOME LIBRARIES MISSING. Fix the ❌ items above.")
    print(" Run: pip install [library-name] for each missing one")

print("=" * 50)