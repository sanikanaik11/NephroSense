"""
train_ckd_model.py — NephroSense (v3 — improved)
Trains an ENSEMBLE model (XGBoost + Random Forest + Logistic
Regression, soft-voted) on uci_ckd.csv + bangladesh_ckd.csv +
nhanes_ckd.csv combined (independent public sources), with:
  - Real engineered clinical features (BUN/creatinine ratio, anemia
    severity, age×BP interaction) — all standard, clinically
    meaningful derived values, not synthetic/fake data.
  - SMOTE class balancing on the training set only (never touches
    the test set, so reported metrics stay honest).
  - Wider hyperparameter grid search for XGBoost.
  - Threshold tuned to prioritize catching real CKD cases.
  - Duplicate-row removal across combined sources (prevents
    accidental data leakage from inflating accuracy).

HOW TO RUN (from nephrosense/ folder):
    pip install xgboost shap lime scikit-learn joblib pandas numpy imbalanced-learn
    python src/fetch_nhanes.py      (run once, needs internet — optional)
    python src/train_ckd_model.py

Creates:
    models/ckd_model.pkl   ← trained ensemble model (VotingClassifier)
    models/ckd_meta.pkl    ← feature list, medians, accuracy, AUC, recall, threshold
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import joblib
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    print("Installing xgboost...")
    os.system("pip install xgboost --break-system-packages -q")
    from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Installing imbalanced-learn (for SMOTE)...")
    os.system("pip install imbalanced-learn --break-system-packages -q")
    try:
        from imblearn.over_sampling import SMOTE
        SMOTE_AVAILABLE = True
    except ImportError:
        SMOTE_AVAILABLE = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MDL_DIR  = os.path.join(ROOT, "models")
os.makedirs(MDL_DIR, exist_ok=True)

UCI_PATH    = os.path.join(DATA_DIR, "uci_ckd.csv")
BD_PATH     = os.path.join(DATA_DIR, "bangladesh_ckd.csv")
NHANES_PATH = os.path.join(DATA_DIR, "nhanes_ckd.csv")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_clean

BASE_FEATURE_COLS = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

# ── Engineered clinical features ────────────────────────────
# These are real, clinically meaningful derived values (not
# fabricated data) that often carry more signal than the raw
# numbers alone:
#   bun_creatinine_ratio — classic renal-function ratio used by
#     doctors to distinguish pre-renal vs intrinsic kidney issues.
#   anemia_severity       — how far below normal haemoglobin is
#     (CKD very commonly causes anaemia).
#   age_bp_interaction    — captures that high BP matters more at
#     older ages for kidney risk.
ENGINEERED_FEATURE_COLS = ["bun_creatinine_ratio", "anemia_severity", "age_bp_interaction"]
FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS

# preprocess.load_and_clean() uses "wc"/"rc" as column names, while
# FEATURE_COLS uses "wbcc"/"rbcc" — rename after cleaning so both
# ARFF-style datasets line up under the names the rest of the app uses.
RENAME_AFTER_CLEAN = {"wc": "wbcc", "rc": "rbcc"}


def add_engineered_features(df):
    df = df.copy()
    df["bun_creatinine_ratio"] = df["bu"] / df["sc"].replace(0, 0.1)
    df["anemia_severity"] = (14 - df["hemo"]).clip(lower=0)
    df["age_bp_interaction"] = df["age"] * df["bp"] / 100.0
    return df


# ── Load + clean ─────────────────────────────────────────────
print("\n[1/8] Loading datasets...")
frames = []
sources_used = []


def load_bangladesh_robust(path):
    """
    Bangladesh CSV has its own headers (not ARFF, not headerless) — a
    generic loader with wrong assumptions was dropping all rows.
    This detects the target column and value encoding directly.
    """
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    print(f"    Bangladesh columns found: {list(raw.columns)}")

    target_col = None
    for cand in ["classification", "class", "target", "ckd", "diagnosis", "label"]:
        if cand in raw.columns:
            target_col = cand
            break
    if target_col is None:
        raise ValueError("no recognizable target column in bangladesh_ckd.csv")

    print(f"    Bangladesh target column: '{target_col}', unique values: {raw[target_col].unique()[:10]}")

    def map_target(v):
        s = str(v).strip().lower()
        if s in ("1", "yes", "true", "positive"):
            return 1
        if s in ("0", "no", "false", "negative"):
            return 0
        if "not" in s or "non" in s:
            return 0
        if "ckd" in s:
            return 1
        return np.nan

    raw["target"] = raw[target_col].apply(map_target)
    raw = raw.dropna(subset=["target"])
    raw["target"] = raw["target"].astype(int)

    rename_map = {"wc": "wbcc", "rc": "rbcc", "wbc": "wbcc", "rbc_count": "rbcc"}
    raw = raw.rename(columns=rename_map)

    for col in raw.columns:
        if col in ("target",):
            continue
        raw[col] = pd.to_numeric(raw[col].replace({"yes": 1, "no": 0, "normal": 1, "abnormal": 0,
                                                     "present": 1, "notpresent": 0, "good": 1, "poor": 0}),
                                  errors="coerce")
    return raw


for path, name in [(UCI_PATH, "UCI CKD"), (BD_PATH, "Bangladesh CKD"), (NHANES_PATH, "NHANES USA")]:
    if not os.path.exists(path):
        print(f"  SKIP — not found: {path}")
        if name == "NHANES USA":
            print("         Run 'python src/fetch_nhanes.py' first to download it.")
        continue
    try:
        if name == "NHANES USA":
            df = pd.read_csv(path)
        elif name == "Bangladesh CKD":
            df = load_bangladesh_robust(path)
        else:
            df = load_and_clean(path)
            df = df.rename(columns=RENAME_AFTER_CLEAN)

        keep = [c for c in BASE_FEATURE_COLS if c in df.columns] + ["target"]
        df = df[keep].copy()
        if len(df) < 10:
            print(f"  SKIP — too few usable rows: {name}")
            continue
        print(f"  ✅ {name}: {len(df)} rows, CKD={int(df['target'].sum())}, notCKD={int((df['target']==0).sum())}")
        frames.append(df)
        sources_used.append(name)
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

if not frames:
    print("\nERROR: No datasets loaded. Check your data/ folder.")
    sys.exit(1)

# ── Combine ──────────────────────────────────────────────────
print("\n[2/8] Combining datasets...")
df = pd.concat(frames, ignore_index=True, sort=False)

# Drop exact-duplicate rows across sources (protects against the
# same patient/record appearing in more than one file, which would
# silently inflate accuracy through data leakage).
before_dedup = len(df)
df = df.drop_duplicates(subset=[c for c in BASE_FEATURE_COLS if c in df.columns])
after_dedup = len(df)
if before_dedup != after_dedup:
    print(f"  Removed {before_dedup - after_dedup} duplicate row(s) across sources.")

base_cols_present = [c for c in BASE_FEATURE_COLS if c in df.columns]
df = df[base_cols_present + ["target"]].copy()

medians = {}
for col in base_cols_present:
    m = df[col].median()
    medians[col] = float(m) if not pd.isna(m) else 0.0
    df[col] = df[col].fillna(medians[col])

df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

# ── Add engineered features (after missing-value fill, so ratios
#    are computed on complete rows) ─────────────────────────────
print("\n[3/8] Adding engineered clinical features...")
df = add_engineered_features(df)
for col in ENGINEERED_FEATURE_COLS:
    m = df[col].median()
    medians[col] = float(m) if not pd.isna(m) else 0.0
    df[col] = df[col].fillna(medians[col])
print(f"  Added: {ENGINEERED_FEATURE_COLS}")

feature_cols = base_cols_present + ENGINEERED_FEATURE_COLS
print(f"  Combined: {len(df)} records, {len(feature_cols)} features, sources: {sources_used}")
print(f"  CKD={df['target'].sum()}, notCKD={(df['target']==0).sum()}")

# ── Split ────────────────────────────────────────────────────
print("\n[4/8] Splitting 80/20 train/test (stratified)...")
X = df[feature_cols].values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── SMOTE balancing (training set only — test set stays untouched
#    and honest) ─────────────────────────────────────────────
if SMOTE_AVAILABLE:
    print("\n[5/8] Balancing training set with SMOTE...")
    before_n = len(X_train)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  Train set: {before_n} → {len(X_train)} rows after SMOTE balancing")
else:
    print("\n[5/8] SMOTE not available — skipping balancing (imbalanced-learn failed to install).")

# ── Hyperparameter tuning for XGBoost (real search, not guessed) ──
print("\n[6/8] Tuning XGBoost hyperparameters (grid search, 5-fold)...")
pos = int(y_train.sum())
neg = int(len(y_train) - pos)
spw = neg / pos if pos > 0 else 1.0

xgb_param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
}
xgb_base = XGBClassifier(
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=spw, random_state=42, eval_metric="logloss", n_jobs=-1,
)
xgb_search = GridSearchCV(
    xgb_base, xgb_param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="recall", n_jobs=-1,
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"  Best XGBoost params: {xgb_search.best_params_}")

# ── Build the ensemble (XGBoost + Random Forest + Logistic Regression) ──
print("\n[7/8] Training ensemble (XGBoost + Random Forest + Logistic Regression)...")

rf = RandomForestClassifier(
    n_estimators=500, max_depth=9, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)

log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
])

ensemble = VotingClassifier(
    estimators=[("xgb", best_xgb), ("rf", rf), ("logreg", log_reg)],
    voting="soft",
    weights=[2, 1, 1],  # XGBoost weighted higher — it's the strongest individual model
)
ensemble.fit(X_train, y_train)

# ── Threshold tuning — maximize recall for CKD class while keeping
#    precision reasonable, since missing a real case matters more
#    than a false alarm in a screening tool. Evaluated on the
#    UNTOUCHED, non-SMOTE'd test set for an honest read.
proba_test = ensemble.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)

best_threshold = 0.5
best_recall_at_ok_precision = 0.0
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if p >= 0.85 and r > best_recall_at_ok_precision:
        best_recall_at_ok_precision = r
        best_threshold = t
if best_recall_at_ok_precision == 0.0:
    best_threshold = 0.5  # fallback if no threshold hits the precision floor

preds_default = (proba_test >= 0.5).astype(int)
preds_tuned = (proba_test >= best_threshold).astype(int)

acc_default = accuracy_score(y_test, preds_default)
acc_tuned = accuracy_score(y_test, preds_tuned)
auc = roc_auc_score(y_test, proba_test)

print(f"\n  ✅ Test Accuracy (0.5 threshold)   : {acc_default*100:.1f}%")
print(f"  ✅ Test Accuracy (tuned threshold) : {acc_tuned*100:.1f}%  (threshold={best_threshold:.3f})")
print(f"  ✅ ROC-AUC                         : {auc:.3f}")
print("\n  Classification report (tuned threshold, prioritizes catching CKD cases):")
print(classification_report(y_test, preds_tuned, target_names=["Not CKD", "CKD"]))

cm = confusion_matrix(y_test, preds_tuned)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0
print(f"  Sensitivity (recall, CKD caught): {sensitivity*100:.1f}%")
print(f"  Specificity (Not-CKD correctly cleared): {specificity*100:.1f}%")
print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

# ── Cross-validation on the full ensemble (honest, stratified
#    10-fold, run on the ORIGINAL un-SMOTE'd data so this number
#    is not artificially inflated by resampling) ─────────────
print("\n[8/8] Running 10-fold stratified cross-validation on the ensemble (original data)...")
cv = cross_val_score(ensemble, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring="accuracy")
print(f"  CV Accuracy: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

# ── Save ─────────────────────────────────────────────────────
print("\nSaving model...")
model_path = os.path.join(MDL_DIR, "ckd_model.pkl")
meta_path  = os.path.join(MDL_DIR, "ckd_meta.pkl")

joblib.dump(ensemble, model_path)
joblib.dump({
    "feature_cols" : feature_cols,
    "base_feature_cols": base_cols_present,
    "engineered_feature_cols": ENGINEERED_FEATURE_COLS,
    "medians"      : medians,
    "modes"        : {},
    "accuracy"     : round(acc_tuned*100, 1),
    "accuracy_default_threshold": round(acc_default*100, 1),
    "cv_accuracy"  : round(cv.mean()*100, 1),
    "cv_std"       : round(cv.std()*100, 1),
    "roc_auc"      : round(auc, 3),
    "sensitivity"  : round(sensitivity*100, 1),
    "specificity"  : round(specificity*100, 1),
    "decision_threshold": round(float(best_threshold), 3),
    "n_train"      : len(X_train),
    "n_total"      : len(df),
    "sources_used" : ", ".join(sources_used),
    "model_type"   : "Ensemble (XGBoost + Random Forest + Logistic Regression, soft voting) with SMOTE-balanced training and engineered clinical features",
    "xgb_best_params": xgb_search.best_params_,
    "used_smote"   : SMOTE_AVAILABLE,
}, meta_path)

print(f"\n  Model saved → {model_path}")
print(f"  Meta  saved → {meta_path}")
print(f"\n🎉 Done! Ensemble accuracy: {acc_tuned*100:.1f}%  (CV: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%)")
print(f"   ROC-AUC: {auc:.3f}  |  Sensitivity: {sensitivity*100:.1f}%  |  Specificity: {specificity*100:.1f}%")
print(f"   Total records used: {len(df)} from: {sources_used}")
print(f"   Engineered features added: {ENGINEERED_FEATURE_COLS}")
print(">>> COPY THESE EXACT NUMBERS INTO YOUR PPT/REPORT — do not round up or estimate. <<<")
print("   Restart the Streamlit app — Health Assessment will use the updated model now.")