"""
calibrate_eye_colour.py — NephroSense  (NEW FILE -> put in src/)

Makes the Visual Screening colour score data-driven and measures how well it works.
Run ONCE from the project folder (same place you run train_anemia_model.py):

    python src/calibrate_eye_colour.py

It measures the colour of the inner-eyelid lining (Lab a*, R/G) in every Hb-labelled training photo
(India, Italy, Ghana) with the SAME code the app uses, fits a small logistic model, prints honest
cross-validated results (accuracy, AUC, sensitivity, specificity, per country) and saves
models/eye_colour_cal.json. The app then uses it automatically. Quote these printed numbers — and no others —
when you describe the Visual Screening feature.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import eye_model as em


def extract_features(path):
    """(a_star, r_over_g) of the lining, or None if the photo could not be used."""
    try:
        img = em._open_image(open(path, "rb").read())
        r = em.assess_photo(img)
        if not r["ok"]:
            return None
        return float(r["colour"]["a"]), float(r["colour"]["rg"])
    except Exception:
        return None


def fit_calibration(X, y):
    """Logistic fit on standardised features, returned as raw-unit weights: z = w_a*a + w_rg*rg + b."""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit((X - mu) / sd, y)
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - np.sum(clf.coef_[0] * mu / sd))
    return {"w_a": float(w[0]), "w_rg": float(w[1]), "b": b}


def evaluate(X, y, sources=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    p = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    out = {"auc": float(roc_auc_score(y, p)), "accuracy": float((pred == y).mean()),
           "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1))}
    if sources is not None:
        sources = np.asarray(sources)
        out["by_source"] = {}
        for s in sorted(set(sources)):
            m = sources == s
            if len(set(y[m])) == 2:
                out["by_source"][s] = round(float(roc_auc_score(y[m], p[m])), 3)
    return out


def main():
    from train_anemia_model import build_file_list
    df = build_file_list()
    if len(df) < 30:
        print("Not enough Hb-labelled photos found (run from the project folder).")
        return
    feats, labels, srcs, skipped = [], [], [], 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        f = extract_features(row.path)
        if f is None:
            skipped += 1
            continue
        feats.append(f)
        labels.append(int(row.label))
        srcs.append(row.source)
        if i % 100 == 0:
            print(f"  measured {i}/{len(df)} photos...")
    print(f"\nUsable photos: {len(feats)}   skipped (lining not found / too blurry): {skipped}")
    if len(set(labels)) < 2 or len(feats) < 30:
        print("Not enough usable photos of both groups to calibrate.")
        return
    res = evaluate(feats, labels, srcs)
    cal = fit_calibration(feats, labels)
    cal.update({"n": len(feats), "auc_cv": round(res["auc"], 3), "accuracy_cv": round(res["accuracy"], 3),
                "sensitivity_cv": round(res["sensitivity"], 3), "specificity_cv": round(res["specificity"], 3)})
    out = os.path.join(os.path.dirname(em.MODEL_PATH), "eye_colour_cal.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(cal, open(out, "w"), indent=2)
    print("\n5-fold cross-validated results of the COLOUR score (honest, not trained-on):")
    print(f"  AUC {res['auc']:.3f} | accuracy {res['accuracy']*100:.1f}% | sensitivity {res['sensitivity']*100:.1f}% "
          f"| specificity {res['specificity']*100:.1f}%")
    print("  AUC per country:", res.get("by_source"))
    print(f"\nSaved calibration -> {out}\nRestart the app to use it.")
    print(">>> Use THESE numbers in your PPT/report for the colour-based Visual Screening. <<<")


if __name__ == "__main__":
    main()