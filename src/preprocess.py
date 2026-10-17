import numpy as np
import pandas as pd
from io import StringIO

COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "classification",
]

NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc",
]
CATEGORICAL_MAP = {
    "rbc": {"normal": 1, "abnormal": 0},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
}


def _read_text_robust(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


def _clean_text(x):
    if isinstance(x, str):
        x = x.strip().lower().replace("\t", "").replace("?", "")
        x = x.strip("'").strip('"').strip()
        if x in ("", "nan"):
            return np.nan
        return x
    return x


def _read_arff_as_df(text, path):
    lines = text.splitlines()
    data_lines = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        low = stripped.lower()
        if low.startswith("@data"):
            in_data = True
            continue
        if stripped.startswith("@"):
            continue
        if in_data:
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"'{path}' has no data rows after @data.")

    csv_text = "\n".join(data_lines)

    try:
        df = pd.read_csv(StringIO(csv_text), header=None, names=COLUMNS)
    except pd.errors.ParserError:
        df = pd.read_csv(
            StringIO(csv_text),
            header=None,
            names=COLUMNS,
            engine="python",
            on_bad_lines="skip",
        )
        skipped = len(data_lines) - len(df)
        if skipped > 0:
            print(f"       Note: skipped {skipped} malformed row(s).")

    return df


def load_and_clean(path):
    text = _read_text_robust(path)

    if "@data" in text.lower() and ("@attribute" in text.lower() or "@relation" in text.lower()):
        df = _read_arff_as_df(text, path)
    else:
        try:
            df = pd.read_csv(StringIO(text))
        except pd.errors.ParserError as e:
            raise ValueError(f"'{path}' could not be parsed. Original error: {e}") from e

        if list(df.columns[:3]) not in ([["age", "bp", "sg"]], [COLUMNS[:3]]):
            first_col_name = str(df.columns[0]).strip().lower()
            if first_col_name not in ("age", "id"):
                df = pd.read_csv(StringIO(text), header=None, names=COLUMNS)

    if "id" in [c.lower() for c in df.columns]:
        id_col = [c for c in df.columns if c.lower() == "id"][0]
        df = df.drop(columns=[id_col])

    df.columns = [str(c).strip().lower() for c in df.columns]

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"'{path}' has {df.shape[1]} columns, expected {len(COLUMNS)}: {COLUMNS}"
        )

    for col in df.columns:
        df[col] = df[col].apply(_clean_text)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, mapping in CATEGORICAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    target_col = "classification" if "classification" in df.columns else "class"
    df[target_col] = df[target_col].apply(_clean_text)
    df[target_col] = df[target_col].map({"ckd": 1, "notckd": 0})
    df = df.rename(columns={target_col: "target"})

    df = df.dropna(subset=["target"])

    if len(df) == 0:
        raise ValueError(f"'{path}' parsed but 0 usable rows remain.")

    return df


def fill_missing(df, medians=None, modes=None):
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "target"]

    if medians is None:
        medians = {c: df[c].median() for c in NUMERIC_COLS if c in df.columns}
    if modes is None:
        modes = {
            c: df[c].mode().iloc[0] if not df[c].mode().empty else 0
            for c in feature_cols
            if c not in NUMERIC_COLS
        }

    for c, v in medians.items():
        df[c] = df[c].fillna(v)
    for c, v in modes.items():
        df[c] = df[c].fillna(v)

    return df, medians, modes