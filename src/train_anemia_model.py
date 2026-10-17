"""
train_anemia_model.py — NephroSense
Trains a real anemia-risk model on conjunctiva photos from THREE
sources (India + Italy Excel with real Hgb values, Ghana/CP-AnemiC
folder-labeled images), using EfficientNet-B3 to match the
architecture stated in the project synopsis/PPT.

Usage:
    pip install torch torchvision pandas pillow openpyxl
    python src/train_anemia_model.py
"""
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

DATA_ROOT = "data/dataset anemia"
EXCEL_COUNTRIES = ["India", "Italy"]
IMG_SIZE = 300  # EfficientNet-B3 native input size
BATCH_SIZE = 16
EPOCHS = 15
MODEL_OUT = "models/anemia_model.pt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_hgb(hgb):
    try:
        return float(str(hgb).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


def who_anemic(hgb, gender):
    gender = str(gender).strip().lower()
    if gender.startswith("m"):
        return 1 if hgb < 13.0 else 0
    return 1 if hgb < 12.0 else 0


def find_palpebral_image(folder):
    matches = glob.glob(os.path.join(folder, "*palpebral*"))
    return matches[0] if matches else None


def image_is_valid(path):
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def build_excel_rows():
    rows = []
    skipped_no_excel = skipped_no_folder = skipped_no_image = 0
    skipped_bad_note = skipped_bad_hgb = skipped_missing_fields = 0
    skipped_corrupt_image = 0

    for country in EXCEL_COUNTRIES:
        xlsx_path = os.path.join(DATA_ROOT, country, f"{country}.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"WARNING: {xlsx_path} not found, skipping {country}.")
            skipped_no_excel += 1
            continue

        df = pd.read_excel(xlsx_path)
        df.columns = [str(c).strip() for c in df.columns]

        for _, row in df.iterrows():
            number = row.get("Number")
            hgb_raw = row.get("Hgb")
            gender = row.get("Gender")
            note = str(row.get("Note", "")).lower()

            if pd.isna(number) or pd.isna(hgb_raw) or pd.isna(gender):
                skipped_missing_fields += 1
                continue
            if "not visible" in note or "poor" in note:
                skipped_bad_note += 1
                continue

            hgb = parse_hgb(hgb_raw)
            if hgb is None:
                skipped_bad_hgb += 1
                continue

            folder = os.path.join(DATA_ROOT, country, str(int(number)))
            if not os.path.isdir(folder):
                skipped_no_folder += 1
                continue

            img_path = find_palpebral_image(folder)
            if img_path is None:
                skipped_no_image += 1
                continue

            if not image_is_valid(img_path):
                skipped_corrupt_image += 1
                continue

            rows.append({"path": img_path, "label": who_anemic(hgb, gender), "source": country})

    print(f"  India+Italy: {len(rows)} usable images")
    print(f"    skipped - no excel: {skipped_no_excel}, missing fields: {skipped_missing_fields}, "
          f"bad hgb: {skipped_bad_hgb}, no folder: {skipped_no_folder}, no image: {skipped_no_image}, "
          f"bad note: {skipped_bad_note}, corrupt: {skipped_corrupt_image}")
    return rows


def build_ghana_rows():
    rows = []
    skipped_corrupt = 0
    for label, subfolder in ((1, "anemia"), (0, "normal")):
        folder = os.path.join(DATA_ROOT, "Ghana", subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                continue
            fpath = os.path.join(folder, fname)
            if not image_is_valid(fpath):
                skipped_corrupt += 1
                continue
            rows.append({"path": fpath, "label": label, "source": "Ghana"})
    print(f"  Ghana: {len(rows)} usable images (skipped corrupt: {skipped_corrupt})")
    return rows


def build_file_list():
    print("Scanning all sources...")
    rows = build_excel_rows() + build_ghana_rows()
    df = pd.DataFrame(rows)
    if len(df):
        print(f"\nCombined total: {len(df)} images from sources: {df['source'].unique().tolist()}")
    return df


class AnemiaDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        if train:
            self.tf = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        img = self.tf(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label


def build_efficientnet_b3():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def main():
    df = build_file_list()
    if len(df) < 20:
        print("\nNot enough usable images found (need at least ~20).")
        return

    print(f"\nAnemia class balance:\n{df['label'].value_counts()}")

    full_ds = AnemiaDataset(df, train=True)
    n_val = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    val_ds.dataset = AnemiaDataset(df, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining EfficientNet-B3 on: {device}")

    model = build_efficientnet_b3().to(device)

    n_pos = int(df["label"].sum())
    n_neg = int(len(df) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Class imbalance detected ({n_pos} anemic vs {n_neg} normal) — applying pos_weight={pos_weight.item():.2f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs).squeeze(1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = torch.sigmoid(model(imgs).squeeze(1))
                preds = (out > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"\nBest validation accuracy (EfficientNet-B3): {best_val_acc:.3f}")
    print(f"Total training images used: {len(df)}")
    print(f"Saved best model to: {MODEL_OUT}")
    print("\n>>> COPY THIS NUMBER INTO YOUR PPT — do not use any other figure. <<<")


if __name__ == "__main__":
    main()