# -------------------------- Group Poject -  Exercice 2 - MLP --------------------------
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import csv, json, random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1) Paths
# so it works on py or notebook
def get_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent  
    except NameError:
        return Path.cwd()                       

ROOT = get_project_root()
DATA_ROOT = ROOT / "Data" / "MNIST-full"       # contains gt-test/train.tsv + images
TRAIN_TSV = DATA_ROOT / "gt-train.tsv"
TEST_TSV  = DATA_ROOT / "gt-test.tsv"
IMG_ROOT  = DATA_ROOT

# in case TSVs are missing it fails fast
for p in (TRAIN_TSV, TEST_TSV):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}  (expected under {DATA_ROOT})")


# Reproductibility + device
seed = 1337
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2) Read train index
train_df = pd.read_csv(TRAIN_TSV, sep="\t", header=None, names=["rel_path","label"])

# 3) Materialization utils 
# We create missing PNGs on disk to match the TSV references, using torchvision MNIST
# Just for the splits we will load now
from collections import deque
from torchvision.datasets import MNIST

def materialize_split(tsv_df: pd.DataFrame, split: str, show_every: int = 500) -> None:

    ds = MNIST(root=ROOT/"_cache_mnist", train=(split=="train"), download=True)
    queues = {c: deque([i for i,t in enumerate(ds.targets) if int(t)==c]) for c in range(10)}

    total = sum(1 for rp in tsv_df["rel_path"] if str(rp).startswith(f"{split}/"))
    created = skipped = seen = 0
    print(f"[materialize {split}] need to check ~{total} files…")

    for rel_path, lbl in tsv_df[["rel_path","label"]].itertuples(index=False):
        if not str(rel_path).startswith(f"{split}/"):
            continue
        out = DATA_ROOT / rel_path
        if out.exists(): # if already exist, skip
            skipped += 1; seen += 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            idx = queues[int(lbl)].popleft()
            img, _ = ds[idx]
            img.save(out)  # PNG
            created += 1; seen += 1
        if seen % show_every == 0:
            print(f"  {seen}/{total} done  |  created={created}, skipped={skipped}")

    print(f"[materialize {split}] created={created}, skipped_existing={skipped}")

# 4) Train/val split
# Split onlly the training index
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
# We don't need features for stratification, pass dummy X of zeros
(train_idx, val_idx), = sss.split(np.zeros(len(train_df)), train_df["label"])

# subset DataFrames for train/val
train_sub = train_df.iloc[train_idx].reset_index(drop=True)
val_sub   = train_df.iloc[val_idx].reset_index(drop=True)

# Materialize only what we will use now (train + val)
materialize_split(train_sub, "train")
materialize_split(val_sub,   "train")

def df_to_lists(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    paths  = [(IMG_ROOT / rp).as_posix() for rp in df["rel_path"]]
    labels = df["label"].astype(int).tolist()
    return paths, labels

train_paths_final, train_labels_final = df_to_lists(train_sub)
val_paths,         val_labels         = df_to_lists(val_sub)

print("lens:", len(train_paths_final), len(val_paths))
print("sample train file exists?", Path(train_paths_final[0]).exists())

# 5) Load images -> numpy arrays
def load_numpy(paths: List[str], labels: List[int], resize_to=(28,28)) -> Tuple[np.ndarray, np.ndarray]:
    # Return (X, y) where X is [N, 784] float32 in [0,1]
    X = np.empty((len(paths), resize_to[0]*resize_to[1]), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L")
        if img.size != resize_to:
            img = img.resize(resize_to)
        arr = np.array(img, dtype=np.float32) / 255.0
        X[i] = arr.flatten()
    return X, y

# just train/val (test is for the end
X_train_np, y_train_np = load_numpy(train_paths_final, train_labels_final)
X_val_np,   y_val_np   = load_numpy(val_paths,          val_labels)
print("Shapes (raw):", X_train_np.shape, X_val_np.shape)

# Standardize to help MLP training (fit only on train)
scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train_np)
X_val   = scaler.transform(X_val_np)

# 6) Torch loaders
def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

BATCH = 256
train_loader = to_loader(X_train, y_train_np, BATCH, shuffle=True)
val_loader   = to_loader(X_val,   y_val_np,   BATCH, shuffle=False)

# 7) MLP (1 hidden layer)
# 1 hidden layer MLP for 28x28 grayscale digits
# variable in [10, 256] per exercise spec
# output : 10 classes (digits 0 to 9)
class MLP1H(nn.Module):
    def __init__(self, in_dim=28*28, hidden_size=128, dropout=0.0, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, optimizer=None):
    # one epoch of train
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0; total_acc = 0.0; total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, yb) * bs
        total_n    += bs
    return total_loss/total_n, total_acc/total_n

def train_model(hp, epochs=15, verbose=True):
    model = MLP1H(hidden_size=hp["hidden_size"], dropout=hp.get("dropout",0.0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay",0.0))
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0; best_state = None
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer=opt)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer=None)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)
        if verbose:
            print(f"[{ep:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_val

def plot_curves(history_df: pd.DataFrame, title: str, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"],   label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(title+" — loss")
    plt.tight_layout(); plt.savefig(out_prefix.with_name(out_prefix.name+"_loss.png")); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    plt.plot(history_df["epoch"], history_df["val_acc"],   label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title(title+" — accuracy")
    plt.tight_layout(); plt.savefig(out_prefix.with_name(out_prefix.name+"_acc.png")); plt.close()

# Small hyperparameter grid 
# hidden_size {64, 128, 256} in the range [10, 256]
# Learning rate {1e-4, 5e-4, 1e-3} in the range [0.001, 0.1]
grid = [
    {"hidden_size": 64,  "lr": 1e-3, "dropout": 0.10, "weight_decay": 0.0},
    {"hidden_size": 128, "lr": 1e-3, "dropout": 0.10, "weight_decay": 0.0},
    {"hidden_size": 256, "lr": 1e-3, "dropout": 0.20, "weight_decay": 0.0},
    {"hidden_size": 128, "lr": 5e-4, "dropout": 0.10, "weight_decay": 0.0},
    {"hidden_size": 128, "lr": 1e-4, "dropout": 0.20, "weight_decay": 1e-4},
]

results = []
best = {"val_acc": -1, "hparams": None, "model": None, "history": None}
for i, hp in enumerate(grid, 1):
    print(f"\n=== Combo {i}/{len(grid)}: {hp} ===")
    model, hist, val_acc = train_model(hp, epochs=15, verbose=True)
    # save curves per HP combo i
    plot_curves(hist, title=f"MLP {hp}", out_prefix=(ROOT / "results" / f"mlp_{i}"))
    results.append({**hp, "val_acc": float(val_acc)})
    if val_acc > best["val_acc"]:
        best = {"val_acc": float(val_acc), "hparams": hp, "model": model, "history": hist}

resultat_df = pd.DataFrame(results).sort_values("val_acc", ascending=False)
print("\nGrid results:\n", resultat_df)
( ROOT / "results" ).mkdir(parents=True, exist_ok=True)
resultat_df.to_csv(ROOT / "results" / "grid_results.csv", index=False)
print("\nBest HP:", best["hparams"], "| best val acc:", f"{best['val_acc']:.4f}")

# 9) Final training on train+val
# Refit scaler on (train+val) before final test
X_full_np = np.vstack([X_train_np, X_val_np])
y_full_np = np.concatenate([y_train_np, y_val_np])

final_scaler = StandardScaler(with_mean=True, with_std=True).fit(X_full_np)
X_full = final_scaler.transform(X_full_np)
full_loader = to_loader(X_full, y_full_np, BATCH, shuffle=True)

hp = best["hparams"]
final_model = MLP1H(hidden_size=hp["hidden_size"], dropout=hp.get("dropout",0.0)).to(device)
final_opt   = torch.optim.Adam(final_model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay",0.0))

# track final training curves
EPOCHS_FINAL = 20
final_hist = {"epoch": [], "train_loss": [], "train_acc": []}
for ep in range(1, EPOCHS_FINAL+1):
    tr_loss, tr_acc = run_epoch(final_model, full_loader, optimizer=final_opt)
    final_hist["epoch"].append(ep); final_hist["train_loss"].append(tr_loss); final_hist["train_acc"].append(tr_acc)
    print(f"[final {ep:03d}] loss {tr_loss:.4f} acc {tr_acc:.4f}")

final_hist_df = pd.DataFrame(final_hist)
final_hist_df.to_csv(ROOT / "results" / "final_train_history.csv", index=False)
plt.figure(figsize=(6,4)); plt.plot(final_hist_df["epoch"], final_hist_df["train_loss"]); plt.xlabel("epoch"); plt.ylabel("loss")
plt.tight_layout(); plt.savefig(ROOT / "results" / "final_train_loss.png"); plt.close()
plt.figure(figsize=(6,4)); plt.plot(final_hist_df["epoch"], final_hist_df["train_acc"]); plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.tight_layout(); plt.savefig(ROOT / "results" / "final_train_acc.png"); plt.close()

# 10) TEST: read & materialize now, then evaluate
# read the test index and materialize its images
test_df = pd.read_csv(TEST_TSV, sep="\t", header=None, names=["rel_path","label"])
missing_test = [p for p in test_df.rel_path if not (DATA_ROOT/p).exists()]
if missing_test:
    materialize_split(test_df, "test")

def df_to_lists(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    paths  = [(IMG_ROOT / rp).as_posix() for rp in df["rel_path"]]
    labels = df["label"].astype(int).tolist()
    return paths, labels

test_paths, test_labels = df_to_lists(test_df)
X_test_np, y_test_np    = load_numpy(test_paths, test_labels)
X_test = final_scaler.transform(X_test_np)
test_loader = to_loader(X_test, y_test_np, BATCH, shuffle=False)
# Final test evaluation
test_loss, test_acc = run_epoch(final_model, test_loader, optimizer=None)
print(f"\n TEST accuracy: {test_acc:.4f}")

# Save artifacts 
OUT = ROOT / "results"
torch.save(best["model"].state_dict(), OUT / "best_on_val_mlp.pth")
torch.save(final_model.state_dict(),   OUT / "final_mlp_trainval.pth")
with open(OUT / "best_hparams.json", "w", encoding="utf-8") as f:
    json.dump({"best_val_acc": best["val_acc"], "hparams": best["hparams"]}, f, indent=2)
print("Artifacts saved in:", OUT)
