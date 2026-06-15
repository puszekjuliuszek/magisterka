# === MARKDOWN CELL 1 ===
# # Approach 13: ECGFounder as an Interpretable Clinical Proxy
#
# This notebook loads the pre-trained ECGFounder model (trained on 10M+ ECGs with 150 clinical labels) and uses it as a feature extractor.
#     The 150-dimensional probability vector for each ECG is extracted, and an interpretable model (XGBoost and Logistic Regression) is
#     trained to predict Chagas disease. Feature importance is analyzed via SHAP.
#
# Evaluation setup: the same 3-way CODE-15% split as approaches 4-10 is used:
# - Train / Val / Test: all drawn from CODE-15% (`train_test_split`, `test_size=0.15` then `test_size=0.18`, stratified on Chagas label,
#     `random_state=42`)
# - OOD (SaMi-Trop): standalone domain-shift evaluation, sensitivity only, since SaMi-Trop is 100% Chagas+
#
# This enables direct metric comparison with ResNet/CNN approaches on a common held-out CODE-15% test set.

# === CODE CELL 2 ===
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    f1_score, roc_curve, precision_recall_curve,
)
import xgboost as xgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from tqdm.auto import tqdm
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sns.set_context("notebook")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 5)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chagas")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"PyTorch {torch.__version__} | Device : {device}")


# === CODE CELL 3 ===
CFG = {
    "preprocessed_cache": "preprocessed_cache_brazil.h5",
    "val_fraction": 0.15,
    "test_fraction": 0.15, # Note on OOD: when source=="samitrop" is used for test, this test_fraction is ignored.
    "random_seed": 42,
    "target_fs": 500,
    "target_duration_s": 10,
    
    # ECGFounder
    "ecgfounder_path": "ECGFounder",
    "weights_path": "ECGFounder/weights/12_lead_ECGFounder.pth",
    "n_classes_pretrain": 150,
    
    "batch_size": 8,
    "num_workers": 0,
}


# === CODE CELL 4 ===
sys.path.append(os.path.abspath(CFG["ecgfounder_path"]))
from net1d import Net1D

def load_ecgfounder(weights_path, device, n_classes=150):
    model = Net1D(
        in_channels=12, 
        base_filters=64,
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    )
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"Missing keys: {missing}, Unexpected: {unexpected}")
        assert len(missing) == 0 and len(unexpected) == 0, "Model architecture mismatch!"
    model.eval()
    model.to(device)
    return model

logger.info(f"Loading ECGFounder from {CFG['weights_path']}")
model = load_ecgfounder(CFG["weights_path"], device, CFG["n_classes_pretrain"])


# === CODE CELL 5 ===
cache_file = CFG["preprocessed_cache"]
assert Path(cache_file).exists(), f"Missing {cache_file} -- run chagas_resnet_classifier.ipynb first."

with h5py.File(cache_file, "r") as f:
    n_total = f["labels"].shape[0]
    all_labels = f["labels"][:]
    all_exam_ids = f["exam_ids"][:] if "exam_ids" in f else np.arange(n_total)
    print(f"Samples: {n_total:,} | Chagas+ {100*all_labels.mean():.1f}%")

code15_dir = "4916206"
samitrop_dir = "sami-trop"

df_code15 = pd.read_csv(os.path.join(code15_dir, "exams.csv")) if os.path.exists(os.path.join(code15_dir, "exams.csv")) else pd.DataFrame()
df_samitrop = pd.read_csv(os.path.join(samitrop_dir, "exams.csv")) if os.path.exists(os.path.join(samitrop_dir,
    "exams.csv")) else pd.DataFrame()

if not df_code15.empty: df_code15["source"] = "code15"
if not df_samitrop.empty: df_samitrop["source"] = "samitrop"

df_meta = pd.concat([df_code15, df_samitrop], ignore_index=True)

assert not df_meta.empty and "exam_id" in df_meta.columns, "Could not load patient metadata."
exam_to_source = dict(zip(df_meta["exam_id"], df_meta["source"]))
sources = np.array([exam_to_source.get(eid, "unknown") for eid in all_exam_ids])

# SaMi-Trop: OOD only (100% positive, used for sensitivity evaluation, not official test)
samitrop_indices = np.where(sources == "samitrop")[0]

# CODE-15% only: 3-way split matching approaches 4-10
from sklearn.model_selection import train_test_split

code15_indices = np.where(sources == "code15")[0]
code15_labels  = all_labels[code15_indices]

code15_trainval_idx, test_indices = train_test_split(
    code15_indices,
    test_size=0.15,
    stratify=code15_labels,
    random_state=42,
)

# 0.18 * 0.85 ~= 15% of total
code15_trainval_labels = all_labels[code15_trainval_idx]

train_indices, val_indices = train_test_split(
    code15_trainval_idx,
    test_size=0.18,
    stratify=code15_trainval_labels,
    random_state=42,
)

print(f"\n{'='*70}")
print(f"  DATASET SPLIT SUMMARY (Train / Val / Test / OOD)")
print(f"{'='*70}")
print(f"  Training    : {len(train_indices):>7,} samples  "
      f"(Chagas+ {int(all_labels[train_indices].sum()):>5,}, {100*all_labels[train_indices].mean():.1f}%)")
print(f"  Validation  : {len(val_indices):>7,} samples  "
      f"(Chagas+ {int(all_labels[val_indices].sum()):>5,}, {100*all_labels[val_indices].mean():.1f}%)")
print(f"  Test (CODE) : {len(test_indices):>7,} samples  "
      f"(Chagas+ {int(all_labels[test_indices].sum()):>5,}, {100*all_labels[test_indices].mean():.1f}%)")
print(f"  OOD SaMiTrop: {len(samitrop_indices):>7,} samples  "
      f"(Chagas+ {int(all_labels[samitrop_indices].sum()):>5,}, 100.0%)")
print(f"{'='*70}")

class CachedChagasDataset(Dataset):
    def __init__(self, cache_path, indices):
        self.cache_path = cache_path
        self.indices = np.sort(indices)
        self._file = None
        with h5py.File(cache_path, "r") as f:
            self.labels = f["labels"][self.indices]

    def __len__(self):
        return len(self.indices)

    def _f(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, "r")
        return self._file

    def __getitem__(self, idx):
        f = self._f()
        ri = self.indices[idx]
        x = torch.from_numpy(f["signals"][ri]).float()
        
        # Standard clinical cache has leads: I, II, III, aVR, aVF, aVL, V1-V6.
        # ECGFounder expects: I, II, III, aVR, aVL, aVF, V1-V6.
        # Swap lead index 4 (aVF) and 5 (aVL).
        x = x[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11], :]
        
        # Z-score normalization (matching ECGFounder pre-training)
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / (x_std + 1e-8)
        
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

pin = device.type == "cuda"
train_loader = DataLoader(CachedChagasDataset(cache_file, train_indices), batch_size=CFG["batch_size"], shuffle=False, pin_memory=pin)
val_loader   = DataLoader(CachedChagasDataset(cache_file, val_indices),   batch_size=CFG["batch_size"], shuffle=False, pin_memory=pin)
test_loader  = DataLoader(CachedChagasDataset(cache_file, test_indices),  batch_size=CFG["batch_size"], shuffle=False, pin_memory=pin)
ood_loader   = DataLoader(CachedChagasDataset(cache_file, samitrop_indices), batch_size=CFG["batch_size"], shuffle=False, pin_memory=False)


# === CODE CELL 6 ===
import os, time
import numpy as np

CACHE_TRAIN = "ecgfounder_train_code15.npz"
CACHE_VAL   = "ecgfounder_val_code15.npz"
CACHE_TEST  = "ecgfounder_test_code15.npz"
CACHE_OOD   = "ecgfounder_samitrop_ood.npz"

CHUNK_SIZE = 100

# Release the MPS model before loading a new one on CPU. Holding two models
# simultaneously (MPS + CPU) with torch.load(weights_only=False) crashes the
# kernel on macOS/MPS.
if 'model' in dir() and hasattr(model, 'parameters'):
    del model
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Load ECGFounder directly on CPU from the weights file. Calling
# model.to(cpu) on an existing MPS model hangs the kernel.
cpu = torch.device("cpu")
logger.info("Loading ECGFounder on CPU (fresh instance from weights file)...")
cpu_model = Net1D(
    in_channels=12, base_filters=64, ratio=1,
    filter_list=[64,160,160,400,400,1024,1024],
    m_blocks_list=[2,2,2,3,3,4,4],
    kernel_size=16, stride=2, groups_width=16,
    verbose=False, use_bn=False, use_do=False, n_classes=150
)
_ckpt = torch.load(CFG["weights_path"], map_location=cpu, weights_only=False)
_sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt
cpu_model.load_state_dict(_sd, strict=False)
cpu_model.eval()
logger.info("CPU model ready.")

@torch.no_grad()
def extract_features_chunked(model, loader, desc="Extracting", chunk_size=CHUNK_SIZE):
    tmp_path = f"_tmp_{desc.lower().replace(' ', '_')}.npz"
    done_features, done_labels = [], []
    start_batch = 0
    if os.path.exists(tmp_path):
        tmp = np.load(tmp_path, allow_pickle=True)
        done_features = list(tmp["X"])
        done_labels   = list(tmp["y"])
        start_batch   = int(tmp["next_batch"])
        logger.info(f"{desc}: resuming from batch {start_batch}/{len(loader)}")

    t0 = time.time()
    for i, (x, y) in enumerate(tqdm(loader, desc=desc, initial=start_batch, total=len(loader))):
        if i < start_batch:
            continue
        logits = model(x.to(cpu))
        probs = torch.sigmoid(logits)
        done_features.append(probs.numpy())
        done_labels.append(y.numpy())

        if i == start_batch:
            elapsed = time.time() - t0
            est = elapsed * (len(loader) - start_batch)
            logger.info(f"First batch: {elapsed:.1f}s -> estimated time: {est/60:.1f} min")

        if (i + 1) % chunk_size == 0:
            np.savez(tmp_path,
                     X=np.concatenate(done_features),
                     y=np.concatenate(done_labels),
                     next_batch=i + 1)

    X = np.concatenate(done_features)
    y = np.concatenate(done_labels)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return X, y

def load_or_extract(cache_path, model, loader, desc):
    if os.path.exists(cache_path):
        logger.info(f"Cache found: {cache_path}")
        npz = np.load(cache_path)
        return npz["X"], npz["y"]
    X, y = extract_features_chunked(model, loader, desc=desc)
    np.savez(cache_path, X=X, y=y)
    logger.info(f"Saved: {cache_path} shape={X.shape}")
    return X, y

X_train, y_train = load_or_extract(CACHE_TRAIN, cpu_model, train_loader, "Train")
X_val,   y_val   = load_or_extract(CACHE_VAL,   cpu_model, val_loader,   "Val")
X_test,  y_test  = load_or_extract(CACHE_TEST,  cpu_model, test_loader,  "Test (CODE-15)")
X_ood,   y_ood   = load_or_extract(CACHE_OOD,   cpu_model, ood_loader,   "OOD (SaMi-Trop)")

logger.info(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
logger.info(f"X_ood:   {X_ood.shape}   | OOD Chagas+ rate: {y_ood.mean():.3f}")
print(f"X_train shape: {X_train.shape}")
print(f"Test set (CODE-15%): {len(y_test)} samples | Chagas+: {int(y_test.sum())} ({100*y_test.mean():.1f}%)")
print(f"OOD (SaMi-Trop):     {len(y_ood)} samples | Chagas+: {int(y_ood.sum())} ({100*y_ood.mean():.1f}%) [sensitivity only]")


# === CODE CELL 8 ===
import joblib
from sklearn.model_selection import StratifiedKFold

XGB_PKL    = "approach13_xgb_code15_best.pkl"
LR_PKL     = "approach13_lr_code15_best.pkl"
SCALER_PKL = "approach13_scaler_code15.pkl"

if os.path.exists(XGB_PKL):
    logger.info(f"[CACHE HIT] Loading XGBoost from {XGB_PKL}")
    clf = joblib.load(XGB_PKL)
else:
    logger.info("Training XGBoost Classifier with GridSearchCV...")

    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=CFG["random_seed"],
        eval_metric="aucpr",
    )

    param_grid = {
        'early_stopping_rounds': [None],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500]
    }

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='average_precision',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG['random_seed']),
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    clf = grid_search.best_estimator_
    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")

    clf.set_params(early_stopping_rounds=20)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    logger.info(f"Best iteration: {clf.best_iteration}")

    joblib.dump(clf, XGB_PKL)
    logger.info(f"Saved XGBoost model to {XGB_PKL}")

scaler = StandardScaler()

if os.path.exists(LR_PKL) and os.path.exists(SCALER_PKL):
    logger.info(f"[CACHE HIT] Loading Logistic Regression from {LR_PKL}")
    lr_model = joblib.load(LR_PKL)
    scaler = joblib.load(SCALER_PKL)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
else:
    logger.info("Training Logistic Regression (L1/Lasso) Baseline...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    lr_base = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        random_state=CFG["random_seed"],
        max_iter=1000
    )
    lr_param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    lr_grid = GridSearchCV(
        lr_base,
        lr_param_grid,
        scoring='average_precision',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG['random_seed']),
        verbose=1
    )
    lr_grid.fit(X_train_scaled, y_train)
    lr_model = lr_grid.best_estimator_
    logger.info(f"Best Logistic Regression parameters: {lr_grid.best_params_}")

    joblib.dump(lr_model, LR_PKL)
    joblib.dump(scaler, SCALER_PKL)
    logger.info(f"Saved LR model to {LR_PKL} and scaler to {SCALER_PKL}")


# === CODE CELL 9 ===
def bootstrap_metrics(y_true, y_prob, y_pred, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    sens_scores = []
    spec_scores = []
    ppv_scores = []
    npv_scores = []
    
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping CIs"):
        indices = resample(np.arange(len(y_true)))
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        y_t = y_true[indices]
        y_p_prob = y_prob[indices]
        y_p_pred = y_pred[indices]
        
        auroc_scores.append(roc_auc_score(y_t, y_p_prob))
        auprc_scores.append(average_precision_score(y_t, y_p_prob))
        f1_scores.append(f1_score(y_t, y_p_pred))
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_p_pred).ravel()
        sens_scores.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec_scores.append(tn / (tn + fp) if (tn + fp) else 0.0)
        ppv_scores.append(tp / (tp + fp) if (tp + fp) else 0.0)
        npv_scores.append(tn / (tn + fn) if (tn + fn) else 0.0)
        
    def get_ci(scores):
        return np.percentile(scores, 2.5), np.percentile(scores, 97.5)
        
    logger.info(f"Bootstrap: used {len(auroc_scores)}/{n_bootstraps} valid samples")
    return {
        "AUROC_CI": get_ci(auroc_scores),
        "AUPRC_CI": get_ci(auprc_scores),
        "F1_CI": get_ci(f1_scores),
        "Sens_CI": get_ci(sens_scores),
        "Spec_CI": get_ci(spec_scores),
        "PPV_CI": get_ci(ppv_scores),
        "NPV_CI": get_ci(npv_scores),
        "AUROC_scores": auroc_scores,
        "AUPRC_scores": auprc_scores
    }

def evaluate_model(y_val_true, y_prob_val, y_test_true, y_prob_test, name="Model"):
    """Select threshold on val, report full metrics on test (both classes required)."""
    # Threshold chosen on validation set with fixed sensitivity >= 0.90.
    precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_prob_val)
    valid_idx = np.where(recalls[:-1] >= 0.90)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        best_threshold = thresholds[best_idx]
        logger.info(f"Chosen threshold (Sens >= 0.90): {best_threshold:.4f}")
    else:
        f1_scores_val = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores_val)
        best_threshold = thresholds[best_idx]
        logger.info(f"Fallback threshold (Max F1): {best_threshold:.4f}")
    
    y_pred_test = (y_prob_test >= best_threshold).astype(int)
    
    auroc = roc_auc_score(y_test_true, y_prob_test)
    auprc = average_precision_score(y_test_true, y_prob_test)
    f1 = f1_score(y_test_true, y_pred_test)
    brier = brier_score_loss(y_test_true, y_prob_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_pred_test).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    
    cis = bootstrap_metrics(y_test_true, y_prob_test, y_pred_test)
    
    logger.info(f"--- {name} Results ---")
    logger.info(f"AUROC: {auroc:.3f} (95% CI: {cis['AUROC_CI'][0]:.3f}-{cis['AUROC_CI'][1]:.3f})")
    logger.info(f"AUPRC: {auprc:.3f} (95% CI: {cis['AUPRC_CI'][0]:.3f}-{cis['AUPRC_CI'][1]:.3f})")
    logger.info(f"F1:    {f1:.3f} (95% CI: {cis['F1_CI'][0]:.3f}-{cis['F1_CI'][1]:.3f})")
    logger.info(f"Sens:  {sens:.3f} (95% CI: {cis['Sens_CI'][0]:.3f}-{cis['Sens_CI'][1]:.3f})")
    logger.info(f"Spec:  {spec:.3f} (95% CI: {cis['Spec_CI'][0]:.3f}-{cis['Spec_CI'][1]:.3f})")
    logger.info(f"PPV:   {ppv:.3f} (95% CI: {cis['PPV_CI'][0]:.3f}-{cis['PPV_CI'][1]:.3f})")
    logger.info(f"NPV:   {npv:.3f} (95% CI: {cis['NPV_CI'][0]:.3f}-{cis['NPV_CI'][1]:.3f})")
    logger.info(f"Brier: {brier:.4f}")
    
    return y_prob_test, best_threshold, cis

X_test_scaled = scaler.transform(X_test)
X_ood_scaled  = scaler.transform(X_ood)

# CODE-15% held-out test set has both classes present (Chagas+ ~13.8%),
# so AUROC/AUPRC are well-defined and comparable with approaches 4-10.
logger.info("Evaluating XGBoost on CODE-15% test set...")
y_test_prob_xgb, xgb_thresh, xgb_cis = evaluate_model(
    y_val, clf.predict_proba(X_val)[:, 1],
    y_test, clf.predict_proba(X_test)[:, 1],
    name="XGBoost (CODE-15% Test)"
)

logger.info("Evaluating Logistic Regression on CODE-15% test set...")
y_test_prob_lr, lr_thresh, lr_cis = evaluate_model(
    y_val, lr_model.predict_proba(X_val_scaled)[:, 1],
    y_test, lr_model.predict_proba(X_test_scaled)[:, 1],
    name="Logistic Regression (CODE-15% Test)"
)

diff_scores = np.array(xgb_cis["AUPRC_scores"]) - np.array(lr_cis["AUPRC_scores"])
p_value = np.mean(diff_scores <= 0)
logger.info(f"Paired Bootstrap Test on AUPRC (XGBoost vs LR): p-value = {p_value:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_predictions(y_test, y_test_prob_xgb, n_bins=10, name="XGBoost", ax=ax)
CalibrationDisplay.from_predictions(y_test, y_test_prob_lr,  n_bins=10, name="Logistic Regression", ax=ax)
ax.set_title("Calibration Curve -- CODE-15% Test Set (held-out, IID)")
plt.show()

# SaMi-Trop OOD set is single-class (100% Chagas+), so AUROC/AUPRC are
# undefined. Sensitivity is reported at the threshold selected on the val set.
logger.info("OOD Evaluation: SaMi-Trop (domain-shift, 100% Chagas+)...")

y_ood_prob_xgb = clf.predict_proba(X_ood)[:, 1]
y_ood_prob_lr  = lr_model.predict_proba(X_ood_scaled)[:, 1]

xgb_ood_sens = (y_ood_prob_xgb >= xgb_thresh).mean()
lr_ood_sens  = (y_ood_prob_lr  >= lr_thresh).mean()

logger.info(f"OOD SaMi-Trop Sensitivity -- XGBoost: {xgb_ood_sens:.3f} | LR: {lr_ood_sens:.3f}")
print(f"\nSaMi-Trop OOD (domain-shift, n={len(y_ood)}):")
print(f"  XGBoost sensitivity at val threshold ({xgb_thresh:.4f}): {xgb_ood_sens:.3f}")
print(f"  LR sensitivity      at val threshold ({lr_thresh:.4f}): {lr_ood_sens:.3f}")


# === CODE CELL 10 ===
# TODO: replace with the actual ECGFounder label names (e.g., from the
# MIMIC-IV-ECG documentation or paper supplement).
label_names = ['ABNORMAL ECG', 'NORMAL SINUS RHYTHM', 'NORMAL ECG', 'SINUS RHYTHM', 'SINUS BRADYCARDIA', 'ATRIAL FIBRILLATION',
    'SINUS TACHYCARDIA', 'OTHERWISE NORMAL ECG', 'LEFT AXIS DEVIATION', 'PREMATURE VENTRICULAR COMPLEXES', 'BORDERLINE ECG',
    'RIGHT BUNDLE BRANCH BLOCK', 'SEPTAL INFARCT', 'LEFT ATRIAL ENLARGEMENT', 'NONSPECIFIC T WAVE ABNORMALITY', 'LOW VOLTAGE QRS',
    'PREMATURE ATRIAL COMPLEXES', 'ANTERIOR INFARCT', 'INCOMPLETE RIGHT BUNDLE BRANCH BLOCK', 'PREMATURE SUPRAVENTRICULAR COMPLEXES',
    'LEFT BUNDLE BRANCH BLOCK', 'NONSPECIFIC T WAVE ABNORMALITY NOW EVIDENT IN', 'NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN',
    'T WAVE INVERSION NOW EVIDENT IN', 'LATERAL INFARCT', 'NONSPECIFIC ST ABNORMALITY', 'LEFT VENTRICULAR HYPERTROPHY',
    'T WAVE INVERSION NO LONGER EVIDENT IN', 'WITH RAPID VENTRICULAR RESPONSE', 'QT HAS SHORTENED', 'QT HAS LENGTHENED', 'FUSION COMPLEXES',
    'ATRIAL FLUTTER', 'MARKED SINUS BRADYCARDIA', 'WITH SINUS ARRHYTHMIA', 'NONSPECIFIC ST AND T WAVE ABNORMALITY',
    'LEFT ANTERIOR FASCICULAR BLOCK', 'RIGHT AXIS DEVIATION', 'ECTOPIC ATRIAL RHYTHM', 'UNDETERMINED RHYTHM', 'ANTEROSEPTAL INFARCT',
    'RIGHTWARD AXIS', 'ST NOW DEPRESSED IN', 'WITH SHORT PR', 'WITH MARKED SINUS ARRHYTHMIA', 'ST NO LONGER DEPRESSED IN',
    'INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN', 'NON-SPECIFIC CHANGE IN ST SEGMENT IN',
    'NONSPECIFIC T WAVE ABNORMALITY HAS REPLACED INVERTED T WAVES IN', 'JUNCTIONAL RHYTHM', 'ELECTRONIC ATRIAL PACEMAKER',
    'ABERRANT CONDUCTION', 'ELECTRONIC VENTRICULAR PACEMAKER', 'T WAVE INVERSION LESS EVIDENT IN', 'ANTEROLATERAL INFARCT',
    'WITH REPOLARIZATION ABNORMALITY', "RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY",
    'T WAVE INVERSION MORE EVIDENT IN', 'WIDE QRS RHYTHM', 'WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES',
    'RIGHT ATRIAL ENLARGEMENT', 'INFERIOR INFARCT', 'INCOMPLETE LEFT BUNDLE BRANCH BLOCK',
    'VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY', 'OR DIGITALIS EFFECT', 'BIFASCICULAR BLOCK', 'ST NO LONGER ELEVATED IN',
    'WITH SLOW VENTRICULAR RESPONSE', 'ST ELEVATION NOW PRESENT IN', 'PREMATURE ECTOPIC COMPLEXES', 'LEFT POSTERIOR FASCICULAR BLOCK',
    'T WAVE AMPLITUDE HAS DECREASED IN', 'WITH A COMPETING JUNCTIONAL PACEMAKER', 'RIGHT SUPERIOR AXIS DEVIATION', 'BIATRIAL ENLARGEMENT',
    'VENTRICULAR-PACED RHYTHM', 'ATRIAL-PACED RHYTHM', 'T WAVE AMPLITUDE HAS INCREASED IN', 'WITH QRS WIDENING', 'WITH 1ST DEGREE AV BLOCK',
    'PROLONGED QT', 'WITH PROLONGED AV CONDUCTION', 'RIGHT VENTRICULAR HYPERTROPHY', 'WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY',
    'ATRIAL-SENSED VENTRICULAR-PACED RHYTHM', 'AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER', 'PULMONARY DISEASE PATTERN',
    'ACUTE MI / STEMI', 'INFERIOR-POSTERIOR INFARCT', 'NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY',
    'PREMATURE VENTRICULAR AND FUSION COMPLEXES', 'IN A PATTERN OF BIGEMINY', 'AV DUAL-PACED RHYTHM', 'SUPRAVENTRICULAR TACHYCARDIA',
    'VENTRICULAR-PACED COMPLEXES', 'WIDE QRS TACHYCARDIA', "RSR' PATTERN IN V1", 'ST LESS DEPRESSED IN', 'VENTRICULAR TACHYCARDIA',
    'EARLY REPOLARIZATION', 'ST MORE DEPRESSED IN', 'ANTEROLATERAL LEADS', 'ELECTRONIC DEMAND PACING',
    'RBBB AND LEFT ANTERIOR FASCICULAR BLOCK', 'LATERAL INJURY PATTERN', 'BIVENTRICULAR PACEMAKER DETECTED',
    'SUSPECT UNSPECIFIED PACEMAKER FAILURE', 'WOLFF-PARKINSON-WHITE', 'WITH VENTRICULAR ESCAPE COMPLEXES', 'INFERIOR INJURY PATTERN',
    'CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT', 'ST ELEVATION HAS REPLACED ST DEPRESSION IN',
    'NONSPECIFIC INTRAVENTRICULAR BLOCK', 'MASKED BY FASCICULAR BLOCK', 'PEDIATRIC ECG ANALYSIS', 'BLOCKED',
    'WITH UNDETERMINED RHYTHM IRREGULARITY', 'LEFTWARD AXIS', 'WITH 2ND DEGREE SA BLOCK MOBITZ I', 'ACUTE', 'ABNORMAL LEFT AXIS DEVIATION',
    'WITH COMPLETE HEART BLOCK', 'NO P-WAVES FOUND', 'ST LESS ELEVATED IN', 'WITH RETROGRADE CONDUCTION', 'ST MORE ELEVATED IN',
    'JUNCTIONAL BRADYCARDIA', 'WITH VARIABLE AV BLOCK', 'ANTERIOR INJURY PATTERN', 'WITH JUNCTIONAL ESCAPE COMPLEXES', 'ACUTE MI',
    'ACUTE PERICARDITIS', 'POSTERIOR INFARCT', 'IDIOVENTRICULAR RHYTHM', 'WITH 2ND DEGREE SA BLOCK MOBITZ II', 'R IN AVL',
    'SINUS/ATRIAL CAPTURE', 'AV DUAL-PACED COMPLEXES', 'INFEROLATERAL INJURY PATTERN', 'RBBB AND LEFT POSTERIOR FASCICULAR BLOCK',
    'ANTEROLATERAL INJURY PATTERN', 'ATRIAL-PACED COMPLEXES', 'WITH SINUS PAUSE', 'BIVENTRICULAR HYPERTROPHY',
    'ABNORMAL RIGHT AXIS DEVIATION', 'SUPRAVENTRICULAR COMPLEXES', 'WITH 2ND DEGREE AV BLOCK MOBITZ I', 'WITH 2:1 AV CONDUCTION',
    'WITH AV DISSOCIATION', 'MULTIFOCAL ATRIAL TACHYCARDIA']

logger.info("Calculating SHAP values for XGBoost...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
plt.title("SHAP Summary Plot (XGBoost) -- CODE-15% Test Set")
shap.summary_plot(shap_values, X_test, feature_names=label_names, show=False)
plt.tight_layout()
plt.show()

logger.info("Calculating Odds Ratios for Logistic Regression...")
odds_ratios = np.exp(lr_model.coef_[0])
sorted_indices = np.argsort(odds_ratios)[::-1]

print("Top 10 Clinical Features Increasing Chagas Risk (Logistic Regression):")
for idx in sorted_indices[:10]:
    print(f"{label_names[idx]}: OR = {odds_ratios[idx]:.3f}")

print("\nTop 10 Clinical Features Decreasing Chagas Risk (Logistic Regression):")
for idx in sorted_indices[-10:][::-1]:
    print(f"{label_names[idx]}: OR = {odds_ratios[idx]:.3f}")


