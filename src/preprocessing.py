"""
ECG signal preprocessing utilities for the Chagas disease classification pipeline.

Handles loading WFDB-format ECGs, bandpass filtering, resampling, and
standardisation to a fixed-length 12-lead tensor suitable for model input.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

def find_records(data_dir: str) -> list[str]:
    """Return a sorted list of record paths (no extension) under *data_dir*."""
    data_dir = Path(data_dir)
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".hea"):
                rel = os.path.relpath(os.path.join(root, f[:-4]), data_dir)
                records.append(rel)
    records.sort()
    return records

def load_record(data_dir: str, record_name: str) -> Tuple[np.ndarray, dict]:
    """Load a single WFDB record.

    Returns
    -------
    signal : np.ndarray, shape (n_samples, n_leads)
        Physical (mV) signal values.
    meta : dict
        Keys: ``fs`` (sampling frequency), ``age``, ``sex``, ``chagas``
        (label: 1 positive, 0 negative, -1 unknown).
    """
    rec = wfdb.rdrecord(os.path.join(data_dir, record_name))
    signal = rec.p_signal
    fs = rec.fs

    age = _extract_age(rec)
    sex = _extract_sex(rec)
    chagas = _extract_label(rec)

    meta = {"fs": fs, "age": age, "sex": sex, "chagas": chagas}
    return signal, meta

def _extract_age(rec: wfdb.Record) -> float:
    """Best-effort extraction of patient age from WFDB comments."""
    for comment in (rec.comments or []):
        lower = comment.lower()
        if "age" in lower:
            parts = lower.split(":")
            if len(parts) >= 2:
                try:
                    return float(parts[-1].strip())
                except ValueError:
                    pass
    return float("nan")

def _extract_sex(rec: wfdb.Record) -> int:
    """Return 0 = female, 1 = male, -1 = unknown."""
    for comment in (rec.comments or []):
        lower = comment.lower()
        if "sex" in lower:
            if "male" in lower and "female" not in lower:
                return 1
            if "female" in lower:
                return 0
    return -1

def _extract_label(rec: wfdb.Record) -> int:
    """Return 1 = Chagas positive, 0 = negative, -1 = unknown."""
    for comment in (rec.comments or []):
        lower = comment.lower().strip()
        if lower.startswith("#chagas") or lower.startswith("chagas"):
            parts = lower.split(":")
            if len(parts) >= 2:
                val = parts[-1].strip()
                if val in ("true", "1", "yes"):
                    return 1
                if val in ("false", "0", "no"):
                    return 0
        if "labels" in lower or "dx" in lower:
            if "chagas" in lower:
                return 1
    return -1

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 47.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to each lead."""
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=0).astype(np.float32)

def resample_signal(
    signal: np.ndarray, fs_orig: float, fs_target: float
) -> np.ndarray:
    """Resample the signal from *fs_orig* to *fs_target* Hz."""
    if fs_orig == fs_target:
        return signal
    up = int(fs_target)
    down = int(fs_orig)
    d = gcd(up, down)
    up, down = up // d, down // d
    return resample_poly(signal, up, down, axis=0).astype(np.float32)

def pad_or_truncate(
    signal: np.ndarray, target_length: int
) -> np.ndarray:
    """Ensure the signal has exactly *target_length* samples (axis 0)."""
    n = signal.shape[0]
    if n >= target_length:
        return signal[:target_length]
    pad_width = [(0, target_length - n)] + [(0, 0)] * (signal.ndim - 1)
    return np.pad(signal, pad_width, mode="constant", constant_values=0.0)

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Per-lead zero-mean, unit-variance standardisation."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((signal - mean) / std).astype(np.float32)

def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    target_fs: float = 500.0,
    target_duration_s: float = 10.0,
    filter_low: float = 0.5,
    filter_high: float = 47.0,
) -> np.ndarray:
    """Full preprocessing pipeline: filter, resample, pad/truncate, normalise.

    Returns array of shape (target_fs * target_duration_s, n_leads).
    """
    signal = bandpass_filter(signal, fs, low=filter_low, high=filter_high)
    signal = resample_signal(signal, fs, target_fs)
    target_len = int(target_fs * target_duration_s)
    signal = pad_or_truncate(signal, target_len)
    signal = normalize_signal(signal)
    return signal
