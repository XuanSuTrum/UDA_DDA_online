"""Data utilities for new-user quasi-online UDA-DDA experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from create_de4 import process_one_trial


SEED_LABELS_BY_SESSION = {
    1: [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1],
    2: [0, 1, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1],
    3: [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1],
}

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["negative", "neutral", "positive"]


@dataclass
class TrialData:
    subject_id: int
    trial_id: int
    raw_shape: Tuple[int, int]
    feature: np.ndarray
    trial_label: int
    window_labels: np.ndarray


@dataclass
class FeatureStandardizer:
    scaler: StandardScaler

    def transform_trial(self, trial: TrialData) -> TrialData:
        feature = self.scaler.transform(trial.feature).astype(np.float32)
        return TrialData(
            subject_id=trial.subject_id,
            trial_id=trial.trial_id,
            raw_shape=trial.raw_shape,
            feature=feature,
            trial_label=trial.trial_label,
            window_labels=trial.window_labels.astype(np.int64),
        )

    def transform_trials(self, trials: Sequence[TrialData]) -> List[TrialData]:
        return [self.transform_trial(trial) for trial in trials]


def raw_label_to_class(raw_label: int) -> int:
    if int(raw_label) not in LABEL_MAP:
        raise ValueError(f"unknown SEED label {raw_label}; expected one of {sorted(LABEL_MAP)}")
    return LABEL_MAP[int(raw_label)]


def seed_trial_labels(session: int = 1) -> List[int]:
    labels = SEED_LABELS_BY_SESSION.get(session)
    if labels is None:
        raise ValueError(f"unsupported SEED session {session}; available: {sorted(SEED_LABELS_BY_SESSION)}")
    return [raw_label_to_class(label) for label in labels]


def _subject_id_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def _sort_key_with_digits(text: str) -> Tuple[int, str]:
    digits = re.findall(r"\d+", text)
    return (int(digits[-1]) if digits else 10**9, text)


def _detect_eeg_keys(mat_data: dict) -> List[str]:
    keys = [key for key in mat_data if not key.startswith("__")]
    eeg_keys = [
        key
        for key in keys
        if "eeg" in key.lower()
        and isinstance(mat_data[key], np.ndarray)
        and mat_data[key].ndim == 2
    ]
    if not eeg_keys:
        eeg_keys = [
            key
            for key in keys
            if isinstance(mat_data[key], np.ndarray)
            and mat_data[key].ndim == 2
            and 62 in mat_data[key].shape
        ]
    return sorted(eeg_keys, key=_sort_key_with_digits)


def process_trial_to_de(
    subject_id: int,
    trial_id: int,
    raw_trial: np.ndarray,
    trial_label: int,
    fs: int = 200,
    window_size: float = 1.0,
    apply_lds: bool = True,
) -> TrialData:
    feature = process_one_trial(raw_trial, fs=fs, window_size=window_size, apply_lds=apply_lds)
    labels = np.full(feature.shape[0], trial_label, dtype=np.int64)
    raw_shape = tuple(int(dim) for dim in raw_trial.shape)
    return TrialData(subject_id, trial_id, raw_shape, feature, trial_label, labels)


def load_seed_raw_subject_trials(
    raw_data_dir: str,
    subject_ids: Optional[Sequence[int]] = None,
    fs: int = 200,
    session: int = 1,
    baseline_seconds: float = 63.0,
    window_size: float = 1.0,
    apply_lds: bool = True,
) -> Dict[int, List[TrialData]]:
    """Load raw SEED .mat subjects while preserving trial order and boundaries."""
    data_dir = Path(raw_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"raw_data_dir does not exist: {data_dir}")

    requested = set(subject_ids or [])
    subject_files = sorted(data_dir.glob("*.mat"), key=lambda path: _subject_id_from_path(path, 10**9))
    if not subject_files:
        raise FileNotFoundError(f"no .mat files found in raw_data_dir: {data_dir}")

    trial_labels = seed_trial_labels(session)
    subjects: Dict[int, List[TrialData]] = {}
    for fallback_idx, mat_path in enumerate(subject_files, start=1):
        subject_id = _subject_id_from_path(mat_path, fallback_idx)
        if requested and subject_id not in requested:
            continue

        mat_data = scio.loadmat(mat_path)
        eeg_keys = _detect_eeg_keys(mat_data)
        if not eeg_keys:
            raise ValueError(f"no EEG trial arrays detected in {mat_path}")
        if len(eeg_keys) < len(trial_labels):
            raise ValueError(
                f"{mat_path} has {len(eeg_keys)} EEG trials, expected at least {len(trial_labels)}"
            )

        baseline_samples = int(round(baseline_seconds * fs))
        trials: List[TrialData] = []
        for trial_idx, key in enumerate(eeg_keys[: len(trial_labels)], start=1):
            raw_trial = np.asarray(mat_data[key], dtype=np.float64)
            if raw_trial.shape[0] != 62 and raw_trial.shape[1] == 62:
                raw_trial = raw_trial.T
            if baseline_samples > 0 and raw_trial.shape[1] > baseline_samples:
                raw_trial = raw_trial[:, baseline_samples:]
            label = trial_labels[trial_idx - 1]
            trials.append(
                process_trial_to_de(
                    subject_id,
                    trial_idx,
                    raw_trial,
                    label,
                    fs=fs,
                    window_size=window_size,
                    apply_lds=apply_lds,
                )
            )
        subjects[subject_id] = trials

    missing = requested - set(subjects)
    if missing:
        raise FileNotFoundError(f"requested subject_ids not found in {data_dir}: {sorted(missing)}")
    return subjects


def _read_struct_field(struct, field: str):
    try:
        return struct[field][0, 0]
    except Exception:
        return None


def _load_flat_de_mat(mat_data: dict, session: int):
    candidates = [
        f"dataset_session{session}",
        "dataset",
        "data",
    ]
    for key in candidates:
        if key in mat_data:
            value = mat_data[key]
            feature = _read_struct_field(value, "feature")
            label = _read_struct_field(value, "label")
            if feature is not None and label is not None:
                return np.asarray(feature), np.asarray(label).reshape(-1)
    if "feature" in mat_data and "label" in mat_data:
        return np.asarray(mat_data["feature"]), np.asarray(mat_data["label"]).reshape(-1)
    return None, None


def load_seed_de_mat_subject_trials(
    de_data_dir: str,
    subject_ids: Optional[Sequence[int]] = None,
    session: int = 1,
    trials_per_subject: int = 15,
) -> Dict[int, List[TrialData]]:
    """Fallback loader for processed DE .mat files.

    If per-trial arrays are unavailable, flat subject-level features are split
    into contiguous trial chunks. Raw mode is preferred for exact trial lengths.
    """
    data_dir = Path(de_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"de_data_dir does not exist: {data_dir}")
    requested = set(subject_ids or [])
    files = sorted(data_dir.glob("*.mat"), key=lambda path: _subject_id_from_path(path, 10**9))
    if not files:
        raise FileNotFoundError(f"no .mat files found in de_data_dir: {data_dir}")

    default_labels = seed_trial_labels(session)
    subjects: Dict[int, List[TrialData]] = {}
    for fallback_idx, mat_path in enumerate(files, start=1):
        subject_id = _subject_id_from_path(mat_path, fallback_idx)
        if requested and subject_id not in requested:
            continue
        mat_data = scio.loadmat(mat_path)

        trial_keys = [
            key
            for key in mat_data
            if not key.startswith("__")
            and re.search(r"trial", key, flags=re.IGNORECASE)
            and isinstance(mat_data[key], np.ndarray)
            and mat_data[key].ndim == 2
        ]
        trials: List[TrialData] = []
        if trial_keys:
            for idx, key in enumerate(sorted(trial_keys, key=_sort_key_with_digits), start=1):
                feature = np.asarray(mat_data[key], dtype=np.float32)
                label = default_labels[idx - 1] if idx <= len(default_labels) else int(0)
                trials.append(
                    TrialData(subject_id, idx, (62, 0), feature, label, np.full(feature.shape[0], label))
                )
        else:
            feature, label = _load_flat_de_mat(mat_data, session)
            if feature is None or label is None:
                raise ValueError(f"could not find feature/label fields in {mat_path}")
            feature = np.asarray(feature, dtype=np.float32)
            label = np.asarray(label).reshape(-1)
            if label.min() < 0:
                label = np.asarray([raw_label_to_class(int(x)) for x in label], dtype=np.int64)
            else:
                label = label.astype(np.int64)
            feature_chunks = np.array_split(feature, trials_per_subject)
            label_chunks = np.array_split(label, trials_per_subject)
            for idx, (feature_chunk, label_chunk) in enumerate(zip(feature_chunks, label_chunks), start=1):
                if idx <= len(default_labels):
                    trial_label = default_labels[idx - 1]
                else:
                    trial_label = int(np.bincount(label_chunk).argmax())
                trials.append(
                    TrialData(
                        subject_id,
                        idx,
                        (62, 0),
                        feature_chunk.astype(np.float32),
                        int(trial_label),
                        label_chunk.astype(np.int64),
                    )
                )
        subjects[subject_id] = trials

    missing = requested - set(subjects)
    if missing:
        raise FileNotFoundError(f"requested subject_ids not found in {data_dir}: {sorted(missing)}")
    return subjects


def build_target_trial_stream(subject_trials: Sequence[TrialData]) -> List[TrialData]:
    return sorted(subject_trials, key=lambda trial: trial.trial_id)


def split_calibration_and_test_trials(
    trials: Sequence[TrialData],
    k: int,
) -> Tuple[List[TrialData], List[TrialData]]:
    ordered = build_target_trial_stream(trials)
    if k < 0:
        raise ValueError("k must be non-negative")
    if k >= len(ordered):
        raise ValueError(f"k={k} leaves no test trials for {len(ordered)} available trials")
    return ordered[:k], ordered[k:]


def concat_trials(trials: Sequence[TrialData]):
    if not trials:
        return (
            np.empty((0, 310), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    features = []
    labels = []
    meta = []
    for trial in trials:
        features.append(trial.feature.astype(np.float32))
        labels.append(trial.window_labels.astype(np.int64))
        for window_id in range(trial.feature.shape[0]):
            meta.append((trial.subject_id, trial.trial_id, window_id))
    return np.vstack(features), np.concatenate(labels), meta


def fit_source_standardizer(source_trials: Sequence[TrialData]) -> FeatureStandardizer:
    features, _, _ = concat_trials(source_trials)
    if features.shape[0] == 0:
        raise ValueError("cannot fit standardizer on empty source trials")
    scaler = StandardScaler()
    scaler.fit(features)
    return FeatureStandardizer(scaler)


def make_tensor_loader(
    trials: Sequence[TrialData],
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    features, labels, _ = concat_trials(trials)
    if features.shape[0] == 0:
        raise ValueError("cannot build loader from empty trials")
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)


def make_source_loader(source_trials: Sequence[TrialData], batch_size: int) -> DataLoader:
    return make_tensor_loader(source_trials, batch_size=batch_size, shuffle=True, drop_last=False)


def make_target_buffer_loader(buffer_trials: Sequence[TrialData], batch_size: int) -> DataLoader:
    return make_tensor_loader(buffer_trials, batch_size=batch_size, shuffle=False, drop_last=False)


def generate_synthetic_subject_trials(
    num_subjects: int = 3,
    num_trials: int = 15,
    windows_per_trial: int = 8,
    feature_dim: int = 310,
    seed: int = 42,
) -> Dict[int, List[TrialData]]:
    """Create deterministic synthetic DE-like features for --dry_run."""
    rng = np.random.default_rng(seed)
    labels = seed_trial_labels(1)
    class_centers = rng.normal(0.0, 1.0, size=(3, feature_dim)).astype(np.float32)
    subjects: Dict[int, List[TrialData]] = {}
    for subject_id in range(1, num_subjects + 1):
        subject_shift = rng.normal(0.0, 0.35, size=(feature_dim,)).astype(np.float32)
        trials = []
        for trial_id in range(1, num_trials + 1):
            label = labels[(trial_id - 1) % len(labels)]
            drift = (trial_id - 1) * 0.015
            feature = (
                class_centers[label]
                + subject_shift
                + drift
                + rng.normal(0.0, 0.6, size=(windows_per_trial, feature_dim))
            ).astype(np.float32)
            window_labels = np.full(windows_per_trial, label, dtype=np.int64)
            trials.append(TrialData(subject_id, trial_id, (62, windows_per_trial * 200), feature, label, window_labels))
        subjects[subject_id] = trials
    return subjects
