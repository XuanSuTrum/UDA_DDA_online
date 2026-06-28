"""SEED raw-trial EEG to DE feature conversion.

The online experiment keeps trial boundaries, so this module exposes a
single-trial conversion function instead of writing subject-level .mat files.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft


DEFAULT_FS = 200
DEFAULT_WINDOW_SIZE = 1.0

FREQ_BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (14, 30),
    "gamma": (31, 50),
}


def compute_stft_es(signal: np.ndarray, fs: int, low: float, high: float) -> float:
    """Compute mean spectral energy in a frequency band with a Hanning window."""
    signal = np.asarray(signal, dtype=np.float64)
    n_points = signal.shape[0]
    if n_points == 0:
        return 0.0

    fft_size = max(512, int(2 ** np.ceil(np.log2(n_points))))
    windowed = signal * np.hanning(n_points)
    padded = np.pad(windowed, (0, fft_size - n_points), mode="constant")
    spectrum = fft(padded)
    psd = np.abs(spectrum) ** 2 / fft_size
    freqs = np.linspace(0, fs / 2, fft_size // 2 + 1)
    band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(psd[: fft_size // 2 + 1][band_mask]))


def compute_de(es: float) -> float:
    """Differential entropy under a Gaussian assumption."""
    if es <= 0:
        return 0.0
    return float(0.5 * np.log(2 * np.pi * np.e * es))


def lds_smoothing(data: np.ndarray, q: float = 0.0001, r: float = 0.01) -> np.ndarray:
    """Lightweight LDS smoothing along the window dimension."""
    if data.shape[0] <= 1:
        return data

    smoothed = np.zeros_like(data)
    for feature_idx in range(data.shape[1]):
        y = data[:, feature_idx]
        n_steps = len(y)
        x = np.zeros(n_steps)
        p = np.zeros(n_steps)
        x[0] = y[0]
        p[0] = r

        for t in range(1, n_steps):
            x_pred = x[t - 1]
            p_pred = p[t - 1] + q
            k_gain = p_pred / (p_pred + r)
            x[t] = x_pred + k_gain * (y[t] - x_pred)
            p[t] = (1.0 - k_gain) * p_pred

        xs = np.zeros(n_steps)
        xs[-1] = x[-1]
        for t in range(n_steps - 2, -1, -1):
            j_gain = p[t] / (p[t] + q)
            xs[t] = x[t] + j_gain * (xs[t + 1] - x[t])
        smoothed[:, feature_idx] = xs

    return smoothed


def process_one_trial(
    trial_data: np.ndarray,
    fs: int = DEFAULT_FS,
    window_size: float = DEFAULT_WINDOW_SIZE,
    apply_lds: bool = True,
) -> np.ndarray:
    """Convert one raw EEG trial from ``(62, T)`` to ``(n_window, 310)`` DE features."""
    trial_data = np.asarray(trial_data, dtype=np.float64)
    if trial_data.ndim != 2:
        raise ValueError(f"trial_data must be 2-D, got shape {trial_data.shape}")
    if trial_data.shape[0] != 62 and trial_data.shape[1] == 62:
        trial_data = trial_data.T
    if trial_data.shape[0] != 62:
        raise ValueError(f"expected 62 EEG channels, got shape {trial_data.shape}")

    n_channels, n_points = trial_data.shape
    samples_per_window = int(round(fs * window_size))
    if samples_per_window <= 0:
        raise ValueError("fs * window_size must be positive")

    n_windows = n_points // samples_per_window
    if n_windows == 0:
        return np.empty((0, n_channels * len(FREQ_BANDS)), dtype=np.float32)

    trimmed = trial_data[:, : n_windows * samples_per_window]
    features = np.zeros((n_windows, n_channels * len(FREQ_BANDS)), dtype=np.float64)

    for window_idx in range(n_windows):
        start = window_idx * samples_per_window
        end = start + samples_per_window
        window = trimmed[:, start:end]
        for ch_idx in range(n_channels):
            for band_idx, (low, high) in enumerate(FREQ_BANDS.values()):
                es = compute_stft_es(window[ch_idx], fs, low, high)
                features[window_idx, ch_idx * len(FREQ_BANDS) + band_idx] = compute_de(es)

    if apply_lds:
        features = lds_smoothing(features)
    return features.astype(np.float32)
