"""
ECG 파형 필터링 및 정규화
- Bandpass filter (0.5-50Hz)
- 60Hz Notch filter
- Z-score 정규화
"""
import numpy as np
from scipy.signal import butter, sosfilt, iirnotch
from config import SAMPLE_RATE, HIGHPASS_FREQ, LOWPASS_FREQ, NOTCH_FREQ, FILTER_ORDER


# 필터 계수를 미리 계산 (매번 계산하지 않도록)
_bandpass_sos = butter(FILTER_ORDER, [HIGHPASS_FREQ, LOWPASS_FREQ],
                       btype="bandpass", fs=SAMPLE_RATE, output="sos")
_notch_b, _notch_a = iirnotch(NOTCH_FREQ, Q=30.0, fs=SAMPLE_RATE)


def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """0.5-50Hz Butterworth bandpass filter. Input shape: (n_samples,) or (n_samples, n_leads)"""
    return sosfilt(_bandpass_sos, signal, axis=0)


def notch_filter(signal: np.ndarray) -> np.ndarray:
    """60Hz notch filter. Input shape: (n_samples,) or (n_samples, n_leads)"""
    from scipy.signal import filtfilt
    return filtfilt(_notch_b, _notch_a, signal, axis=0)


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """리드별 Z-score 정규화. Input shape: (n_samples, n_leads)"""
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0  # 0으로 나누기 방지
    return (signal - mean) / std


def preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    전체 전처리 파이프라인: bandpass → notch → z-score
    Input:  (n_samples, n_leads) raw ADC values
    Output: (n_samples, n_leads) 정규화된 신호
    """
    signal = signal.astype(np.float64)
    signal = bandpass_filter(signal)
    signal = notch_filter(signal)
    signal = normalize_zscore(signal)
    return signal.astype(np.float32)


def check_signal_quality(signal: np.ndarray) -> bool:
    """
    기본 품질 검사.
    - 전체가 0인 리드가 있으면 불량
    - NaN/Inf 포함시 불량
    - 진폭이 비정상적으로 큰 경우 불량
    """
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False
    # 리드별 표준편차가 0인 경우 (flat line)
    stds = np.std(signal, axis=0)
    if np.any(stds < 1e-6):
        return False
    return True
