
import os
import mne
import numpy as np
import pandas as pd
import scipy.signal as sp_sig
# from fooof import FOOOF
from specparam import SpectralModel
from mne_connectivity import spectral_connectivity_time
import sys; sys.path.append(os.getcwd())
from common.timefreq import bandpower, differentialentropy


def extract_differentialentropy(data, fbands=[0.5, 4, 8, 14, 31, 49], fs=128):
    in_shape = data.shape
    data = np.reshape(data, (-1, in_shape[-2], in_shape[-1]))
    n_epochs, n_timepoints, n_channels = data.shape
    n_features = len(fbands) - 1
    features = np.zeros((n_epochs, n_features, n_channels))
    for i in range(n_epochs):
        for j in range(n_channels):
            signal_ij = data[i, :, j]
            features[i, :, j] = differentialentropy(signal_ij, fs, fbands)
    features = np.reshape(features, (*in_shape[:-2], n_features, n_channels))
    return features


def extract_bandpower(data, fbands=[0.5, 4, 8, 14, 31, 49], fs=128):
    in_shape = data.shape
    data = np.reshape(data, (-1, in_shape[-2], in_shape[-1]))
    n_epochs, n_timepoints, n_channels = data.shape
    n_features = len(fbands) - 1
    features = np.zeros((n_epochs, n_features, n_channels))
    for i in range(n_epochs):
        for j in range(n_channels):
            signal_ij = data[i, :, j]
            features[i, :, j] = bandpower(signal_ij, fs, fbands)
    features = np.reshape(features, (*in_shape[:-2], n_features, n_channels))
    return features


def extract_connectivity_features(data, fbands=[0.5, 4, 8, 14, 31, 49], fs=128, method='coh', ch_names=None):
    """Extract connectivity features from epoched EEG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    """
    # Bandpass filter
    freq_broad = (fbands[0], fbands[-1])
    bands = [(fbands[i], fbands[i+1]) for i in range(len(fbands)-1)]

    n_epochs, n_samples, n_channels = data.shape
    data = data.transpose(0, 2, 1)  # Shape (n_epochs, n_channels, n_samples)
    # data = filter_data(data, sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names, fs, ch_types="eeg")
    info.set_montage(montage)
    epochs = mne.EpochsArray(data, info, verbose=False)

    lower_size = int(n_channels * (n_channels - 1) / 2)
    feature_bands = np.zeros((n_epochs, len(bands), lower_size))
    freqs = np.linspace(freq_broad[0], freq_broad[1], int((freq_broad[1] - freq_broad[0]) * 4 + 1))
    conn = spectral_connectivity_time(
        epochs, freqs, method=method, sfreq=fs, 
        fmin=freq_broad[0], fmax=freq_broad[1], faverage=False, n_jobs=1, verbose=False)
    conn_data = conn.get_data('dense')

    feature_cols = []
    idx_lower = np.tril_indices(n_channels, k=-1)
    for i, (fmin, fmax, b) in enumerate(bands):
        idx_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
        conn_band = np.mean(conn_data[:, :, :, idx_freq], axis=-1)
        conn_band = np.nan_to_num(conn_band) # Replace NaN with 0
        feature_bands[:, i, :] = conn_band[:, idx_lower[0], idx_lower[1]]
        feature_cols += [f"{b}_{ch_names[c1]}_{ch_names[c2]}" for c1, c2 in zip(idx_lower[0], idx_lower[1])]

    return feature_bands


def aggregate_peaks(peaks, bands):
    """对单个epoch的峰值按频段聚合，每个频段保留1组最显著特征（功率最高的峰值）"""
    aggregated = {}
    for fmin, fmax, band_name in bands:
        band_peaks = [p for p in peaks if (fmin <= p[0] < fmax)]
        
        if not band_peaks:
            aggregated[f"{band_name}_exist"] = 0
            aggregated[f"{band_name}_freq"] = 0
            aggregated[f"{band_name}_power"] = 0
            aggregated[f"{band_name}_bw"] = 0
        else:
            band_peaks_sorted = sorted(band_peaks, key=lambda x: x[1], reverse=True)
            best_peak = band_peaks_sorted[0]
            aggregated[f"{band_name}_exist"] = 1
            aggregated[f"{band_name}_freq"] = best_peak[0]
            aggregated[f"{band_name}_power"] = best_peak[1]
            aggregated[f"{band_name}_bw"] = best_peak[2]
    return aggregated

def aperiodic_fooof(signal_ch, fs=128, fmin=0.5, fmax=49):
    """使用FOOOF方法拟合信号的PSD并提取aperiodic特征"""
    freqs, psd = sp_sig.welch(signal_ch, fs, nperseg=int(2*fs))
    # signal_ch = filter_data(signal_ch.tolist(), sfreq=fs, l_freq=fmin, h_freq=fmax, verbose=False)
    # psd, freqs = mne.time_frequency.psd_array_welch(
    #     signal_ch, sfreq=fs, fmin=fmin, fmax=fmax, n_per_seg=int(2*fs), verbose=False)
    # psd = psd + 1e-10  # Avoid log(0)
    if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
        raise ValueError("PSD contains NaN or Inf values.")
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_cropped = freqs[freq_mask]
    psd_cropped = psd[freq_mask]
    psd_cropped = np.maximum(psd_cropped, 1e-12)
    fm = SpectralModel(peak_width_limits=[0.5, 12], min_peak_height=0.05, verbose=False)
    fm.fit(freqs_cropped, psd_cropped, freq_range=[fmin, fmax])
    offset, exponent = fm.aperiodic_params_
    peaks = list(fm.peak_params_)  # Each peak: center_freq, power, bandwidth
    return exponent, offset, peaks

def extract_aperiodic_features(data, fbands=[0.5, 4, 8, 14, 31, 49], fs=128):
    """Extract aperiodic features from 30s epoched EEG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    features: pd.DataFrame of shape (n_epochs, n_features)
    features include: 
        ['exponent', 'offset', 
        'band1_exist', 'band1_freq', 'band1_power', 'band1_bw',
        'band2_exist', 'band2_freq', 'band2_power', 'band2_bw',
        ...]
    """
    n_epochs, n_samples, n_channels = data.shape

    freq_broad = (fbands[0], fbands[-1])
    bands = [(fbands[i], fbands[i+1], f"band{i+1}") for i in range(len(fbands)-1)]
    n_features = 2 + 4 * (len(fbands) - 1)  # exponent, offset + 4 features per band

    features = np.zeros((n_epochs, n_features, n_channels))
    for n in range(n_epochs):
        for c in range(n_channels):
            # Initialize features to 0
            feat_ch = np.zeros((n_features,))
    
            epoch_ch = data[n, :, c]
            # Check if channel is empty
            maxv, minv = np.max(epoch_ch), np.min(epoch_ch)
            if maxv-minv != 0:
                # - Extract aperiodic features using FOOOF
                exponent, offset, peaks = aperiodic_fooof(epoch_ch, fs=fs, fmin=freq_broad[0], fmax=freq_broad[1])
                feat_ch[0] = exponent
                feat_ch[1] = offset
                aggregated_peaks = aggregate_peaks(peaks, bands)
                for band in bands:
                    feat_ch_idx = 2 + (bands.index(band)) * 4
                    feat_ch[feat_ch_idx] = aggregated_peaks[f"{band[2]}_exist"]
                    feat_ch[feat_ch_idx + 1] = aggregated_peaks[f"{band[2]}_freq"]
                    feat_ch[feat_ch_idx + 2] = aggregated_peaks[f"{band[2]}_power"]
                    feat_ch[feat_ch_idx + 3] = aggregated_peaks[f"{band[2]}_bw"]
            features[n, :, c] = feat_ch
    return features
