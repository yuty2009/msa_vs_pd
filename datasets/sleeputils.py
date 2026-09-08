
import os
import mne
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from scipy import interpolate
from scipy.integrate import simpson
from mne.filter import filter_data
from mne_connectivity import spectral_connectivity_time
from sklearn.preprocessing import robust_scale
import sys; sys.path.append(os.getcwd())
from common.eegfeature import aggregate_peaks, aperiodic_fooof


def resample(signal, signal_frequency, target_frequency):
    resampling_ratio = signal_frequency / target_frequency
    x_base = np.arange(0, len(signal))

    interpolator = interpolate.interp1d(x_base, signal, axis=0, bounds_error=False, fill_value='extrapolate',)

    x_interp = np.arange(0, len(signal), resampling_ratio)

    signal_duration = signal.shape[0] / signal_frequency
    resampled_length = round(signal_duration * target_frequency)
    resampled_signal = interpolator(x_interp)
    if len(resampled_signal) < resampled_length:
        padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
        resampled_signal = np.concatenate([resampled_signal, padding])

    return resampled_signal


def get_features(data, sf, ch_names=None, metadata=None, center_rolling=False):
    """Extract features from 30s epoched PSG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    features: pd.DataFrame of shape (n_epochs, n_features)
    features include: 
        ['std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 
         'delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma', 
         'r_delta', 'r_theta', 'r_alpha', 'r_sigma', 'r_beta', 'r_gamma',
         'dt', 'ds', 'db', 'dg', 'at', 'abspow', 'perm', 'higuchi', 'petrosian']
    """

    # feature names
    features_eeg = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 
        'delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma',
        'r_delta', 'r_theta', 'r_alpha', 'r_sigma', 'r_beta', 'r_gamma',
        'dt', 'ds', 'db', 'dg', 'at', 'abspow', 'perm', 'higuchi', 'petrosian'
    ]
    features_eog = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp',
        'delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma',
        'r_delta', 'r_theta', 'r_alpha', 'r_sigma', 'r_beta', 'r_gamma',
        'abspow', 'perm', 'higuchi', 'petrosian'
    ]
    features_emg = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 
        'abspow', 'perm', 'higuchi', 'petrosian'
    ]

    # Bandpass filter
    freq_broad = (0.5, 45)
    # FFT & bandpower parameters
    win_sec = 5  # = 2 / freq_broad[0]
    win = int(win_sec * sf)
    kwargs_welch = dict(window="hamming", nperseg=win, average="median")
    bands = [
        (0.5, 4, "delta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
        (30, 45, "gamma"),
    ]

    #######################################################################
    # CALCULATE FEATURES
    #######################################################################

    features = []
    n_epochs, n_samples, n_channels = data.shape
    if ch_names is None: ch_names = [f"ch{i}" for i in range(n_channels)]
    for i, c in enumerate(ch_names):

        # Initialize features to 0
        feat = {}
        if c.startswith('eeg'):
            feature_names = features_eeg
        elif c == 'eog':
            feature_names = features_eog
        elif c == 'emg':
            feature_names = features_emg
        for name in feature_names: feat[name] = np.zeros((n_epochs))

        # Check if channel is empty
        maxv, minv = np.max(data[:, :, i]), np.min(data[:, :, i])
        if maxv-minv != 0:
            # Preprocessing
            epochs = np.zeros((n_epochs, 3000))
            for n in range(n_epochs):
            # - Filter the data
                epochs[n, :] = filter_data(
                    data[n, :, i].tolist(), sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
                )
            # - Extract epochs. Data is now of shape (n_epochs, n_samples).
            # times, epochs = sliding_window(dt_filt, sf=sf, window=30)
            times = np.arange(n_epochs) * 30.0

            # Calculate standard descriptive statistics
            hmob, hcomp = ant.hjorth_params(epochs, axis=1)

            feat = {
                "std": np.std(epochs, ddof=1, axis=1),
                "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
                "skew": sp_stats.skew(epochs, axis=1),
                "kurt": sp_stats.kurtosis(epochs, axis=1),
                "nzc": ant.num_zerocross(epochs, axis=1),
                "hmob": hmob,
                "hcomp": hcomp,
            }

            # Calculate spectral power features (for EEG + EOG)
            freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
            if c != "emg":
                bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
                for j, (_, _, b) in enumerate(bands):
                    feat[b] = bp[j]

            # Add total power
            idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            feat["abspow"] = np.trapz(psd[:, idx_broad], dx=dx)

            # Add relative power for EEG
            if c.startswith("eeg"):
                feat["r_delta"] = feat["delta"] / feat["abspow"]
                feat["r_theta"] = feat["theta"] / feat["abspow"]
                feat["r_alpha"] = feat["alpha"] / feat["abspow"]
                feat["r_sigma"] = feat["sigma"] / feat["abspow"]
                feat["r_beta"] = feat["beta"] / feat["abspow"]
                feat["r_gamma"] = feat["gamma"] / feat["abspow"]

            # Add power ratios for EEG
            if c.startswith("eeg"):
                feat["dt"] = feat["delta"] / feat["theta"]
                feat["ds"] = feat["delta"] / feat["sigma"]
                feat["db"] = feat["delta"] / feat["beta"]
                feat["dg"] = feat["delta"] / feat["gamma"] # 
                feat["at"] = feat["alpha"] / feat["theta"]

            # Calculate entropy and fractal dimension features
            feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
            feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
            feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)

        # Convert to dataframe 
        feat = pd.DataFrame(feat).add_prefix(str(i) + "_")
        features.append(feat)

    # Save features to dataframe
    features = pd.concat(features, axis=1)
    features.index.name = "epoch"

    if center_rolling:
        # Apply centered rolling average (15 epochs = 7 min 30)
        # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
        #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        rollc = features.rolling(window=15, center=True, min_periods=1, win_type="triang").mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix("_c7min_norm")

        # Now look at the past 2 minutes
        rollp = features.rolling(window=4, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix("_p2min_norm")

        # Add to current set of features
        features = features.join(rollc).join(rollp)

    #######################################################################
    # TEMPORAL + METADATA FEATURES AND EXPORT
    #######################################################################

    # Add temporal features
    features["time_hour"] = times / 3600
    features["time_norm"] = times / times[-1]

    # Add metadata if present
    if metadata is not None:
        for c in metadata.keys():
            features[c] = metadata[c]

    # Make sure that age and sex are encoded as int
    if "age" in features.columns:
        features["age"] = features["age"].astype(int)
    if "male" in features.columns:
        features["male"] = features["male"].astype(int)

    # Sort the column names here (same behavior as lightGBM)
    features.sort_index(axis=1, inplace=True)

    return features


def get_features_yasa(data, sf, ch_names=None, metadata=None, center_rolling=False):
    """Extract features from 30s epoched PSG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    features: pd.DataFrame of shape (n_epochs, n_features)
    features include: 
        ['std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 'sdelta',
         'fdelta', 'theta', 'alpha', 'sigma', 'beta', 'dt', 'ds', 'db', 
         'at', 'abspow', 'perm', 'higuchi', 'petrosian']
    """

    # feature names
    features_eeg = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 'sdelta',
        'fdelta', 'theta', 'alpha', 'sigma', 'beta', 'dt', 'ds', 'db', 
        'at', 'abspow', 'perm', 'higuchi', 'petrosian'
    ]
    features_eog = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 'sdelta',
        'fdelta', 'theta', 'alpha', 'sigma', 'beta', 'abspow', 'perm', 
        'higuchi', 'petrosian'
    ]
    features_emg = [
        'std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 'abspow', 
        'perm', 'higuchi', 'petrosian'
    ]

    # Bandpass filter
    freq_broad = (0.4, 30)
    # FFT & bandpower parameters
    win_sec = 5  # = 2 / freq_broad[0]
    win = int(win_sec * sf)
    kwargs_welch = dict(window="hamming", nperseg=win, average="median")
    bands = [
        (0.4, 1, "sdelta"),
        (1, 4, "fdelta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
    ]

    #######################################################################
    # CALCULATE FEATURES
    #######################################################################

    features = []
    n_epochs, n_samples, n_channels = data.shape
    if ch_names is None: ch_names = [f"ch{i}" for i in range(n_channels)]
    for i, c in enumerate(ch_names):

        # Initialize features to 0
        feat = {}
        if c.startswith('eeg'):
            feature_names = features_eeg
        elif c == 'eog':
            feature_names = features_eog
        elif c == 'emg':
            feature_names = features_emg
        for name in feature_names: feat[name] = np.zeros((n_epochs))

        # Check if channel is empty
        maxv, minv = np.max(data[:, :, i]), np.min(data[:, :, i])
        if maxv-minv != 0:
            # Preprocessing
            epochs = np.zeros((n_epochs, 3000))
            for n in range(n_epochs):
            # - Filter the data
                epochs[n, :] = filter_data(
                    data[n, :, i].tolist(), sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
                )
            # - Extract epochs. Data is now of shape (n_epochs, n_samples).
            # times, epochs = sliding_window(dt_filt, sf=sf, window=30)
            times = np.arange(n_epochs) * 30.0

            # Calculate standard descriptive statistics
            hmob, hcomp = ant.hjorth_params(epochs, axis=1)

            feat = {
                "std": np.std(epochs, ddof=1, axis=1),
                "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
                "skew": sp_stats.skew(epochs, axis=1),
                "kurt": sp_stats.kurtosis(epochs, axis=1),
                "nzc": ant.num_zerocross(epochs, axis=1),
                "hmob": hmob,
                "hcomp": hcomp,
            }

            # Calculate spectral power features (for EEG + EOG)
            freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)
            if c != "emg":
                bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
                for j, (_, _, b) in enumerate(bands):
                    feat[b] = bp[j]

            # Add power ratios for EEG
            if c.startswith("eeg"):
                delta = feat["sdelta"] + feat["fdelta"]
                feat["dt"] = delta / feat["theta"]
                feat["ds"] = delta / feat["sigma"]
                feat["db"] = delta / feat["beta"]
                feat["at"] = feat["alpha"] / feat["theta"]

            # Add total power
            idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
            dx = freqs[1] - freqs[0]
            feat["abspow"] = np.trapz(psd[:, idx_broad], dx=dx)

            # Calculate entropy and fractal dimension features
            feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
            feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
            feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)

        # Convert to dataframe 
        feat = pd.DataFrame(feat).add_prefix(c + "_")
        features.append(feat)

    # Save features to dataframe
    features = pd.concat(features, axis=1)
    features.index.name = "epoch"

    if center_rolling:
        # Apply centered rolling average (15 epochs = 7 min 30)
        # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
        #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        rollc = features.rolling(window=15, center=True, min_periods=1, win_type="triang").mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix("_c7min_norm")

        # Now look at the past 2 minutes
        rollp = features.rolling(window=4, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix("_p2min_norm")

        # Add to current set of features
        features = features.join(rollc).join(rollp)

    #######################################################################
    # TEMPORAL + METADATA FEATURES AND EXPORT
    #######################################################################

    # Add temporal features
    features["time_hour"] = times / 3600
    features["time_norm"] = times / times[-1]

    # Add metadata if present
    if metadata is not None:
        for c in metadata.keys():
            features[c] = metadata[c]

    # Downcast float64 to float32 (to reduce size of training datasets)
    cols_float = features.select_dtypes(np.float64).columns.tolist()
    features[cols_float] = features[cols_float].astype(np.float32)
    # Make sure that age and sex are encoded as int
    if "age" in features.columns:
        features["age"] = features["age"].astype(int)
    if "male" in features.columns:
        features["male"] = features["male"].astype(int)

    # Sort the column names here (same behavior as lightGBM)
    features.sort_index(axis=1, inplace=True)

    return features


def get_aperiodic_features(data, sf=100, ch_names=None):
    """Extract aperiodic features from 30s epoched EEG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    features: pd.DataFrame of shape (n_epochs, n_features)
    features include: 
        ['exponent', 'offset', 
        'band1_exist', 'band1_freq', 'band1_power', 'band1_bw',
        'band2_exist', 'band2_freq', 'band2_power', 'band2_bw',
        ...]
    """
    freq_broad = (0.5, 45)
    bands=[
        (0.5, 4, "delta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
        (30, 45, "gamma")
    ]
    
    n_epochs, n_samples, n_channels = data.shape
    if ch_names is None: ch_names = [f"ch{i}" for i in range(n_channels)]
    features = []
    for i, c in enumerate(ch_names):
        # Initialize features to 0
        feat = {}
        for name in ["exponent", "offset"]:
            feat[name] = np.zeros((n_epochs))
        for band in bands:
            band_name = band[2]
            feat[f"{band_name}_exist"] = np.zeros((n_epochs))
            feat[f"{band_name}_freq"] = np.zeros((n_epochs))
            feat[f"{band_name}_power"] = np.zeros((n_epochs))
            feat[f"{band_name}_bw"] = np.zeros((n_epochs))
        # Check if channel is empty
        maxv, minv = np.max(data[:, :, i]), np.min(data[:, :, i])
        if maxv-minv != 0:
            for n in range(n_epochs):
                # - Filter the data
                epoch_ch = data[n, :, i]
                # - Extract aperiodic features using FOOOF
                exponent, offset, peaks = aperiodic_fooof(epoch_ch, fs=sf, fmin=freq_broad[0], fmax=freq_broad[1])
                feat[f"exponent"][n] = exponent
                feat[f"offset"][n] = offset
                aggregated_peaks = aggregate_peaks(peaks, bands)
                for band in bands:
                    band_name = band[2]
                    feat[f"{band_name}_exist"][n] = aggregated_peaks[f"{band_name}_exist"]
                    feat[f"{band_name}_freq"][n] = aggregated_peaks[f"{band_name}_freq"]
                    feat[f"{band_name}_power"][n] = aggregated_peaks[f"{band_name}_power"]
                    feat[f"{band_name}_bw"][n] = aggregated_peaks[f"{band_name}_bw"]
        # Convert to dataframe 
        feat = pd.DataFrame(feat).add_prefix(c + "_")
        features.append(feat)

    features = pd.concat(features, axis=1)
    features.index.name = "epoch"

    return features


def get_connectivity_features(data, method='coh', sf=100, ch_names=None):
    """Extract connectivity features from 30s epoched EEG data.
    data: np.array of shape (n_epochs, n_samples, n_channels)
    """
    # Bandpass filter
    freq_broad = (0.5, 45)
    bands = [
        (0.5, 4, "delta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
        (30, 45, "gamma"),
    ]

    n_epochs, n_samples, n_channels = data.shape
    data = data.transpose(0, 2, 1)  # Shape (n_epochs, n_channels, n_samples)
    # data = filter_data(data, sf, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names, sf, ch_types="eeg")
    info.set_montage(montage)
    epochs = mne.EpochsArray(data, info, verbose=False)

    lower_size = int(n_channels * (n_channels - 1) / 2)
    feature_bands = np.zeros((n_epochs, len(bands), lower_size))
    freqs = np.linspace(freq_broad[0], freq_broad[1], int((freq_broad[1] - freq_broad[0]) * 4 + 1))
    conn = spectral_connectivity_time(
        epochs, freqs, method=method, sfreq=sf, 
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

    features = pd.DataFrame(feature_bands.reshape(n_epochs, -1), columns=feature_cols).add_prefix(method + "_")
    features.index.name = "epoch"

    return features

def get_connectivity_feature_cols(ch_names=None):
    bands = [
        (0.5, 4, "delta"),
        (4, 8, "theta"),
        (8, 12, "alpha"),
        (12, 16, "sigma"),
        (16, 30, "beta"),
        (30, 45, "gamma"),
    ]
    n_channels = len(ch_names)
    feature_cols = []
    idx_lower = np.tril_indices(n_channels, k=-1)
    for i, (fmin, fmax, b) in enumerate(bands):
        feature_cols += [f"{b}_{ch_names[c1]}_{ch_names[c2]}" for c1, c2 in zip(idx_lower[0], idx_lower[1])]


def bandpower_from_psd_ndarray(
    psd,
    freqs,
    bands=[
        (0.5, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 16, "Sigma"),
        (16, 30, "Beta"),
        (30, 45, "Gamma"),
    ],
    relative=True,
):
    """Compute bandpowers in N-dimensional PSD.

    This is a NumPy-only implementation of the :py:func:`yasa.bandpower_from_psd` function,
    which supports 1-D arrays of shape (n_freqs), or N-dimensional arays (e.g. 2-D (n_chan,
    n_freqs) or 3-D (n_chan, n_epochs, n_freqs))

    .. versionadded:: 0.2.0

    Parameters
    ----------
    psd : :py:class:`numpy.ndarray`
        Power spectral density of data, in uV^2/Hz. Must be a N-D array of shape (..., n_freqs).
        See :py:func:`scipy.signal.welch` for more details.
    freqs : :py:class:`numpy.ndarray`
        Array of frequencies. Must be a 1-D array of shape (n_freqs,)
    bands : list of tuples
        List of frequency bands of interests. Each tuple must contain the lower and upper
        frequencies, as well as the band name (e.g. (0.5, 4, 'Delta')).
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    -------
    bandpowers : :py:class:`numpy.ndarray`
        Bandpower array of shape *(n_bands, ...)*.
    """
    # Type checks
    assert isinstance(bands, list), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    assert psd.shape[-1] == freqs.shape[-1], "n_freqs must be last axis of psd"

    # Extract frequencies of interest
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29"
        )
        print(msg)

    # Calculate total power
    total_power = simpson(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float64)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simpson(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        bp /= total_power
    return bp


if __name__ == "__main__":

    # Example usage
    data = np.random.randn(10, 3000, 6)
    sf = 100
    metadata = {"age": 30, "sex": 0}
    features = get_features(data, sf, ch_names=['eeg']*4 + ['eog', 'emg'], metadata=metadata)
    features = features.to_numpy()
    print(features.shape)

    features_con = get_connectivity_features(data, 'coh', sf, ch_names=["F3", "F4", "C3", "C4", "O1", "O2"])
    features_con = features_con.to_numpy()
    print(features_con.shape)
