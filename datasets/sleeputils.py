"""Lightweight utilities shared by the sleep dataset readers.

Dependencies are limited to numpy, scipy and pyedflib (mne is imported only
when an EDF file cannot be opened by pyedflib).
"""

from fractions import Fraction
import numpy as np
import scipy.signal as sp_sig
import pyedflib


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def resample(signal, signal_frequency, target_frequency, axis=-1):
    """Resample a signal with polyphase FIR anti-aliasing filtering.

    The output length along ``axis`` is ``int(n_samples / signal_frequency *
    target_frequency)``, so that epoch boundaries stay aligned with time.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal_frequency = float(signal_frequency)
    target_frequency = float(target_frequency)
    n_samples = signal.shape[axis]
    new_samples = int(n_samples / signal_frequency * target_frequency)
    if signal_frequency == target_frequency:
        return signal

    ratio = Fraction(target_frequency / signal_frequency).limit_denominator(1000)
    out = sp_sig.resample_poly(signal, ratio.numerator, ratio.denominator, axis=axis, padtype='line')
    # resample_poly yields ceil(n * up / down) samples; trim (or edge-pad when the
    # ratio was approximated) to the exact expected length
    if out.shape[axis] >= new_samples:
        out = np.take(out, np.arange(new_samples), axis=axis)
    else:
        pad = [(0, 0)] * out.ndim
        pad[axis] = (0, new_samples - out.shape[axis])
        out = np.pad(out, pad, mode='edge')
    return out



# ---------------------------------------------------------------------------
# EDF reading
# ---------------------------------------------------------------------------

class EdfFile:
    """Read EDF channels at their native sampling rates.

    pyedflib is used whenever it can open the file. Files it rejects (e.g. a
    non-compliant header, as in DREAMS) are read with mne instead, one channel
    at a time so that mne does not upsample all channels to a common rate.
    Either way, signals are returned in the physical units of the EDF header
    (typically uV), and channels may be given by index or by label.
    """

    def __init__(self, fname):
        self.fname = fname
        self._edf, self._raw, self._cache = None, None, {}
        try:
            self._edf = pyedflib.EdfReader(fname)
        except OSError:
            import mne
            self._raw = mne.io.read_raw_edf(fname, preload=False, verbose='ERROR')

        if self._edf is not None:
            self._labels = self._edf.getSignalLabels()
            self.start_datetime = self._edf.getStartdatetime()
            self.file_duration = self._edf.getFileDuration()
        else:
            self._labels = list(self._raw.ch_names)
            meas_date = self._raw.info['meas_date']
            self.start_datetime = None if meas_date is None else meas_date.replace(tzinfo=None)
            self.file_duration = self._raw.n_times / self._raw.info['sfreq']
        # public copy; callers may rename it, reads still use the original labels
        self.ch_names = list(self._labels)

    def _index(self, ch):
        return self.ch_names.index(ch) if isinstance(ch, str) else int(ch)

    def _read_mne(self, idx):
        if idx not in self._cache:
            import mne
            name = self._labels[idx]
            raw = mne.io.read_raw_edf(self.fname, include=[name], preload=True, verbose='ERROR')
            # undo the unit gain mne applied (e.g. 1e-6 for 'uV'; 1 for units it does
            # not recognise, such as the upper-case 'UV' in DREAMS)
            gain = raw._raw_extras[0]['units'][0]
            self._cache[idx] = (raw.get_data()[0] / gain, raw.info['sfreq'])
        return self._cache[idx]

    def sampling_rate(self, ch):
        idx = self._index(ch)
        if self._edf is not None:
            return self._edf.getSampleFrequency(idx)
        return self._read_mne(idx)[1]

    def read_signal(self, ch):
        idx = self._index(ch)
        if self._edf is not None:
            return self._edf.readSignal(idx)
        return self._read_mne(idx)[0]

    def close(self):
        if self._edf is not None:
            self._edf.close()
        self._edf, self._raw, self._cache = None, None, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()



# ---------------------------------------------------------------------------
# Preprocessed npz loading
# ---------------------------------------------------------------------------

def load_npz_file(npz_file, n_channels=None, nan=None, extra_keys=('ch_label', 'keep_idx', 'edf_path')):
    """Load data and labels from a npz file.

    n_channels: keep only the first n channels of 3D data (n_epochs, n_samples, n_channels).
    nan: None to leave NaNs untouched, 'zero' to replace them with 0, 'raise' to reject them.
    Returns (data, labels, sampling_rate, *[f[k] for k in extra_keys]).
    """
    with np.load(npz_file) as f:
        data = f["x"]
        if n_channels is not None and data.ndim > 2:
            data = data[..., :n_channels]
        if nan == 'zero':
            data[np.isnan(data)] = 0.
        elif nan == 'raise' and np.isnan(data).any():
            raise ValueError(f"Found NaN in {npz_file}.")
        labels = f["y"]
        sampling_rate = f["fs"]
        extras = [f[k] for k in extra_keys]
    return (data, labels, sampling_rate, *extras)


def load_npz_list_files(npz_files, load_fn=load_npz_file, verbose=True, **load_kwargs):
    """Load data and labels from a list of npz files.

    load_fn(npz_file, **load_kwargs) must return (data, labels, sampling_rate, *extras);
    with the default load_fn, load_kwargs are the options of load_npz_file.
    Returns (data, labels, *extras), each a list with one entry per file.
    """
    data = []
    labels = []
    extras = None
    fs = None
    for npz_f in npz_files:
        if verbose:
            print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate, *tmp_extras = load_fn(npz_f, **load_kwargs)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # # Reshape the data to match the input of the model - conv1d
        if len(tmp_data.shape) == 2:
            tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        data.append(tmp_data.astype(np.float32))
        labels.append(tmp_labels.astype(np.int32))
        if extras is None:
            extras = [[] for _ in tmp_extras]
        for lst, item in zip(extras, tmp_extras):
            lst.append(item)

    return (data, labels, *(extras or []))
