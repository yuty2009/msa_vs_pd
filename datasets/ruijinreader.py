
import os
import mne
import glob
import natsort
import pyedflib
import numpy as np
import pandas as pd
import sys; sys.path.append(os.path.dirname(__file__))
from sleeputils import resample, get_features, get_features_yasa, get_aperiodic_features, get_connectivity_features

stage_dict = {
    'W': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'R': 4,
    'M': 5,
    '?': 6,
}
diagnosis_dict = {
    'HC': 0,
    'MSA': 1,
    'PD': 2,
    'RBD': 3,
}


def load_eegdata_ruijin(psg_fname, ann_fname, select_ch=['C4-A1', 'C4-M1', 'C4', 'C3-A2', 'C3-M2', 'C3'], target_fs=100.):
    if ann_fname is not None:
        labels = np.loadtxt(ann_fname, dtype=str).ravel()
        labels = np.array([stage_dict[l] for l in labels])
    else:
        labels = np.array([])

    raw, psg_f = None, None
    try:
        psg_f = pyedflib.EdfReader(psg_fname)
        # start_datetime = psg_f.getStartdatetime()
    except Exception as e:
        try:
            raw = mne.io.read_raw_edf(psg_fname, preload=True)
        except Exception as e1:
            return None

    if raw is not None:
        ch_names = raw.info['ch_names'] # 
    elif psg_f is not None:
        ch_names = psg_f.getSignalLabels() # 

    select_ch_idx = -1
    for s in range(len(select_ch)):
        if select_ch[s] in ch_names:
            select_ch_idx = ch_names.index(select_ch[s])
            break
    
    if select_ch_idx == -1:
        return None

    ch_selected = ch_names[select_ch_idx]
    if raw is not None:
        sampling_rate = raw.info['sfreq'] # 
        signals = raw.get_data(picks=ch_selected)[0] # 
    elif psg_f is not None:
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx) # raw.info['sfreq'] # 
        signals = psg_f.readSignal(select_ch_idx) # raw.get_data(picks=ch_selected)[0] #

    if sampling_rate != target_fs:
        signals = resample(signals, sampling_rate, target_fs)
    
    epoch_duration = 30
    n_epoch_samples = int(epoch_duration * target_fs)
    n_epochs = len(signals) // n_epoch_samples
    signals = signals[:n_epochs * n_epoch_samples].reshape(n_epochs, n_epoch_samples)

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    print(x.shape)
    print(y.shape)
    
    if len(y) > 0:
        if len(x) != len(y):
            print("Length mismatch")
            return None
        y = y[:len(x)]
        y_full = y.copy()
        keep_idx = np.arange(len(y_full))
    else:
        y = np.array([6] * len(x), dtype=np.int32)
        y_full = y.copy()
        keep_idx = np.arange(len(y_full))

    # Remove movement and unknown
    remove_idx = np.where(y >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]
        keep_idx = keep_idx[select_idx]

    remove_idx = np.where(y_full >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(y_full)), remove_idx)
        y_full = y_full[select_idx]

    data_dict = {
        "x": x,
        "y": y,
        "y_full": y_full,
        "keep_idx": keep_idx,
        "fs": target_fs,
        "ch_label": ch_selected,
        "edf_path": psg_fname,
    }

    return data_dict

def load_eegdata_ruijin_multichannel(
        psg_fname, ann_fname, 
        eeg=['F3', 'F4', 'C3', 'C4', 'O1', 'O2'], 
        eog=['EOG', 'HEOG', 'VEOG', 'LOC', 'ROC', 'E1', 'E2'], 
        emg=['Chin', 'EMG'],
        ecg=['ECG', 'EKG'],
        target_fs=100.
    ):
    if ann_fname is not None:
        labels = np.loadtxt(ann_fname, dtype=str).ravel()
        labels = np.array([stage_dict[l] for l in labels])
    else:
        labels = np.array([])

    raw, psg_f = None, None
    try:
        psg_f = pyedflib.EdfReader(psg_fname)
        # start_datetime = psg_f.getStartdatetime()
    except Exception as e:
        try:
            raw = mne.io.read_raw_edf(psg_fname, preload=True)
        except Exception as e1:
            return None

    if raw is not None:
        ch_names = raw.info['ch_names'] # 
    elif psg_f is not None:
        ch_names = psg_f.getSignalLabels() # 
    select_ch_idx = -np.ones((len(eeg)+3), dtype=int) # EEG(6) + EOG(1) + EMG(1) + ECG(1)
    for s in range(len(eeg)):
        for ch in ch_names:
            if ch.startswith(eeg[s]):
                select_ch_idx[s] = ch_names.index(ch)
                break
    for s in range(len(eog)):
        for ch in ch_names:
            if ch.startswith(eog[s]):
                select_ch_idx[-3] = ch_names.index(ch)
                break
        if select_ch_idx[-3] != -1:
            break
    for s in range(len(emg)):
        for ch in ch_names:
            if ch.startswith(emg[s]):
                select_ch_idx[-2] = ch_names.index(ch)
                break
        if select_ch_idx[-2] != -1:
            break
    for s in range(len(ecg)):
        for ch in ch_names:
            if ch.startswith(ecg[s]):
                select_ch_idx[-1] = ch_names.index(ch)
                break
        if select_ch_idx[-1] != -1:
            break

    if -1 in select_ch_idx[:-3]:
        # raise Exception("Some EEG channels are missing.")
        pass

    n_channels = len(select_ch_idx)
    n_totalpoints = 1
    signals, ch_selected, ch_mask = [], [], []
    for s in range(n_channels):
        if select_ch_idx[s] == -1:
            ch_mask.append(False)
            signal_ch = np.zeros((1))
        else:
            ch_i = ch_names[select_ch_idx[s]]
            ch_selected.append(ch_i)
            ch_mask.append(True)
            if raw is not None:
                sampling_rate = raw.info['sfreq'] # 
                signal_ch = raw.get_data(picks=ch_i)[0] # 
            elif psg_f is not None:
                sampling_rate = psg_f.getSampleFrequency(select_ch_idx[s]) # raw.info['sfreq'] # 
                signal_ch = psg_f.readSignal(select_ch_idx[s]) # raw.get_data(picks=ch_selected)[0] # 
            if sampling_rate != target_fs:
                signal_ch = resample(signal_ch, sampling_rate, target_fs)
            n_totalpoints = len(signal_ch)
        signals.append(signal_ch)
    for s in range(n_channels):
        if select_ch_idx[s] == -1:
            signals[s] = np.zeros((n_totalpoints))
    signals = np.stack(signals, axis=0)
    
    epoch_duration = 30
    n_epochlen = int(epoch_duration * target_fs)
    n_epochs = signals.shape[1] // n_epochlen
    signals = signals[:, :n_epochs * n_epochlen].reshape(n_channels, n_epochs, n_epochlen)
    signals = signals.transpose([1, 2, 0]) # n_epochs, n_epochlen, n_channels

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    print(x.shape)
    print(y.shape)

    if len(y) > 0:
        if len(x) != len(y):
            print("Length mismatch")
            return None
        y = y[:len(x)]
        y_full = y.copy()
        keep_idx = np.arange(len(y_full))
    else:
        y = np.array([6] * len(x), dtype=np.int32)
        y_full = y.copy()
        keep_idx = np.arange(len(y_full))

    # Remove movement and unknown
    # remove_idx = np.where(y >= 5)[0]
    # if len(remove_idx) > 0:
    #     select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
    #     x = x[select_idx]
    #     y = y[select_idx]
    #     keep_idx = keep_idx[select_idx]

    # remove_idx = np.where(y_full >= 5)[0]
    # if len(remove_idx) > 0:
    #     select_idx = np.setdiff1d(np.arange(len(y_full)), remove_idx)
    #     y_full = y_full[select_idx]

    data_dict = {
        "x": x,
        "y": y,
        "y_full": y_full,
        "keep_idx": keep_idx,
        "fs": target_fs,
        "ch_label": ch_selected,
        "ch_mask": ch_mask,
        "edf_path": psg_fname,
    }

    return data_dict

def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        if len(f['x'].shape) > 2:
            data = f['x'] # [:, :, :6] # use EEG only
        else:
            data = f['x']
        assert np.isnan(data).any() == False, f"Found NaN in {npz_file}."
        labels = f["y"]
        sampling_rate = f["fs"]
        ch_label = f["ch_label"]
        keep_idx = f['keep_idx']
        edf_path = f['edf_path']
    return data, labels, sampling_rate, ch_label, keep_idx, edf_path

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    ch_labels = []
    keep_idxs = []
    edf_paths = []
    fs = None
    for npz_f in npz_files:
        # print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate, ch_label, keep_idx, edf_path = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # # Reshape the data to match the input of the model - conv2d
        if len(tmp_data.shape) == 2:
            tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)
        ch_labels.append(ch_label)
        keep_idxs.append(keep_idx)
        edf_paths.append(edf_path)

    return data, labels, ch_labels, keep_idxs, edf_paths

def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs, _, _, _ = load_npz_file(npz_f)
    return data, labels

def load_dataset_preprocessed(datapath, subsets=None, subjects=None, extra_info=False):
    npzfiles, subjects = load_npzlist_preprocessed(datapath, subsets=subsets, subjects=subjects)
    data, labels, ch_labels, keep_idxs, edf_paths = load_npz_list_files(npzfiles)
    if extra_info:
        return data, labels, subjects, ch_labels, keep_idxs, edf_paths
    else:
        return data, labels, subjects
    
def load_npzlist_preprocessed(datapath, subsets=None, subjects=None):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    all_subjects = [os.path.basename(npz_f)[:-8] for npz_f in npzfiles]
    if subjects is not None:
        select_idx = [i for i in range(len(all_subjects)) if all_subjects[i] in subjects]
        npzfiles = [npzfiles[i] for i in select_idx]
        subjects = [all_subjects[i] for i in select_idx]
    else:
        subjects = all_subjects
    return npzfiles, subjects

def load_hypnogram_preprocessed(datapath, subsets=None, subjects=None):
    if isinstance(subsets, str):
        subsets = [subsets]
    hypfiles = glob.glob(os.path.join(datapath, "*.txt"))
    hypfiles.sort()
    all_subjects = [os.path.basename(hyp_f)[:-8] for hyp_f in hypfiles]
    if subjects is not None:
        select_idx = [i for i in range(len(all_subjects)) if all_subjects[i] in subjects]
        hypfiles = [hypfiles[i] for i in select_idx]
        subjects = [all_subjects[i] for i in select_idx]
    else:
        subjects = all_subjects
    hypnograms = []
    for hyp_f in hypfiles:
        hypno = pd.read_csv(hyp_f).squeeze("columns")
        hypnograms.append(hypno.to_numpy())
    return hypnograms, subjects


if __name__ == '__main__':

    datapath = '/home/yuty2009/data/eegdata/sleep/ruijin/'
    # savepath = os.path.join(datapath, 'processed/')
    savepath = os.path.join(datapath, 'features_yasa/')
    os.makedirs(savepath, exist_ok=True)

    subsets = ['HC', 'MSA', 'PD', 'RBD']

    psg_files = []
    for subset in subsets:
        subdirs = os.listdir(os.path.join(datapath, subset))
        subject_dirs = []
        for subdir in subdirs:
            subject_dir = os.path.join(datapath, subset, subdir)
            if os.path.isdir(subject_dir):
                subject_dirs.append(subject_dir)
        psg_files_subset = []
        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)
            psg_files_subject = subject_dir + '/' + subject + '_PSG.edf'
            psg_files_subset.append(psg_files_subject)
        psg_files.extend(psg_files_subset)
    psg_files = natsort.natsorted(psg_files)
                
    subjects = []
    errorlog = []
    annotations_stage = []
    info_columns = ['subid', 'label', 'diagnosis', 'channels']
    subject_info = pd.DataFrame(columns=info_columns)
    for psg_f in psg_files:
        subject = os.path.basename(psg_f)[:-len("_PSG.edf")]
        subjects.append(subject.lower())
        print("Processing {} ...".format(psg_f))
        ann_f = psg_f.replace('PSG.edf', 'HYP.TXT')
        if not os.path.exists(psg_f): continue
        if not os.path.exists(ann_f): ann_f = None
        # data_dict = load_eegdata_ruijin(psg_f, ann_f)
        data_dict = load_eegdata_ruijin_multichannel(psg_f, None)
        # npz_f = os.path.join(datapath+'processed/', os.path.basename(psg_f)[:-4]+'.npz')
        if data_dict is None:
            errorlog.append(psg_f)
            continue
        if len(data_dict["y"]) < 60:
            errorlog.append(psg_f)
            continue
        
        sub_diagnosis = subject[:-4]
        sub_label = diagnosis_dict[sub_diagnosis]
        sub_channels = ','.join(data_dict["ch_label"])
        sub_info = [subject, sub_label, sub_diagnosis, sub_channels]
        subject_info.loc[len(subject_info)] = sub_info
        annotations_stage.append(data_dict["y"])
        data_f = os.path.join(savepath, os.path.basename(psg_f)[:-4]+'.npz')
        np.savez(data_f, **data_dict)
        label_f = os.path.join(savepath, os.path.basename(psg_f)[:-4]+'.txt')
        np.savetxt(label_f, data_dict["y"], fmt='%d', delimiter='\n')
        print("Saved {}.".format(data_f))
        
    ann_fp = open(savepath+'/annotations.txt', 'w')
    for row in annotations_stage:
        hypno_str = ','.join([str(d) for d in row])
        ann_fp.write(hypno_str+'\n')
    ann_fp.close()
    subject_info.to_csv(savepath+'/subject_labels.csv', index=False)

    if len(errorlog) > 0:
        csv_f = os.path.join(savepath, 'errorlog.csv')
        pd.DataFrame(errorlog).to_csv(csv_f, index=False, header=False)
