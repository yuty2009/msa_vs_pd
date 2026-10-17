"""Fold-contained learning, alignment and aggregation of data-driven PSG states."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from common.microstate import segment, segment_predict


def fit_states(data_list, n_states, random_state=None):
    """Estimate state centroids from the epochs of the given subjects.

    Parameters
    ----------
    data_list : list of ndarray, each of shape (n_epochs, ...)
        Epoch-level features of the subjects used for state learning.
    n_states : int
        Number of states.
    random_state : int | None
        Seed of the modified k-means initialisation.

    Returns
    -------
    cluster_centers : ndarray, shape (n_states, n_features)
    """
    data_all = np.concatenate([d.reshape(d.shape[0], -1) for d in data_list], axis=0)
    cluster_centers, _ = segment(data_all.T, n_states=n_states, random_state=random_state)
    return cluster_centers


def align_states(cluster_centers, ref_centers):
    """Reorder states to match reference states (Hungarian algorithm).

    States are matched one-to-one by maximising the absolute Pearson
    correlation between centroids. Absolute values are used because state
    assignment (``segment_predict``) ignores centroid polarity.

    Returns
    -------
    aligned_centers : ndarray, shape (n_states, n_features)
        ``cluster_centers`` reordered so that row j corresponds to reference state j.
    matched_corr : ndarray, shape (n_states,)
        Absolute correlation between each reference state and its matched state.
    """
    n_states = len(ref_centers)
    corr = np.corrcoef(ref_centers, cluster_centers)[:n_states, n_states:]
    corr = np.nan_to_num(np.abs(corr))  # an empty state has a zero centroid
    row, col = linear_sum_assignment(-corr)
    return cluster_centers[col], corr[row, col]


def aggregate_states(data_list, cluster_centers):
    """Average epoch-level features within each state for every subject.

    States without any epoch in a subject are filled with zeros.

    Returns
    -------
    features : ndarray, shape (n_subjects, n_states, n_features)
    """
    n_states = len(cluster_centers)
    features = []
    for d in data_list:
        x = d.reshape(d.shape[0], -1)  # n_epochs x n_features
        cluster_labels = segment_predict(cluster_centers, x.T)[0]
        cluster_means = np.zeros((n_states, x.shape[1]))
        for j in range(n_states):
            if np.any(cluster_labels == j):
                cluster_means[j] = x[cluster_labels == j].mean(axis=0)
        features.append(cluster_means)
    return np.array(features)
