"""Functions to segment EEG into microstates.

Based on the Microsegment toolbox for EEGlab, written by Andreas Trier Poulsen [1]_.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>

References
----------
.. [1]  Poulsen, A. T., Pedroni, A., Langer, N., &  Hansen, L. K. (2018).
        Microstate EEGlab toolbox: An introductionary guide. bioRxiv.
"""

import warnings
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.stats import zscore
from scipy.signal import find_peaks
import matplotlib as mpl
from matplotlib import pyplot as plt
import mne

__version__ = "0.4dev0"


def segment(
    data,
    n_states=4,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    normalize=False,
    min_peak_dist=2,
    max_n_peaks=10000,
    return_polarity=False,
    return_best_gev=False,
    weights=None,
    random_state=None,
    verbose=None,
):
    """Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in.
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    thresh : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    normalize : bool
        Whether to normalize (z-score) the data across time before running the
        k-means algorithm. Defaults to ``False``.
    min_peak_dist : int
        Minimum distance (in samples) between peaks in the GFP. Defaults to 2.
    max_n_peaks : int | None
        Maximum number of GFP peaks to use in the k-means algorithm. Chosen
        randomly. Set to ``None`` to use all peaks. Defaults to 10000.
    return_polarity : bool
        Whether to return the polarity of the activation.
        Defaults to ``False``.
    return_best_gev : bool
        Option to return best golbally explained variance. Defaults to False.
    weights : array-like | None
        This is an optional array which can be used to weight each sample during the
        adapted k-means algorithm. It should have the same length as the number of
        samples in the data. Defaults to ``None``.
    random_state : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.
    verbose : int | bool | None
        Controls the verbosity.

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    polarity : ndarray, shape (n_samples,)
        For each sample, the polarity (+1 or -1) of the activation on the
        currently activate map.
    gev : float
        The best global explained variance.

    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    if verbose:
        print("Finding %d microstates, using %d random intitializations" % (n_states, n_inits))

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weight_array = np.array(weights)
        if len(weights) != data.shape[1]:
            raise ValueError(
                "The `weights` array must have the same length as the number of "
                "samples in the data."
            )
        weighted_analysis = True
    else:
        weight_array = None
        weighted_analysis = False     

    if normalize:
        data = zscore(data, axis=1)

    # Find peaks in the global field power (GFP)
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    n_peaks = len(peaks)

    # Limit the number of peaks by randomly selecting them
    if max_n_peaks is not None:
        max_n_peaks = min(n_peaks, max_n_peaks)
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks, replace=False)
        peaks = peaks[chosen_peaks]

    # Cache these values for later
    gfp_sum_sq = np.sum(gfp**2)
    data_mean = np.mean(data, axis=0, dtype=np.float64)
    data_peaks = data[:, peaks]
    n_channels = data.shape[0]
    idx_samples = np.arange(data.shape[1])

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_segmentation = None
    best_polarity = None
    for _ in range(n_inits):
        maps = _mod_kmeans(
            data_peaks,
            n_states,
            n_inits,
            max_iter,
            thresh,
            random_state,
            verbose,
            weighted_analysis,
            weight_array,
        )
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        # assigned_activations = np.choose(segmentations, activation)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates. gfp * corr(data, assigned map) is computed from
        # the activation instead of _corr_vectors, avoiding copies of the full data:
        # gfp * corr = (x.m - n * mean(x) * mean(m)) / (sqrt(n) * ||m - mean(m)||)
        maps_mean = np.mean(maps, axis=1, dtype=np.float64)
        maps_cnorm = np.linalg.norm(maps - maps_mean[:, None], axis=1)
        gfp_corr = (activation[segmentation, idx_samples].astype(np.float64)
                    - n_channels * data_mean * maps_mean[segmentation])
        gfp_corr /= np.sqrt(n_channels) * maps_cnorm[segmentation]
        gev = np.sum(gfp_corr ** 2) / gfp_sum_sq
        if verbose: print("GEV of found microstates: %f" % gev)
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation
            best_polarity = np.sign(np.choose(segmentation, activation))

    output = [best_maps, best_segmentation]
    if return_polarity:
        output.append(best_polarity)
    if return_best_gev:
        output.append(best_gev)
    return tuple(output)


def segment_predict(cluster_centers, data, return_polarity=False):
    """Predict microstate segmentation for new data.

    Parameters
    ----------
    cluster_centers : ndarray, shape (n_states, n_channels)
        The topographic maps of the microstates.
    data : ndarray, shape (n_channels, n_samples)
        The data to segment.
    return_polarity : bool
        Whether to return the polarity of the activation.
        Defaults to ``False``.

    Returns
    -------
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    polarity : ndarray, shape (n_samples,)
        For each sample, the polarity (+1 or -1) of the activation on the
        currently activate map.
    """
    activation = cluster_centers.dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    polarity = np.sign(np.choose(segmentation, activation))

    output = [segmentation]
    if return_polarity:
        output.append(polarity)
    return tuple(output)


def _mod_kmeans(
    data,
    n_states=4,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    random_state=None,
    verbose=None,
    weighted_analysis=False,
    weight_array=None,
):
    """Performs the modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    if weighted_analysis:
        data_sum_sq = np.sum(data**2 * weight_array)
    else:
        data_sum_sq = np.sum(data**2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = segmentation == state
            if np.sum(idx) == 0:
                warnings.warn("Some microstates are never activated")
                maps[state] = 0
                continue
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        if weighted_analysis:
            act_sum_sq = np.sum(
                np.sum(maps[segmentation].T * data, axis=0) ** 2 * weight_array
            )
        else:
            act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            if verbose: print("Converged at %d iterations." % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn("Modified K-means algorithm failed to converge.")

    return maps


def _corr_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)


def plot_segmentation(segmentation, data, times, polarity=None, show=True, ax=None):
    """Plot a microstate segmentation.

    Parameters
    ----------
    segmentation : list of int
        For each sample in time, the index of the state to which the sample has
        been assigned.
    data : ndarray, shape (n_channels, n_samples)
        The data on which the microstates were computed.
    times : list of float
        The time-stamp for each sample.
    polarity : list of int | None
        For each sample in time, the polarity (+1 or -1) of the activation.
    show : bool
        Show figure if ``True``.
    """
    gfp = np.std(data, axis=0)
    if polarity is not None:
        gfp *= polarity

    n_states = len(np.unique(segmentation))
    if ax is None:
        plt.figure(figsize=(6 * np.ptp(times), 2))
        ax = plt.gca()
    cmap = plt.get_cmap("plasma", n_states) # plt.cm.get_cmap("plasma", n_states)
    ax.plot(times, gfp, color="black", linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        ax.fill_between(times, gfp, color=color, where=(segmentation == state))
    norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_title("Segmentation into %d microstates" % n_states)
    plt.autoscale(tight=True)
    plt.tight_layout()
    if show:
        plt.show()


def plot_maps(maps, info, show=True):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_maps, n_channels)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    show : bool
        Show figure if ``True``.
    """
    assert len(maps) != 1, "Only one map found, cannot plot"
    fig, axes = plt.subplots(1, len(maps), figsize=(2 * len(maps), 2))
    for i, (ax, map) in enumerate(zip(axes, maps)):
        mne.viz.plot_topomap(map, info, axes=ax, show=False)
        ax.set_title("Microstate %d" % (i + 1))
    plt.tight_layout()
    if show:
        plt.show()


def extract_gfp_peaks(data, min_peak_distance=2):
    """Extract GFP peaks from input data.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        The data to extract GFP peaks from.
    min_peak_distance : int
        Required minimal horizontal distance (>= 1) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks. Default to 2.

    Returns
    -------
    peaks : array of shape (n_picks)
        The indices when peaks occur.
    """
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    return peaks


def compute_parameters(data, labels, cluster_centers, sfreq=1, norm_gfp: bool = True, return_dist: bool = False):
        """Compute microstate parameters.

        .. warning::

            When working with `~mne.Epochs`, this method will put together segments of
            all epochs. This could lead to wrong interpretation especially on state
            durations. To avoid this behaviour, make sure to set the ``reject_edges``
            parameter to ``True`` when creating the segmentation.

        Parameters
        ----------
        norm_gfp : bool
            If True, the :term:`global field power` (GFP) is normalized.
        return_dist : bool
            If True, return the parameters distributions.

        Returns
        -------
        dict : dict
            Dictionaries containing microstate parameters as key/value pairs.
            Keys are named as follow: ``'{microstate name}_{parameter name}'``.

            Available parameters are listed below:

            * ``mean_corr``: Mean correlation value for each time point assigned to a
              given state.
            * ``gev``: Global explained variance expressed by a given state.
              It is the sum of global explained variance values of each time point
              assigned to a given state.
            * ``timecov``: Time coverage, the proportion of time during which
              a given state is active. This metric is expressed as a ratio.
            * ``meandurs``: Mean durations of segments assigned to a given
              state. This metric is expressed in seconds (s).
            * ``occurrences``: Occurrences per second, the mean number of
              segment assigned to a given state per second. This metrics is expressed
              in segment per second.
            * ``dist_corr`` (req. ``return_dist=True``): Distribution of
              correlations values of each time point assigned to a given state.
            * ``dist_gev`` (req. ``return_dist=True``): Distribution of global
              explained variances values of each time point assigned to a given state.
            * ``dist_durs`` (req. ``return_dist=True``): Distribution of
              durations of each segments assigned to a given state. Each value is
              expressed in seconds (s).
        """

        gfp = np.std(data, axis=0)
        if norm_gfp:
            labeled = np.argwhere(labels != -1)  # ignore unlabeled segments
            gfp /= np.linalg.norm(gfp[labeled])  # normalize

        segments = [(s, list(group)) for s, group in groupby(labels)]

        params = dict()
        for s, state in enumerate(cluster_centers):
            arg_where = np.argwhere(labels == s)
            if len(arg_where) != 0:
                labeled_tp = data.T[arg_where][:, 0, :].T
                labeled_gfp = gfp[arg_where][:, 0]
                state_array = np.array([state] * len(arg_where)).transpose()

                dist_corr = _corr_vectors(state_array, labeled_tp)
                params[f"{s}_mean_corr"] = np.mean(np.abs(dist_corr))
                dist_gev = (labeled_gfp * dist_corr) ** 2 / np.sum(gfp**2)
                params[f"{s}_gev"] = np.sum(dist_gev)

                s_segments = np.array([len(group) for s_, group in segments if s_ == s])
                occurrences = len(s_segments) / len(np.where(labels != -1)[0]) * sfreq
                params[f"{s}_occurrences"] = occurrences

                timecov = np.sum(s_segments) / len(np.where(labels != -1)[0])
                params[f"{s}_timecov"] = timecov

                dist_durs = s_segments / sfreq
                params[f"{s}_meandurs"] = np.mean(dist_durs)

                if return_dist:
                    params[f"{s}_dist_corr"] = dist_corr
                    params[f"{s}_dist_gev"] = dist_gev
                    params[f"{s}_dist_durs"] = dist_durs

            else:
                params[f"{s}_mean_corr"] = 0.0
                params[f"{s}_gev"] = 0.0
                params[f"{s}_timecov"] = 0.0
                params[f"{s}_meandurs"] = 0.0
                params[f"{s}_occurrences"] = 0.0

                if return_dist:
                    params[f"{s}_dist_corr"] = np.array([], dtype=float)
                    params[f"{s}_dist_gev"] = np.array([], dtype=float)
                    params[f"{s}_dist_durs"] = np.array([], dtype=float)

        params["unlabeled"] = len(np.argwhere(labels == -1)) / len(gfp)
        return params


def transition_matrix(seqs, n_states=None, ignore_repetitions=True):
    """Create a state-transition matrix from state sequences.

    .. versionadded:: 0.1.9

    Parameters
    ----------
    seqs : array orlist of arrays
        State sequence. The dtype of ``seq`` must be integer
        (e.g. [0, 2, 2, 1, 1, 1, ...]). 
    n_states : int | None
        Number of states. If ``None``, the number of states is
        inferred from the data.

    Returns
    -------
    counts : :py:class:`pandas.DataFrame`
        Counts transition matrix (number of transitions from stage A to
        stage B). The pre-transition states are the rows and the
        post-transition states are the columns.
    probs : :py:class:`pandas.DataFrame`
        Conditional probability transition matrix, i.e.
        given that current state is A, what is the probability that
        the next state is B.
        ``probs`` is a `right stochastic matrix
        <https://en.wikipedia.org/wiki/Stochastic_matrix>`_,
        i.e. each row sums to 1.

    Examples
    --------
    >>> import numpy as np
    >>> a = [0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 2, 3, 3, 0, 2, 2, 1, 2, 2, 3, 3]
    >>> counts, probs = transition_matrix(a)
    >>> counts
           0  1  2  3
    Stage
    0      2  2  1  0
    1      1  1  2  0
    2      0  1  3  3
    3      1  0  1  3

    >>> probs.round(2)
              0     1     2     3
    Stage
    0      0.40  0.40  0.20  0.00
    1      0.25  0.25  0.50  0.00
    2      0.00  0.14  0.43  0.43
    3      0.20  0.00  0.20  0.60

    Finally, we can plot the transition matrix using :py:func:`seaborn.heatmap`

    .. plot::

        >>> import numpy as np
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> # Calculate probability matrix
        >>> a = [1, 1, 1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0]
        >>> _, probs = transition_matrix(a)
        >>> # Start the plot
        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,
        ...                                 figsize=(5, 5))
        >>> sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
        ...             cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
        ...             cbar_kws={"orientation": "horizontal", "fraction": 0.1,
        ...                       "label": "Transition probability"})
        >>> ax.set_xlabel("To state")
        >>> ax.xaxis.tick_top()
        >>> ax.set_ylabel("From state")
        >>> ax.xaxis.set_label_position('top')
    """
    assert isinstance(seqs, (list, np.ndarray)), "seqs must be a list or array"
    if isinstance(seqs[0], int) or isinstance(seqs[0], np.longlong):
        seqs = [seqs]
    # Number of states
    unique = np.unique(np.concatenate(seqs))
    n = n_states if n_states is not None else len(unique)
    if n != len(unique):
        # warnings.warn(
        #     "Number of states (%d) does not match the number of unique states "
        #     "(%d). Using the number of unique states." % (n, len(unique))
        # )
        unique = np.arange(n)
    # Integer transition counts
    counts = np.zeros((n, n), dtype=int)
    for x in seqs:
        # ignore transition to itself (i.e. 1 -> 1)
        if ignore_repetitions:
            x = [s for s, _ in groupby(x)]
        # number of transitions
        for i, j in zip(x, x[1:]):
            if i == -1 or j == -1:
                continue  # ignore unlabeled
            counts[i, j] += 1
    # Conditional probabilities
    # probs = counts / counts.sum(axis=-1, keepdims=True) # cause division by zero 
    row_sums = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, row_sums, where=row_sums != 0, out=np.zeros_like(counts, dtype=float))
    # Convert to a Pandas DataFrame
    counts = pd.DataFrame(counts, index=unique, columns=unique)
    probs = pd.DataFrame(probs, index=unique, columns=unique)
    counts.index.name = "From State"
    probs.index.name = "From State"
    counts.columns.name = "To State"
    probs.columns.name = "To State"
    return counts, probs


if __name__ == "__main__":

    import seaborn as sns
    from mne.datasets import sample

    # Load MNE sample dataset
    fname = sample.data_path() / 'MEG/sample/sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)

    # Always use an average EEG reference when doing microstate analysis
    raw.set_eeg_reference('average')

    # Highpass filter the data a little bit
    raw.filter(0.2, None)

    # Selecting the sensor types to use in the analysis. In this example, we
    # use only EEG channels
    raw.pick_types(meg=False, eeg=True)

    # Segment the data into 6 microstates
    maps, segmentation = segment(raw.get_data(), n_states=6)

    # Plot the topographic maps of the found microstates
    plot_maps(maps, raw.info)

    # Plot the segmentation of the first 500 samples
    plot_segmentation(segmentation[:500], raw.get_data()[:, :500], raw.times[:500])

    ## Calculate probability matrix
    # a = [
    #     [0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 2, 3, 3, 0, 2, 2, 1, 2, 2, 3, 3],
    #     [0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 2, 3, 3, 0, 2, 2, 1, 2, 2, 3, 3]
    # ]
    # _, probs = transition_matrix(a, ignore_repetitions=True)
    # # Start the plot
    # grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    # f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    # sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
    #             cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
    #             cbar_kws={"orientation": "horizontal", "fraction": 0.1,
    #                       "label": "Transition probability"})
    # ax.set_xlabel("To state")
    # ax.xaxis.tick_top()
    # ax.set_ylabel("From state")
    # ax.xaxis.set_label_position('top')
    # plt.show()
    