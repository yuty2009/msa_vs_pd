"""Incremental value of dynamic PSG features over routinely available PSG-time variables.

Two nested models are compared on identical partitions:
    Reference   : age, sex, AHI and the 46 sleep macro features
    Incremental : reference + dynamic Top-20 features

Disease duration and non-motor symptom scales (e.g. SCOPA-AUT, NMSQ, RBDSQ) are not used as
predictors: autonomic and other non-motor features are part of the clinical diagnostic
criteria for MSA, and disease duration at the time of PSG is not informative in the intended
setting of early, diagnostically uncertain parkinsonism. ``load_clinical`` also returns them
for between-group comparison only.

Cross-validation uses the same ten five-fold partitions as the main analysis. In each
training fold the dynamic features are computed with the fold-specific states and the
fold-specific selected features of the main analysis, so that held-out subjects are not
used for state learning or feature selection. The final models are trained on the whole
development cohort and evaluated on the validation cohort.
"""
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


REFERENCE_CLINICAL = ['Age', 'Male', 'AHI']  # predictors in the reference model

CLINICAL_COLS = {
    'Age': 'Age',
    'Sex': 'Male',
    '病程（年）': 'Disease duration',
    'AHI': 'AHI',
    '(SCOPA-AUT)帕金森病自主神经功能评定量表 ': 'SCOPA-AUT',
    '（RBDSQ）RBD筛查量表': 'RBDSQ',
}


def _parse_duration(s):
    """Disease duration such as '1年7月' or '7月' -> years."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    y = re.search(r'([\d.]+)\s*年', s)
    m = re.search(r'([\d.]+)\s*个?月', s)
    if not (y or m):
        return np.nan
    return (float(y.group(1)) if y else 0.0) + (float(m.group(1)) / 12 if m else 0.0)


def load_clinical(xlsx_file, subjects):
    """Clinical variables for the given subjects (rows in the same order); missing values stay NaN."""
    df = pd.read_excel(xlsx_file)
    df = df.drop_duplicates('Sub_ID').set_index('Sub_ID').reindex(subjects)
    out = pd.DataFrame(index=subjects)
    out['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    out['Male'] = df['Sex'].map({'M': 1.0, 'F': 0.0})
    out['Disease duration'] = df['病程（年）'].map(_parse_duration)
    for col, name in list(CLINICAL_COLS.items())[3:]:
        out[name] = pd.to_numeric(df[col], errors='coerce')
    out['NMSQ'] = pd.to_numeric(df['（NMSQ）帕金森非运动症状筛查量表'], errors='coerce')
    return out


def _sleep_statistics(hyp, sf_hyp=1 / 30):
    """Port of the legacy ``yasa.sleep_statistics(hypno, sf_hyp)`` (21 statistics).

    The integer-array function was deprecated in YASA 0.8 and the ``yasa.Hypnogram`` method
    returns a different set of statistics, so it is reimplemented here to give identical
    features regardless of the installed YASA version.
    """
    stages = {'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    st = {'TIB': len(hyp)}
    idx_sleep = np.flatnonzero(hyp > 0)
    has_sleep = idx_sleep.size > 0
    hyp_s = hyp[idx_sleep[0]:idx_sleep[-1] + 1] if has_sleep else hyp[:0]
    st['SPT'] = hyp_s.size
    st['WASO'] = np.count_nonzero(hyp_s == 0) if has_sleep else np.nan
    st['TST'] = np.count_nonzero(hyp_s > 0)
    for name, val in stages.items():
        st[name] = np.count_nonzero(hyp == val)
    st['NREM'] = st['N1'] + st['N2'] + st['N3']
    st['SOL'] = idx_sleep[0] if has_sleep else np.nan
    for name, val in stages.items():
        idx = np.flatnonzero(hyp == val)
        st[f'Lat_{name}'] = idx[0] if idx.size else np.nan
    st = {k: v / (60 * sf_hyp) for k, v in st.items()}  # epochs -> minutes
    for name in [*stages, 'NREM']:
        st[f'%{name}'] = 100 * st[name] / st['TST'] if has_sleep else np.nan
    st['SE'] = 100 * st['TST'] / st['TIB']
    st['SME'] = 100 * st['TST'] / st['SPT'] if has_sleep else np.nan
    return {k: float(v) for k, v in st.items()}


def _transition_probs(hyp, n_stages=5):
    """Stage-transition probabilities P(from -> to) of a hypnogram coded 0..n_stages-1.

    Same as the legacy ``yasa.transition_matrix`` reindexed to all stages; stages without
    outgoing transitions get probability 0.
    """
    counts = np.zeros((n_stages, n_stages))
    np.add.at(counts, (hyp[:-1], hyp[1:]), 1)
    tot = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)


def macro_features(hypnograms, subjects):
    """46 sleep macro features: 21 YASA sleep statistics + 25 stage-transition probabilities.

    Hypnograms are coded 0=W, 1=N1, 2=N2, 3=N3, 4=REM (30-s epochs).
    """
    stages = ['W', 'N1', 'N2', 'N3', 'R']
    rows = []
    for hyp in hypnograms:
        hyp = np.asarray(hyp, dtype=int)
        hyp = np.where(hyp > 4, -1, hyp)  # movement/unscored -> artefact
        st = _sleep_statistics(hyp, sf_hyp=1 / 30)
        probs = _transition_probs(hyp[hyp >= 0])
        tr = {f'P({stages[a]}->{stages[b]})': probs[a, b] for a in range(5) for b in range(5)}
        rows.append({**st, **tr})
    return pd.DataFrame(rows, index=subjects)


def _xgb(seed):
    return xgb.XGBClassifier(
        learning_rate=0.02, booster='gbtree', objective='binary:logistic', max_leaves=127,
        verbosity=0, seed=seed, nthread=-1, colsample_bytree=0.6, subsample=0.7,
        eval_metric='logloss')


def _fit_predict(X_train, y_train, X_test, seed):
    # Missing values are imputed with training medians, so that missingness itself, which
    # differs between diagnoses, cannot be used by the classifier.
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_test = np.where(np.isnan(X_test), med, X_test)
    class_weights = len(y_train) / (2 * np.bincount(y_train))
    model = _xgb(seed)
    model.fit(X_train, y_train, sample_weight=class_weights[y_train])
    return model.predict_proba(X_test)[:, 1]


# ---- DeLong test for paired AUCs (Sun & Xu, 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _delong(y, preds):
    """AUCs and their covariance for several predictions on the same subjects."""
    y = np.asarray(y)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(p) for p in neg])
    tz = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def auc_ci(y, p, alpha=0.05):
    auc, cov = _delong(y, np.atleast_2d(p))
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return auc[0], max(auc[0] - z * se, 0.0), min(auc[0] + z * se, 1.0)


def auc_diff(y, p_new, p_ref, alpha=0.05):
    """Paired AUC difference (new - ref) with DeLong 95% CI and two-sided p value."""
    aucs, cov = _delong(y, np.vstack([p_new, p_ref]))
    d = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(d) / se) if se > 0 else np.nan
    return d, d - z * se, d + z * se, p


def run_incremental(X_ref, y, cv_seeds, dyn_cv, X_dyn_dev, test=None, final_seed=10, n_folds=5):
    """Compare the reference model (X_ref) with the reference model plus dynamic features.

    X_ref : reference predictors of the development cohort (NaN allowed; imputed within
        each training fold).
    dyn_cv(k, train_index) -> dynamic features (n_dev, n_dyn) of partition-fold k, computed
        with that fold's states and selected features.
    X_dyn_dev : dynamic Top-20 features of the development cohort (reference states).
    test : dict(ref=..., dyn=..., y=...) for the validation cohort, or None.
    """
    y = np.asarray(y, dtype=int)
    names = ['Reference', 'Reference + dynamic']

    cv_auc = []
    k = 0
    for seed in cv_seeds:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        probs = np.zeros((2, len(y)))
        for train_index, test_index in kf.split(np.zeros((len(y), 1)), y):
            X_dyn = dyn_cv(k, train_index)
            for j, X in enumerate([X_ref, np.hstack([X_ref, X_dyn])]):
                probs[j, test_index] = _fit_predict(X[train_index], y[train_index], X[test_index], seed)
            k += 1
        cv_auc.append([roc_auc_score(y, probs[j]) for j in range(2)])
    cv_auc = np.array(cv_auc)

    res = pd.DataFrame(dict(cv_auc_mean=cv_auc.mean(0), cv_auc_min=cv_auc.min(0),
                            cv_auc_max=cv_auc.max(0)), index=names)
    d = cv_auc[:, 1] - cv_auc[:, 0]
    dres = pd.DataFrame(dict(cv_diff_mean=[d.mean()], cv_diff_min=[d.min()], cv_diff_max=[d.max()],
                             cv_n_positive=[int((d > 0).sum())]),
                        index=['Reference + dynamic vs Reference'])

    if test is not None:
        yt = np.asarray(test['y'], dtype=int)
        tr = [X_ref, np.hstack([X_ref, X_dyn_dev])]
        te = [test['ref'], np.hstack([test['ref'], test['dyn']])]
        pt = [_fit_predict(a, y, b, final_seed) for a, b in zip(tr, te)]
        for j, p in enumerate(pt):
            res.loc[names[j], ['val_auc', 'val_lo', 'val_hi']] = auc_ci(yt, p)
        dres.loc[dres.index[0], ['val_diff', 'val_lo', 'val_hi', 'val_p']] = auc_diff(yt, pt[1], pt[0])
    return res, dres
