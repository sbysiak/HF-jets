from sklearn.metrics import roc_curve
import numpy as np


def signal_eff(y_true, y_proba, mistag_rate_thresh, sample_weight=None):
    """computes signal efficiency (TPR) at given mistagging rate (FPR), supports multiple FPRs"""

    if hasattr(mistag_rate_thresh, "__iter__"):
        effs = []
        for t in mistag_rate_thresh:
            eff = signal_eff(y_proba, y_true, t, sample_weight=sample_weight)
            effs.append(eff)
        return effs

    fpr, tpr, _ = roc_curve(y_true, y_proba, sample_weight=sample_weight)
    for b_tag_eff, mistag_rate in zip(tpr, fpr):
        if mistag_rate > mistag_rate_thresh:
            return b_tag_eff


def signal_significance(
    y_true, y_proba, sig2incl_ratio, threshold=None, sample_weight=None
):
    """calculates signal significance S/sqrt(S+B), where S and B are number of true signal and background samples

    Parameters
    ----------
    y_true : 1D array
        array of true labels
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    sig2incl_ratio : float (0-1)
        assumed signal-to-inclusive ratio
    threshold : float or None
        threshold at which significance will be returned
        if None, then array of significances for each possible threshold are returned
    sample_weight : 1D array or None
        sample weights

    Returns
    -------
    significances : float or array of floats
        array of significances for all possible thresholds if no `threshold` passed
        or significance at given threshold
    thresholds : float or array of floats
        list of thresholds corresponding to each significance if no `threshold` passed
        or single threshold at which significance was calculated (the closest possible threshold
        not smaller than `threshold` arg passed)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
    n_bkg = (1 - sig2incl_ratio) * 100
    n_sig = sig2incl_ratio * 100
    B = n_bkg * fpr
    S = n_sig * tpr
    significances = S / np.sqrt(S + B)
    if threshold:
        for t, s in zip(thresholds, significances):
            if t >= threshold:
                return s, t
    else:
        return significances, thresholds


def get_optimal_threshold(y_true, y_proba, sig2incl_ratio, sample_weight=None):
    """returns threshold optimal from point of view of signal significance"""
    significances, thresholds = signal_significance(
        y_true, y_proba, sig2incl_ratio, sample_weight=sample_weight
    )
    return thresholds[np.nanargmax(significances)]


def purity(y_true, y_pred, sample_weight=None):
    """computes purity aka precision or PPV"""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    TP = np.sum((y_pred) * y_true * sample_weight)
    FP = np.sum((y_pred) * (y_true == 0) * sample_weight)
    return TP / (TP + FP)


def calc_metrics(ts, y_proba, y_true, sample_weights=None):
    """calculates metrics (threshold, tpr, fpr, purity) for given threshold(s)

    Parameters
    ----------
    ts : float or array of floats
        score threshold(s) for which metrics will be computed
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    y_true : 1D array
        array of true labels
    sample_weight : 1D array or None
        sample weights

    Returns
    -------
    ts, tprs, fprs, purities : float or array of floats
        score thresholds and computed metrics
    """
    if not hasattr(ts, "__iter__"):
        ts = [
            ts,
        ]
    if sample_weights is None:
        sample_weights = np.ones_like(y_proba)
    purities = []
    fprs = []
    tprs = []
    n_tot_sig = np.sum(sample_weights[y_true == 1])
    n_tot_bckg = np.sum(sample_weights[y_true == 0])
    for t in ts:
        n_pass_sig = np.sum(sample_weights[(y_true == 1) & (y_proba >= t)])
        n_pass_bckg = np.sum(sample_weights[(y_true == 0) & (y_proba >= t)])
        tprs.append(n_pass_sig / n_tot_sig if n_tot_sig > 0 else np.nan)
        fprs.append(n_pass_bckg / n_tot_bckg if n_tot_bckg > 0 else np.nan)
        purities.append(
            n_pass_sig / (n_pass_sig + n_pass_bckg)
            if (n_pass_sig + n_pass_bckg) > 0
            else 1
        )
    if len(ts) == 1:
        return ts[0], tprs[0], fprs[0], purities[0]
    return np.array(ts), np.array(tprs), np.array(fprs), np.array(purities)


def recursive_threshold_search(
    metric_name, metric_val, y_proba, y_true, sample_weights=None, verbose=False
):
    """finds score threshold closest to the given metric's value in recursive manner

    Parameters
    ----------
    metric_name : string
        name of the metric
    metric_val : float
        value of the metric
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    y_true : 1D array
        array of true labels
    sample_weight : 1D array or None
        sample weights
    verbose : bool

    Returns
    -------
    ts : float
        score threshold value
    vals : float
        value of optimized metric
    """
    ts_next = np.linspace(0, 1, 11)
    prev_min = -1
    prev_max = 999
    ts_final = None
    n_points = 5
    it = 0
    eps_rel = 1e-3
    while True:
        it += 1
        ts, trps, fprs, purities = calc_metrics(
            ts_next, y_proba, y_true, sample_weights
        )

        if metric_name == "score" or metric_name == "proba":
            vals = ts
        elif metric_name == "eff":
            vals = trps
        elif metric_name == "mistag_rate":
            vals = fprs
        elif metric_name == "purity":
            vals = purities
        else:
            raise ValueError(f"illegal value for `metric_name`: {metric_name}")

        idx = np.argmin(abs(vals - metric_val))
        if abs(vals[idx] - metric_val) / max(metric_val, 1e-10) < eps_rel:
            if verbose:
                print(f"finish with t={ts[idx]}, v={vals[idx]} [target={metric_val}]")
            break

        if it > 10:
            if verbose:
                print(
                    f"finish with t={ts[idx]}, v={vals[idx]} [target={metric_val}] [due to REP]"
                )
            break

        prev_min = np.min(vals)
        prev_max = np.max(vals)

        if idx == 0:
            ts_next = np.linspace(ts[0], ts[1], n_points)
            continue
        if idx == len(ts) - 1:
            ts_next = np.linspace(ts[-2], ts[-1], n_points)
            continue

        if (vals[idx] - metric_val) * (vals[idx + 1] - metric_val) < 0:
            pair = ts[idx], ts[idx + 1]
            ts_next = np.linspace(min(pair), max(pair), n_points)
        elif (vals[idx] - metric_val) * (vals[idx - 1] - metric_val) < 0:
            pair = ts[idx], ts[idx - 1]
            ts_next = np.linspace(min(pair), max(pair), n_points)
    if abs(vals[idx] - metric_val) / max(metric_val, 1e-10) > 10 * eps_rel:
        print(
            f"Warning: returning {vals[idx]} while target was {metric_val}, relative diff. = {abs(vals[idx]-metric_val) / max(metric_val, 1e-10)}"
        )
    return ts[idx], vals[idx]
