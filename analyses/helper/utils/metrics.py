from sklearn.metrics import roc_curve
import numpy as np

def signal_eff(y_true, y_proba, mistag_rate_thresh, sample_weight=None):
    """computes signal efficiency (TPR) at given mistagging rate (FPR), supports multiple FPRs"""
    
    if hasattr(mistag_rate_thresh, '__iter__'):
        effs = []
        for t in mistag_rate_thresh:
            eff = signal_eff(y_proba, y_true, t, sample_weight=sample_weight)
            effs.append(eff)
        return effs
    
    fpr, tpr, _ = roc_curve(y_true, y_proba, sample_weight=sample_weight)
    for b_tag_eff, mistag_rate in zip(tpr, fpr):
        if mistag_rate > mistag_rate_thresh:
            return b_tag_eff


        
def signal_significance(y_true, y_proba, sig2incl_ratio, threshold=None, sample_weight=None):
    """ calculates signal significance S/sqrt(S+B), where S and B are number of true signal and background samples

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
    n_bkg = (1-sig2incl_ratio)* 100
    n_sig = sig2incl_ratio    * 100
    B = n_bkg*fpr
    S = n_sig*tpr
    significances  = S/np.sqrt(S+B)
    if threshold:
        for t,s in zip(thresholds, significances):
            if t >= threshold:
                return s,t
    else:
        return significances, thresholds
    
    
    
def get_optimal_threshold(y_true, y_proba, sig2incl_ratio, sample_weight=None):
    """returns threshold optimal from point of view of signal significance"""
    significances, thresholds = signal_significance(y_true, y_proba, sig2incl_ratio, sample_weight=sample_weight)
    return thresholds[np.nanargmax(significances)]



def purity(y_true, y_pred, sample_weight=None):
    """computes purity aka precision or PPV"""
    if sample_weight is None: 
        sample_weight=np.ones_like(y_true)
    TP = np.sum((y_pred) * y_true * sample_weight)
    FP = np.sum((y_pred) * (y_true==0) * sample_weight)
    return TP / (TP+FP)
