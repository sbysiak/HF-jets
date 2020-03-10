from sklearn.metrics import roc_curve

def signal_eff(y_true, y_proba, mistag_rate_thresh):
    """computes signal efficiency (TPR) at given mistagging rate (FPR)"""
    
    if hasattr(mistag_rate_thresh, '__iter__'):
        effs = []
        for t in mistag_rate_thresh:
            eff = signal_eff(y_proba, y_true, t)
            effs.append(eff)
        return effs
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    for b_tag_eff, mistag_rate in zip(tpr, fpr):
        if mistag_rate > mistag_rate_thresh:
            return b_tag_eff