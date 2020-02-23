import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score as acc, f1_score, roc_curve, roc_auc_score, classification_report, confusion_matrix, auc

def plot_roc(y_true, y_proba, label='', color='k', title='', ax=None):
    """ plots ROC curve on given `ax`

    Parameters
    ----------
    y_true : 1D array
        array of true labels
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    label : str
        legend entry label, AUC score is added to it
    color : str or any acceptable for matplotlib color
        passed to `plt.plot`
    title : str
        figure title
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    label += f' (AUC = {auc_score:0.3f})'
    
    if not ax: 
        fig,ax = plt.subplots(figsize=(6,6))
        ax.axis('equal')
    ax.plot(fpr, tpr, lw=2, label=label, color=color)
    ax.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    if title: ax.set_title(title)
    return ax 



def plot_tagging_eff(y_true, y_proba, label='', color='k', title='', ax=None):
    """ plots ROC curve in typical HEP form (mistag rate vs tagging eff)

    Parameters
    ----------
    y_true : 1D array
        array of true labels
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    label : str
        legend entry label, AUC score is added to it
    color : str or any acceptable for matplotlib color
        passed to `plt.plot`
    title : str
        figure title
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    b_tag_eff = tpr
    b_mistag_rate = fpr
    
    if not ax: fig,ax = plt.subplots(figsize=(7,5))
    ax.plot(b_tag_eff, b_mistag_rate, lw=3, label=label, color=color)
    ax.set_xlabel('$b$ tagging efficiency (TPR)', horizontalalignment='right', x=1)
    ax.set_ylabel('mistagging efficiency (FPR)', horizontalalignment='right', y=1)
    ax.semilogy()
    ax.grid('y')
    ax.legend()
    if title: plt.title(title)
    return ax



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None,
                          verbose=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized'
        else:
            title = 'Unnormalized'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose: print("Normalized confusion matrix")
    else:
        if verbose: print('Confusion matrix, without normalization')

    if verbose: print(cm)

    if not ax: fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
    return ax



def plot_score_vs_pt(clf, X_sample, y_sample, flavour_ptbin_sample, ptbins, score=(roc_auc_score, 'ROC AUC'), label='', marker='o', color='b', ax=None):
    """ 
    may require rewriting as for now it is recomputing all predictions
    """
    score_func, score_label = score
    scores = []
    y_pred = clf.predict(X_sample)
    y_proba = clf.predict_proba(X_sample)[:,1]
    #print('y_proba = ', y_proba)
    n_ptbins = len(ptbins)-1
    for pt_i in range(1, n_ptbins+1):
    #     ptbin = [k.replace('b', 'udsg'), k.replace('udsg', 'b')]
        ptbin = ['udsg'+str(pt_i), 'b'+str(pt_i)]
        X_bin_sample = X_sample[[fp in ptbin for fp in flavour_ptbin_sample], :]
        y_bin_sample = y_sample[[fp in ptbin for fp in flavour_ptbin_sample]]
        y_bin_sample_pred  = clf.predict(X_bin_sample)
        y_bin_sample_proba = clf.predict_proba(X_bin_sample)[:,1]

        try:
            sc = score_func(y_bin_sample, y_bin_sample_proba)
            score_all = score_func(y_sample, y_proba)
        except:
            sc = score_func(y_bin_sample, y_bin_sample_pred)
            score_all = score_func(y_sample, y_pred)

        scores.append(sc)

        
    if not ax: fig,ax = plt.subplots(figsize=(7,5))
    for i,(low,high,sc) in enumerate(zip(ptbins[:-1], ptbins[1:], scores)):
        if i == 0: cur_label = label
        else: cur_label=None
        ax.plot([low, high], [sc, sc], color=color)
        ax.plot((low+high)/2, sc, color=color, marker=marker, label=cur_label)
    ax.hlines(score_all, ptbins[0], ptbins[-1], color=color, linestyle='dotted', alpha=0.5, label=label+' aver')
    if label: 
        plt.legend(ncol=2)        
    ax.set_xlabel('jet $p_{T}^{reco}$ [GeV/$c$]')
    ax.set_ylabel(score_label)
    plt.tight_layout()
    return ax
