import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score as acc, f1_score, roc_curve, roc_auc_score, classification_report, confusion_matrix, auc
from helper.utils import signal_significance
from ._utils import _add_distplot, _limit_n_points

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
    
    fpr = _limit_n_points(fpr)
    tpr = _limit_n_points(tpr)
    
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
    signal_eff       = _limit_n_points(tpr)
    bckg_mistag_rate = _limit_n_points(fpr)
    
    if not ax: fig,ax = plt.subplots(figsize=(7,5))
    ax.plot(signal_eff, bckg_mistag_rate, '.-', lw=1, label=label, color=color)
    ax.set_xlabel('signal tagging efficiency (TPR)', horizontalalignment='right', x=1)
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



def plot_score_vs_pt(y_true, y_pred, y_proba, flavour_ptbin, ptbins, score=(roc_auc_score, 'ROC AUC'), 
                     label='', marker='o', color='b', ax=None):
    """ plots ROC curve in typical HEP form (mistag rate vs tagging eff)

    Parameters
    ----------
    y_true : 1D array of int
        array of true labels
    y_pred : 1D array of int
        array of predicted labels (usually clf.predict())
    y_proba : 1D array of floats
        array of predicted probabilities (usually clf.predict_proba[:,1])
    flavour_ptbin : 1D array of strings
        denotes flavour and ptbin of each sample
        in form "Fx", where F is flavour, e.g. 'b' or 'udsg' and x is number denoting ptbin 
    score : tuple of function and string
        tuple of metric and its name
    label, marker, color : strings or acceptable by matplotlib
        arguments passed in proper places to matplotlib functions
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """
    score_func, score_label = score
    scores = []

    n_ptbins = len(ptbins)-1
    for pt_i in range(1, n_ptbins+1):
        unique_flavours = list(set([''.join([i for i in s if not i.isdigit()]) for s in flavour_ptbin.unique()]))
        curr_flavour_ptbin = [flavour+str(pt_i) for flavour in unique_flavours]
        bin_idx = [fp in curr_flavour_ptbin for fp in flavour_ptbin]
        y_bin_true  = y_true[bin_idx]
        y_bin_proba = y_proba[bin_idx]
        y_bin_pred  = y_pred[bin_idx]

        try:
            sc = score_func(y_bin_true, y_bin_proba)
            score_all = score_func(y_true, y_proba)
        except:
            sc = score_func(y_bin_true, y_bin_pred)
            score_all = score_func(y_true, y_pred)
        scores.append(sc)

    if not ax: fig,ax = plt.subplots(figsize=(7,5))
    for i,(low,high,sc) in enumerate(zip(ptbins[:-1], ptbins[1:], scores)):
        if i == 0: cur_label = label
        else: cur_label=None
        ax.plot([low, high], [sc, sc], color=color)
        ax.plot((low+high)/2, sc, color=color, marker=marker, label=cur_label)
    ax.hlines(score_all, ptbins[0], ptbins[-1], color=color, linestyle=':', alpha=0.5, label=label+' aver')
    if label: 
        plt.legend(ncol=2)        
    ax.set_xlabel('jet $p_{T}^{reco}$ [GeV/$c$]')
    ax.set_ylabel(score_label)
    plt.tight_layout()
    return ax



def plot_score_vs_col(y_true, y_proba, vals, 
                      bins=20, bins_distplot=None,
                      score=(roc_auc_score, 'ROC AUC'), 
                      label='', color='k', marker='o', xlabel='', 
                      show_aver=True, show_distplot=True, show_errorbars=True,
                      ax=None):
    """ Plots selected metric as a function of training variable

    Parameters
    ----------
    y_true : 1D array
        array of true labels
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    vals : 1D array
        array of values against which score will be plotted
    bins : int or array
        defines binning of `vals` in which score will be agregated, 
        passed to numpy.histogram
    bins_distplot : int or array or None
        passed to distplot
        if None (default), then distplot gets binning defined by `bins`
    score : tuple of function and string
        tuple of metric and its name
    label, marker, color : strings or acceptable by matplotlib
        arguments passed in proper places to matplotlib plotting functions
    xlabel : string
        plot's xlabel
    show_aver : bool
        if lines corresponding to average score should be plotted
    show_distplot : bool
        if underlying distribution of `vals` should be plotted
    show_errorbars : bool
        if errorbars from bootstrap sampling should be plotted
        it may take a lot of time
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """
    
    if not ax: fig,ax = plt.subplots(figsize=(10,5))
    score_func, score_label = score
    
    scores = []
    scores_err = []
    _,edges = np.histogram(vals, bins=bins)
    for el,eh in zip(edges[:-1], edges[1:]):
        mask = (vals >= el) & (vals <= eh)
        try:
            score = score_func(y_true[mask], y_proba[mask])
            if show_errorbars:
                bootstrap_scores = []
                for _ in range(5):
                    N = sum(mask)
                    y_true_tmp = np.random.choice(y_true[mask], size=N, replace=True)
                    y_proba_tmp = np.random.choice(y_proba[mask], size=N, replace=True)
                    score_tmp = score_func(y_true_tmp, y_proba_tmp)
                    bootstrap_scores.append(score_tmp)
                score_err = np.std(bootstrap_scores, ddof=1)
        except ValueError:
            score = None
            score_err = None
            
        scores.append(score)
        if show_errorbars: scores_err.append(score_err)

    for el,eh,sc in zip(edges[:-1], edges[1:], scores):
        ax.plot([el,eh], [sc,sc], '-', color=color)
    if show_errorbars: ax.errorbar((edges[:-1]+edges[1:])/2, scores,yerr=scores_err,  marker=marker, color=color, lw=0, elinewidth=2, label=label)
    else: ax.plot((edges[:-1]+edges[1:])/2, scores,  marker=marker, color=color, lw=0, label=label)
    ax.set_ylabel(score_label)
    ax.set_xlabel(xlabel, fontsize=18)
    if ax.get_ylim()[1] < max(scores): ax.set_ylim(top=max(scores)+0.2*(max(scores)-min(scores)))
    
    legend_ncol = 1
    if show_aver:
        score_all = score_func(y_true, y_proba)
        print(label, xlabel)
        ax.hlines(score_all, edges[0], edges[-1], color=color, linestyle=':', alpha=0.5, label=label+' aver')
        legend_ncol +=1

    if show_distplot:
        if bins_distplot is None: bins_distplot = bins
        _add_distplot(ax, vals, bins_distplot, y=None, color=color, hist_kws=dict(histtype='step'), distplot_y_frac=0.25)

    ax.legend(ncol=legend_ncol)
    return ax



def plot_xgb_learning_curve(eval_res, metric, labels=['train set', 'test set'], ax=None):
    """ plots learning curve based on XGBoost during-training watchlist

    Parameters
    ----------
    eval_res : dict
        output of XGBClassifier.evals_result()
    mteric : string
        metric name to be plotted, must be contained in `eval_res[...]`
    labels : list of strings
        contains plot labels for consecutive arrays contained in eval_res,
        used only if keys in eval_res are 'validation_X'
        default=['train set', 'test set']
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """
    if not ax: 
        fig,ax = plt.subplots(figsize=(6,4))
    for label,color in zip(eval_res.keys(), ['b', 'r', 'g', 'k', 'y', 'o']):
        scores = eval_res[label][metric]
        iters = np.arange(0, len(scores))
        if 'validation_' in label:
            leg_label = labels[int(label.replace('validation_', ''))]
        else:
            leg_label = label
        ax.plot(iters, scores, f'{color}.-', label=leg_label)
    ax.set_xlabel('training iterations (#trees)')
    ax.set_ylabel(metric)
    plt.grid(linestyle='--')
    plt.legend()
    plt.tight_layout()
    return ax



def plot_score_distr(y_true, y_proba, mistag_thresholds=[1e-3, 1e-2, 1e-1], ax=None, **plot_kwargs):
    """ plots distribution of scores for both classes with (optional) threshold lines

    Parameters
    ----------
    y_true : 1D array
        array of true labels
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1])
    mistag_thresholds : list of numbers or None
        mistagging rates (FPR) used to draw vertical threshold lines
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function
    plot_kwargs : dict
        passed to plt.hist() and plt.vlines()

    Returns
    -------
    ax
    """    
    if not ax: 
        fig,ax = plt.subplots(10,6)
    bins = np.linspace(0,1,200)
    ax.hist(y_proba[y_true==1], bins=bins, histtype='step', color='b', density=1, **plot_kwargs)
    ax.hist(y_proba[y_true==0], bins=bins, histtype='step', color='r', density=1, **plot_kwargs)
    ax.semilogy()
    ax.set_xlim(0,1)
    ax.set_ylabel('score probability')
    plt.legend()

    if mistag_thresholds:
        ymin,ymax = ax.get_ylim()
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        for mistag_thresh in mistag_thresholds:
            for b_tag_eff, mistag_rate, thresh in zip(tpr, fpr, thresholds):
                if mistag_rate > mistag_thresh:
                    ax.vlines(thresh, ymin, ymax, alpha=0.5, **plot_kwargs)
                    break
    return ax
   
    
    
def plot_signal_significance(y_true, y_proba, sig2incl_ratio, norm=True, ax=None, **plot_kwargs): 
    """ plots signal significance for assumed signal-to-inclusive ratio

    Parameters
    ----------
    y_true : 1D array
        array of true labels, 
        passed to `signal_significance`
    y_proba : 1D array
        array of predicted probabilities (usually clf.predict_proba[:,1]),
        passed to `signal_significance`
    sig2incl_ratio : float (0-1)
        assumed signal-to-inclusive ratio
        passed to `signal_significance`
    norm : bool
        if the significance should be scaled so that its max = 1
        default=True
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function
    plot_kwargs : dict
        passed to plt.plot()

    Returns
    -------
    ax
    """       
    if not ax: 
        fig,ax = plt.subplots(10,3)
        
    significances, thresholds = signal_significance(y_true, y_proba, sig2incl_ratio)
    if norm: 
        ax.plot(thresholds, significances/np.nanmax(significances), **plot_kwargs)
    else: 
        ax.plot(thresholds, significances, **plot_kwargs)
    ax.set_ylabel('$\\frac{S}{\\sqrt{S+B}}$' + ' normalized' if norm else '')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.8, 1)
    plt.grid()
    plt.legend()
    return ax



def plot_eff_vs_threshold(y_true, y_proba, ax=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    signal_eff = _limit_n_points(tpr)
    bckg_mistag_rate = _limit_n_points(fpr)

    if not ax: fig,ax = plt.subplots(figsize=(7,5))
    ax.plot(thresholds[1:], signal_eff[1:], ',-', color='purple', label='signal eff.')
    ax.plot(thresholds[1:], bckg_mistag_rate[1:], ',-', color='green', label='bckg. mistag. rate')
    ax.set_xlabel('threshold')
    ax.set_ylabel('TPR or FPR')
    ax.legend()
    ax.grid()
    return ax