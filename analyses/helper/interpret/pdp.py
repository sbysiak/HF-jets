import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import partial_dependence

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import partial_dependence

def plot_pdp(clf, X, feature, scaler=None, column_names=None, query=None, xlabel=None, show_deciles=True, show_distplot=False, y=None, pardep_kws={}, plt_kws={}, distplot_kws={}, ax=None):
    """ plots partial dependence plot against `feature` for samples satifying `query`

    Parameters
    ----------
    clf : compatible with sklearn.inspect.parital_dependence
        model
    X : pd.DataFrame or 2D array of numbers
        data to calculated partial dependece
        both training and hold-on set (or combined) is reasonable for this purpose
    feature : str
        name of the feature to compute pdp w.r.t
    scaler : sklearn-compatible scaler e.g. StandardScaler
        scaler used to scale training data
    column_names : iterable of strings
        names of the columns in the dataset,
        if None (default) then X has to be dataframe with correct columns
        other args as `feature` or `query` relies on it
    query : str
        selection criteria, only samples passing it will be used to compute pdp
        has to be valid input for pd.DataFrame.query()
    xlabel : str or None
        xlabel, if None (default) `feature` will be used
    show_deciles : bool
        if small vertical lines (seaborn's rugs) corresponding to deciles of selected `feature` values should be shown,
        selected i.e. passing `query`
    show_distplot : bool
        if distribution of `feature` should be plotted below pdp
        if `y` passed, than it's grouped by y=0/1
    y : array of numbers
        samples labels, used to split distplot, 
        used only if show_distplot is True
    pardep_kws : dict
        passed to sklearn.inspect.parital_dependence
    plt_kws : dict
        passed to plt.plot
    distplot_kws : dict
        passed to sns.distplot, 
        has some defaults - see code
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
        default=None, meaning creating axes inside function

    Returns
    -------
    ax
    """
    if not ax:
        _,ax = plt.subplots(figsize=(7,5))
    if column_names is None:
        column_names = X.columns


    X_orig = X if not scaler else scaler.inverse_transform(X)
    df = pd.DataFrame(X_orig)
    df.columns = column_names
    if query: df = df.query(query)

    df_xgb = pd.DataFrame(scaler.transform(df))
    df_xgb.columns = [f'f{i}' for i in range(df_xgb.shape[1])]
    feat_idx = list(column_names).index(feature)
    feat_name_xgb = f'f{feat_idx}'

    part_dep, feat_vals = partial_dependence(clf, df_xgb[df_xgb[feat_name_xgb].notna()], features=[feat_name_xgb], **pardep_kws)
    part_dep, feat_vals = np.array(part_dep[0]), np.array(feat_vals[0])
    if scaler: feat_vals_orig = feat_vals * np.sqrt(scaler.var_[feat_idx]) + scaler.mean_[feat_idx]
    else: feat_vals_orig = feat_vals

    ax.plot(feat_vals_orig, part_dep, lw=3, **plt_kws)
    ax.set_xlim(left=min(ax.get_xlim()[0], min(feat_vals_orig)), right=max(ax.get_xlim()[1], max(feat_vals_orig)))

    vals = df[feature]
    if show_deciles:
        xlim = ax.get_xlim()
        deciles = np.nanpercentile(vals, np.arange(0, 101, 10))
        sns.rugplot(deciles, ax=ax)
        ax.set_xlim(xlim)
    if show_distplot:
        xlim = ax.get_xlim()
        ax2 = ax.twinx()
        
        # defaults for distplot_kws
        if not distplot_kws:
            distplot_kws = dict(kde=False, hist=True, bins=200)
        if 'kde_kws' not in distplot_kws.keys(): 
            # if distplot_kws is passed but does not contain `kde_kws` key
            bw = (np.max(feat_vals_orig) - np.min(feat_vals_orig))/100
            distplot_kws['kde_kws'] = dict(gridsize=10000, shade=True, legend=False, alpha=0.2, bw=bw)
        if 'hist_kws' not in distplot_kws.keys():
            distplot_kws['hist_kws'] = dict(lw=2)

        if y is None: 
            sns.distplot(vals, ax=ax2, legend=False, distplot_kws=distplot_kws)
        else:
            sns.distplot(vals[np.array(y)==0], ax=ax2, color='b',  **distplot_kws)
            sns.distplot(vals[np.array(y)==1], ax=ax2, color='r',  **distplot_kws)
        ax2.set_ylim(top=ax2.get_ylim()[1])
        ax2.set_yticks([]) 
        ax.set_ylim(bottom=ax.get_ylim()[0]*0.5)
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel if xlabel else feature)
    ax.set_ylabel('partial dependence')
    return ax
