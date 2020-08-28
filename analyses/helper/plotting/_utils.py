import numpy as np
import seaborn as sns

def _limit_n_points(arr, limit=5000):
    """limits #points in the array (while preserving first and last) for better time/memory efficiency"""
    if len(arr) > limit:
        every_n = int(len(arr) / limit)
        return np.hstack([arr[0], arr[1:-1:every_n], arr[-1]])
    else:
        return arr

def _add_distplot(ax, vals, bins, y=None, color='k', hist_kws={}, distplot_y_frac=0.25):
    """
    adds underyling distribution to plot 
    for internal usage e.g. in pdp or score_vs_col

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot object or None
        axes to plot on
    vals : 1D array
        array of feature values
    bins : 1D array or number
        passed to sns.distplot
    y : 1D array or None
        class labels to group the `vals` by
        if None (default) then no grouping is applied
    color : str or any acceptable for matplotlib color
        passed to `sns.distplot`, works only for y=None, 
        otherwise distplots colors are set by default to 'r' (y=0) and 'b' (y=1)
    hist_kws : dict
        other kws passed to `sns.distplot`
    distplot_y_frac : float
        sets fraction of figure's height dedicated to distplot


    Returns
    -------
    None
    """
    xlim = ax.get_xlim()
    ax2 = ax.twinx()

    hist_default_kws=dict(lw=2, alpha=0.3)
    hist_kws = {**hist_default_kws, **hist_kws}  # `hist_kws` overwrites `hist_default_kws`

    if y is None: 
        sns.distplot(vals, ax=ax2,
                     bins=bins, color=color,
                     hist=True, kde=False, norm_hist=True, hist_kws=hist_kws)
    else:
        sns.distplot(vals[np.array(y)==0], ax=ax2,
                     bins=bins, color='r',
                     hist=True, kde=False, norm_hist=True, hist_kws=hist_kws)
        sns.distplot(vals[np.array(y)==1], ax=ax2, 
                     bins=bins, color='b',
                     hist=True, kde=False, norm_hist=True, hist_kws=hist_kws)       

    ax2.set_ylim(top=ax2.get_ylim()[1]/distplot_y_frac)
    ax2.set_yticks([]) 
    ax.set_ylim(bottom=ax.get_ylim()[0]*0.95)
    ax.set_xlim(xlim)