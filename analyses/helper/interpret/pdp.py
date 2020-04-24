import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pdp(clf, X, feature, scaler=None, column_names=None, query=None, xlabel=None, show_deciles=True, pd_kwargs={}, plt_kwargs={}, ax=None):
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
    pd_kwargs : dict
        passed to sklearn.inspect.parital_dependence
    plt_kwargs : dict
        passed to plt.plot
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
    
    part_dep, feat_vals = partial_dependence(clf, df_xgb[df_xgb[feat_name_xgb].notna()], features=[feat_name_xgb], **pd_kwargs)
    part_dep, feat_vals = np.array(part_dep[0]), np.array(feat_vals[0])
    if scaler: feat_vals_orig = feat_vals * np.sqrt(scaler.var_[feat_idx]) + scaler.mean_[feat_idx]
    else: feat_vals_orig = feat_vals

    ax.plot(feat_vals_orig, part_dep, **plt_kwargs)
    ax.set_xlim(left=min(ax.get_xlim()[0], min(feat_vals_orig)), right=max(ax.get_xlim()[1], max(feat_vals_orig)))

    if show_deciles:
        xlim = ax.get_xlim()
        deciles = np.percentile(df[feature], np.arange(0, 101, 10))
        sns.rugplot(deciles, ax=ax)
        ax.set_xlim(xlim)
        
    ax.set_xlabel(xlabel if xlabel else feature)
    ax.set_ylabel('partial dependence')
    ax.legend()
    return ax
