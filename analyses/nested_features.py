def _form_index_name(sorted_object_kind, sortby, order):
    """forms name of sorting index

    Parameters
    ----------
    sorted_object_kind : str
        'Jet_Track', 'Jet_SecVtx' or 'Jet_Splitting'
    sortby : str
        feature to sort by, e.g. 'Pt' or 'Lxy'
    order : str
        'desc', 'asc'

    Returns
    -------
    name : str
        index name e.g. Index__Jet_Track__sortby__Pt__desc
    """
    sorted_object_kind, sortby, order = [i.strip('_') for i in [sorted_object_kind, sortby, order]]
    return f"Index__{sorted_object_kind}__sortby__{sortby}__{order}"


def _colname2kind(colname):
    """extracts object kind (e.g. 'Jet_Track') from column name"""
    for kind in ['Jet_Track', 'Jet_SecVtx', 'Jet_Splitting']:
        if colname.startswith(kind): break
    return kind


def add_sorting_index(df, sorting_col, order):
    """ adds to `df` a new column, which elements are sorted arrays

    df : pd.DataFrame
        input data
    sorting_col : str
        name of column to be used as index, its elements have to be arrays
    order : str
        'desc' or 'asc'
    """
    assert order in ['desc', 'asc']
    order_incr = -1 if order == 'desc' else 1

    kind = _colname2kind(sorting_col)
    feat_name = sorting_col.replace(kind+'_', '')
    index_col_name = _form_index_name(kind, feat_name, order)

    func = lambda row: np.argsort(row[sorting_col])[::order_incr]
    df[index_col_name] = df.apply(func, axis=1)


def add_sorted_col(df, col_name, sortby, order):
    """ adds to a `df` a new column
    its each element is an array with order specified in another column (called index, also containing arrays)

    df : pd.DataFrame
        input data
    col_name : str
        name of column which each element (array) will be reordered
    sortby : str
        feature of obj (like track or SV) which will determine new order
    order : str
        'desc' or 'asc'
    """
    kind = _colname2kind(col_name)
    sorted_col_name = f"{col_name}__sortby__{sortby}__{order}"
    index_name = _form_index_name(kind, sortby, order)
    df[sorted_col_name] = [arr[idx] for _,(arr,idx) in df[[col_name, index_name]].iterrows()]


def add_nth_val(df, col_name, n, fillna=None):
    """ adds to `df` a new column,
    extracted as `n`-th values from `col_name` (which elements are arrays)

    df : pd.DataFrame
        input data
    col_name : str
        name of column to extract from
    order : str
        'desc' or 'asc'
    """
    kind = _colname2kind(col_name)
    new_col_name = col_name.replace(kind, f'{kind}_{n}')
    df[new_col_name] = [arr[n] if n<len(arr) else fillna for arr in df[col_name]]
