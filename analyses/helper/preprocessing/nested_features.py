"""
Small framework for extracting single-value features out of arrays nested in columns
as returned by uproot.open(fileName)[treeName].pandas.df(flatten=False).
All operations happen inplace!

Adding features in steps:
1. create features which you want to sort or to sort-by them, this includes transformations like `abs` or `log`
2. create indexes `add_sorting_index()`
3. filter indices arrays `apply_cuts()`
4. create sorted columns `add_sorted_col()`
5. fill sorted values to new columns: `add_nth_val()`
6. As last step remeber to exclude temporary columns from data set!

E.g. in order to have $p_T$, $\eta$ and $\phi$ of first 5 tracks sorted by abs(IPd) with -0.9 < $\eta$ < 0.9 do:
```
# Ad 1
df['Jet_Track_IPdAbs'] = df.apply(lambda row: abs(row['Jet_Track_IPd']), axis=1)
# Ad 2
add_sorting_index(df, 'Jet_Track_IPdAbs', 'desc')
# Ad 3
apply_cut(df, 'Jet_Track_Eta > -0.9', 'IPdAbs', 'desc')
apply_cut(df, 'Jet_Track_Eta < 0.9' , 'IPdAbs', 'desc')
# Ad 4
add_sorted_col(df, 'Jet_Track_Pt',  'IPdAbs', 'desc')
add_sorted_col(df, 'Jet_Track_Phi', 'IPdAbs', 'desc')
add_sorted_col(df, 'Jet_Track_Eta', 'IPdAbs', 'desc')
# Ad 5
for i in range(5):
    add_nth_val(df, col_name=f'Jet_Track_Pt__sortby__IPdAbs__desc' , n=i, fillna=None)
    add_nth_val(df, col_name=f'Jet_Track_Phi__sortby__IPdAbs__desc', n=i, fillna=None)
    add_nth_val(df, col_name=f'Jet_Track_Eta__sortby__IPdAbs__desc', n=i, fillna=None)
# Ad 6
df.drop(['Index__Jet_Track__sortby__IPdAbs__desc', 'Jet_Track_Pt__sortby__IPdAbs__desc', 'Jet_Track_Phi__sortby__IPdAbs__desc', 'Jet_Track_Eta__sortby__IPdAbs__desc'], inplace=True)
```
"""

import numpy as np
import pandas as pd

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

    func = lambda x: np.argsort(x)[::order_incr]
    df[index_col_name] = df[sorting_col].apply(func)

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
    df[sorted_col_name] = [arr[idx] for _,arr,idx in df[[col_name, index_name]].itertuples()]

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

def apply_cut(df, cut, sortby, order):
    """ filters index according to cut on other column
    Parameters
    ----------
    df : pd.DataFrame
        input data
    cut : str
        form: '[feature] [>/<] [value]' e.g. 'Jet_SecVtx_Chi2 < 10'
    sortby : str
        feature of obj (like track or SV) which will determine new order
    order : str
        'desc' or 'asc'
    """
    if len(cut.split(' ')) != 3 or cut.split(' ')[1] not in ['<', '>', '<=', '>=']:
        raise ValueError("incorrect cut str format, it should be sth like: e.g. \'Jet_SecVtx_Chi2 < 10\' ")
    cut_colname, gt_lt, cut_value = cut.split(' ')
    cut_value = float(cut_value)
    kind = _colname2kind(cut_colname)
    index_name = _form_index_name(kind, sortby, order)
    if gt_lt == '>':
        df[index_name] =                 df.apply(lambda row: [idx for idx, val in zip(row[index_name], np.array(row[cut_colname])[row[index_name]]) if val > cut_value], axis=1)
    elif gt_lt == '<':
        df[index_name] = df.apply(lambda row: [idx for idx, val in zip(row[index_name], np.array(row[cut_colname])[row[index_name]]) if val < cut_value], axis=1)
