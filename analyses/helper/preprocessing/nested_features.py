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
    sorted_object_kind, sortby, order = [
        i.strip("_") for i in [sorted_object_kind, sortby, order]
    ]
    return f"Index__{sorted_object_kind}__sortby__{sortby}__{order}"


def _colname2kind(colname):
    """extracts object kind (e.g. 'Jet_Track') from column name"""
    for kind in ["Jet_Track", "Jet_SecVtx", "Jet_Splitting"]:
        if colname.startswith(kind):
            break
    return kind


def add_sorting_index(df, sorting_col, order):
    """adds to `df` a new column, which elements are sorted arrays

    df : pd.DataFrame
        input data
    sorting_col : str
        name of column to be used as index, its elements have to be arrays
    order : str
        'desc' or 'asc'
    """
    assert order in ["desc", "asc"]
    order_incr = -1 if order == "desc" else 1

    kind = _colname2kind(sorting_col)
    feat_name = sorting_col.replace(kind + "_", "")
    index_col_name = _form_index_name(kind, feat_name, order)

    func = lambda x: np.argsort(x)[::order_incr]
    df[index_col_name] = df[sorting_col].apply(func)


def add_sorted_col(df, col_name, sortby, order):
    """adds to a `df` a new column
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
    df[sorted_col_name] = [
        arr[idx] for _, arr, idx in df[[col_name, index_name]].itertuples()
    ]


def add_nth_val(df, col_name, n, fillna=None):
    """adds to `df` a new column,
    extracted as `n`-th values from `col_name` (which elements are arrays)

    df : pd.DataFrame
        input data
    col_name : str
        name of column to extract from
    order : str
        'desc' or 'asc'
    """
    kind = _colname2kind(col_name)
    new_col_name = col_name.replace(kind, f"{kind}_{n}")
    df[new_col_name] = [arr[n] if n < len(arr) else fillna for arr in df[col_name]]


def apply_cut(df, cut, sortby, order):
    """filters index according to cut on other column
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
    if len(cut.split(" ")) != 3 or cut.split(" ")[1] not in ["<", ">", "<=", ">="]:
        raise ValueError(
            "incorrect cut str format, it should be sth like: e.g. 'Jet_SecVtx_Chi2 < 10' "
        )
    cut_colname, gt_lt, cut_value = cut.split(" ")
    cut_value = float(cut_value)
    kind = _colname2kind(cut_colname)
    index_name = _form_index_name(kind, sortby, order)
    if gt_lt == ">":
        df[index_name] = df.apply(
            lambda row: [
                idx
                for idx, val in zip(
                    row[index_name], np.array(row[cut_colname])[row[index_name]]
                )
                if val > cut_value
            ],
            axis=1,
        )
    elif gt_lt == "<":
        df[index_name] = df.apply(
            lambda row: [
                idx
                for idx, val in zip(
                    row[index_name], np.array(row[cut_colname])[row[index_name]]
                )
                if val < cut_value
            ],
            axis=1,
        )


def extract_features(
    df,
    features_tracks,
    features_sv,
    n_tracks=10,
    n_sv=3,
    sortby_tracks="IPdNsigmaAbs",
    sortby_sv="LxyNsigma",
    sorting_mode_tracks="desc",
    sorting_mode_sv="desc",
):
    """extracts features from non-flattened ROOT tree to flat table

    Parameters
    ----------
    df : pd.DataFrame
        dataframe from `uproot::tree.pandas.df(flatten=False)`
    features_tracks/sv : list of strings
        per-track and per-SV features to be extracted
        each one is passed to `add_sorted_col`
        if they are not computed/not present in df - warning is printed but no error raised
    n_tracks/sv : int
        number of tracks/SV to be stored,
        if real number of tracks or SV in jet is smaller then that, it will be filled with NaN
    sortby_tracks/sv : str
        "Jet_SV__{sortby_sv}" is passed to `add_sorting_index()`
    sorting_mode_tracks : str
        'desc' or 'asc'
        passed to `add_sorting_index()`

    Returns
    -------
    df : pd.DataFrame
        dataframe in flat table format
    """

    #     def IPdNSigmaAbs_cutSmallSigma(row):
    #         pt = row['Jet_Track_Pt']
    #         IPd_sigma = np.sqrt(row['Jet_Track_CovIPd'])
    #         sigma_threshold = 0.004444561*pt**(-0.4790711) if pt < 10 else 0.0016
    #         if IPd_sigma > sigma_threshold:
    #             return abs(row['Jet_Track_IPd'] / IPd_sigma)
    #         else:
    #             return -1

    def subtract_phi(phi1, phi2):
        diff = phi1 - phi2
        if abs(diff) <= np.pi:
            return diff
        elif diff > np.pi:
            return diff - 2 * np.pi
        elif diff < -np.pi:
            return diff + 2 * np.pi

    def subtract_eta(eta1, eta2):
        diff = eta1 - eta2
        return diff

    ### FILTERING
    #     df = df.query('Jet_Pt > 10 and Jet_Pt < 150')

    # add custom features,  df.apply is slow: execute only if needed
    if "Jet_Track_DeltaPhi" in features_tracks or "Jet_Track_DeltaR" in features_tracks:
        df["Jet_Track_DeltaPhi"] = df.apply(
            lambda row: np.array(
                [
                    subtract_phi(tr_phi, row["Jet_Phi"])
                    for tr_phi in row["Jet_Track_Phi"]
                ]
            ),
            axis=1,
        )
    if "Jet_Track_DeltaEta" in features_tracks or "Jet_Track_DeltaR" in features_tracks:
        df["Jet_Track_DeltaEta"] = df.apply(
            lambda row: np.array(
                [
                    subtract_eta(tr_eta, row["Jet_Eta"])
                    for tr_eta in row["Jet_Track_Eta"]
                ]
            ),
            axis=1,
        )
    if "Jet_Track_DeltaR" in features_tracks:
        df["Jet_Track_DeltaR"] = df.apply(
            lambda row: np.array(
                [
                    np.sqrt(tr_phi ** 2 + tr_eta ** 2)
                    for tr_phi, tr_eta in zip(
                        row["Jet_Track_DeltaPhi"], row["Jet_Track_DeltaEta"]
                    )
                ]
            ),
            axis=1,
        )
    if "Jet_Track_PtFrac" in features_tracks:
        df["Jet_Track_PtFrac"] = df.apply(
            lambda row: np.array(
                [(tr_pt / row["Jet_Pt"]) for tr_pt in row["Jet_Track_Pt"]]
            ),
            axis=1,
        )
    #     df['Jet_Track_IPdNsigmaAbs']  = df.apply(lambda row: abs(row['Jet_Track_IPd'] / np.sqrt(row['Jet_Track_CovIPd'])), axis=1)
    #     df['Jet_Track_IPdNsigmaAbs']  = df.apply(lambda row: IPdNsigmaAbs_cutSmallSigma(row), axis=1)
    df["Jet_Track_IPdSigma"] = df["Jet_Track_CovIPd"].pow(0.5)
    df["Jet_Track_IPzSigma"] = df["Jet_Track_CovIPz"].pow(0.5)
    #     df = df.drop(['Jet_Track_CovIPd', 'Jet_Track_CovIPz'], axis=1) #

    df["Jet_Track_IPdAbs"] = eval("abs(a)", dict(a=df["Jet_Track_IPd"]))
    df["Jet_Track_IPzAbs"] = eval("abs(a)", dict(a=df["Jet_Track_IPz"]))
    df["Jet_Track_IPdNsigma"] = eval(
        "a/b", dict(a=df["Jet_Track_IPd"], b=df["Jet_Track_IPdSigma"])
    )
    df["Jet_Track_IPzNsigma"] = eval(
        "a/b", dict(a=df["Jet_Track_IPz"], b=df["Jet_Track_IPzSigma"])
    )
    df["Jet_Track_IPdNsigmaAbs"] = eval(
        "abs(a)/b", dict(a=df["Jet_Track_IPd"], b=df["Jet_Track_IPdSigma"])
    )
    df["Jet_Track_IPzNsigmaAbs"] = eval(
        "abs(a)/b", dict(a=df["Jet_Track_IPz"], b=df["Jet_Track_IPzSigma"])
    )

    #     def cut_val(track_pt):
    #         return 0.004444561*track_pt**(-0.4790711) if track_pt < 10 else 0.0015

    #     df['Jet_Track_CutIPdSigmaVSPt'] = df.apply(lambda row:
    #                                         np.array([int(ipd_sigma < cut_val(pt))  for ipd_sigma, pt in zip(row['Jet_Track_IPdSigma'], row['Jet_Track_Pt'])]),
    #                                         axis=1
    #                                       )
    df["Jet_SecVtx_LxyNsigma"] = eval(
        "a / b", dict(a=df["Jet_SecVtx_Lxy"], b=df["Jet_SecVtx_SigmaLxy"])
    )

    ### create index cols
    add_sorting_index(df, f"Jet_Track_{sortby_tracks}", sorting_mode_tracks)
    add_sorting_index(df, f"Jet_SecVtx_{sortby_sv}", sorting_mode_sv)

    ### apply cuts a.k.a. filter index cols
    #     apply_cut(df, 'Jet_Track_IPdNsigmaAbs < 50', track_sorting_var, 'desc')
    #     apply_cut(df, 'Jet_Track_Pt > 0.5', track_sorting_var, 'desc')
    #     apply_cut(df, 'Jet_Track_CutIPdSigmaVSPt < 0.5', track_sorting_var, 'desc')
    #     apply_cut(df, 'Jet_SecVtx_Chi2 < 10' ,'LxyNsigma', 'desc')
    #     apply_cut(df, 'Jet_SecVtx_Dispersion < 0.01' ,'LxyNsigma', 'desc')
    #     apply_cut(df, 'Jet_SecVtx_SigmaLxy < 0.1' ,'LxyNsigma', 'desc')

    for feat in list(features_tracks) + list(features_sv):
        if feat not in df.columns:
            print(
                f"add_features(): Warning: {feat} not found in DataFrame, probably it was not read from ROOT file or not created during adding features"
            )

    features_tracks = [feat for feat in features_tracks if feat in df.columns]
    features_sv = [feat for feat in features_sv if feat in df.columns]

    for feat in features_tracks:
        add_sorted_col(df, feat, sortby_tracks, sorting_mode_tracks)

    for feat in features_sv:
        add_sorted_col(df, feat, sortby_sv, sorting_mode_sv)

    ### extract n-th value from sorted cols
    for feat in features_tracks:
        for i in range(n_tracks):
            add_nth_val(
                df, col_name=f"{feat}__sortby__{sortby_tracks}__desc", n=i, fillna=None
            )

    for feat in features_sv:
        for i in range(n_sv):
            add_nth_val(
                df, col_name=f"{feat}__sortby__{sortby_sv}__desc", n=i, fillna=None
            )

    ### drop temporary columns, i.e. those containing arrays, like 'Index__*' as well as initial columns used for extraction, like 'Jet_Track_Pt'
    if len(df) == 0:
        return pd.DataFrame()

    columns_to_keep = [
        col
        for col, val in zip(df.columns, df.iloc[0])
        if not hasattr(val, "__iter__") or isinstance(val, str)
    ]
    return df[columns_to_keep]
