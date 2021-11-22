import pickle
import numpy as np
import uproot
import os
import pathlib
from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))


def path2period(fpath):
    idx_LHC = fpath.find("LHC")
    return fpath[idx_LHC : fpath.find("/", idx_LHC)]


def infer_tree_name(fname, flavour):
    """guesses tree name

    Parameters
    ----------
    fname : string
        ROOT file name
    flavour : string
        in MC: 'b' or 'c' or 'udsg', in data: 'all'
    Returns
    -------
    tree_name : string
    """
    froot = uproot.open(fname)
    candidates = [
        k
        for k in [k.decode("utf-8") for k in froot.allkeys()]
        if f"{flavour}Jets" in k and k.startswith("JetTree")
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"infer_tree_name():: Cannot determine tree name! Found following candidates: {candidates} \nPlease provide it explicitly"
        )
    elif not candidates:
        raise ValueError(
            f"infer_tree_name():: Cannot determine tree name! No candidates found! Following keys found in the ROOT file: {froot.allkeys()}"
        )
    else:
        return candidates[0]


def save_df(df, fname_out, debug=True):
    """writes df to hdf5 format

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be written to file
    fname_out : string
        path to output file
    debug : bool
        if memory/storage info should be printed
    Returns
    -------
    None
    """

    print(f"Saving dataframe to {fname_out}...")
    if debug:
        mem_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        mem_usage_idx_mb = df.memory_usage(deep=True)["Index"] / 1024 / 1024
        print(
            f"Memory usage = {mem_usage_mb:.2f} MB (index = {mem_usage_idx_mb:.2f} MB). Ncols= {df.shape[1]}, Nrows = {df.shape[0]}"
        )

    # clear the file first
    if os.path.exists(fname_out):
        with open(fname_out, "w") as _:
            pass
    pathlib.Path(os.path.dirname(fname_out)).mkdir(parents=True, exist_ok=True)
    df.to_hdf(fname_out, key="key", format="table", complib="blosc:zlib", complevel=9)
    if debug:
        fsize_mb = os.path.getsize(fname_out) / 1024 / 1024
        print(f"... done. Output file size = {fsize_mb:.2f} MB")


def save_df_train_test(df, fname_out):
    """splits the df into two parts and stores both in files with some suffixes"""

    def is_train_sample(df):
        """defines logic for splitting samples into train and test set, currenly jets with odd `entry` are tranining samples"""
        entry_idx = df.index.names.index("entry")
        is_odd = (df.index.get_level_values(entry_idx) % 2).to_numpy(dtype=bool)
        is_train = is_odd
        return is_train

    is_train = is_train_sample(df)
    fname_train = fname_out.replace(".", "_train.")
    save_df(df[is_train], fname_train)
    fname_test = fname_out.replace(".", "_test.")
    save_df(df[~is_train], fname_test)


def get_pythia_weight(fpath, flavour="udsg"):
    """returns weight computed as x-section divided by trials,
    taken from fHistXsectionAfterSel and fHistTrialsAfterSel"""

    def read_hist_integral(fpath, flavour, hist_name):
        froot = uproot.open(fpath)
        hist_dir_name = [
            obj.decode("utf-8")
            for obj in froot["ChargedJetsHadronCF"].allkeys()
            if f"{flavour}Jets" in obj.decode("utf-8")
        ][0]
        hist_dir = froot["ChargedJetsHadronCF"][hist_dir_name]
        hist = [h for h in hist_dir if hist_name in str(h.name)][0]
        return (
            hist.values.sum()
            if hasattr(hist, "values")
            else hist._fTsumwy / hist._fTsumw
        )

    xsec = read_hist_integral(fpath, flavour, hist_name="fHistXsectionAfterSel")
    trials = read_hist_integral(fpath, flavour, hist_name="fHistTrialsAfterSel")
    weight = xsec / trials
    return weight


def save_model(model, feat_names, scaler, exp=None, comet_name="model"):
    """pickles a model and list of feature names to "tmp/" dir and if (exp is passed) uploads it to comet.ml

    Parameters
    ----------
    model : any model compatible with pickle
        model object
    feat_names : iterable of strings
        array of features' names used in training
    scaler : sklearn-compatible scaler e.g. StandardScaler
        scaler used to scale training data
    exp : comet_ml.Experiment
        experiment to log model in
    comet_name : str
        dir name inside "models" in "assets" tab in comet, passed to exp.log_model(name)

    Returns
    -------
    fpath_model, fpath_featnames : strings
        relative paths of saved models
    """

    def random_str():
        return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 5))

    fpath_model = f"tmp/model__{random_str()}.pickle"
    fpath_featnames = fpath_model.replace(".pickle", "__feat_names.pickle")
    fpath_scaler = fpath_model.replace(".pickle", "__scaler.pickle")
    pickle.dump(model, open(fpath_model, "wb"))
    pickle.dump(feat_names, open(fpath_featnames, "wb"))
    pickle.dump(scaler, open(fpath_scaler, "wb"))

    if exp:
        exp.log_model(
            name=comet_name, file_or_folder=fpath_model, file_name="model.pickle"
        )
        exp.log_model(
            name=comet_name,
            file_or_folder=fpath_featnames,
            file_name="feat_names.pickle",
        )
        exp.log_model(
            name=comet_name, file_or_folder=fpath_scaler, file_name="scaler.pickle"
        )

    return fpath_model, fpath_featnames, fpath_scaler
