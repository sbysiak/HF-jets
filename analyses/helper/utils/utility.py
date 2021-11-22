import pickle
import numpy as np
import uproot
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
