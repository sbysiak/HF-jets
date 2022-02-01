import pickle
import numpy as np
import uproot3
import ROOT
import os
import pathlib
import subprocess
import lzma
import matplotlib.pyplot as plt
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
    froot = uproot3.open(fname)
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


def process_files_list(input_files_param):
    """
    if `input_files_param` is file ending with txt extension then returns list of files from that file
        lines starting with hashtag `#` are skipped
    if `input_files_param` is command starting from `supported_commands` then it executes it and returns list of files
    """
    if input_files_param is None:
        return None
    list_of_files = []
    if input_files_param.endswith(".txt"):
        print(f"\nReading files' names from the file: {input_files_param}")
        with open(input_files_param) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                list_of_files.append(line.replace("\n", ""))
    else:
        supported_commands = ("ls", "find")  # must be tuple not list
        if input_files_param.startswith(supported_commands):
            cmd = input_files_param
        else:
            cmd = f"ls {input_files_param}"
        print(f"\nRunning command: {cmd}")
        cmd_output = subprocess.check_output(cmd, shell=True, text=True)
        for line in cmd_output.split("\n"):
            if line.startswith("#"):
                continue
            if line:
                list_of_files.append(line.replace("\n", ""))
    if len(list_of_files) < 1:
        raise ValueError("List of files empty!")
    from_to = (
        f"from\n{list_of_files[0]}\nto\n{list_of_files[-1]}"
        if len(list_of_files) > 2
        else f"{list_of_files}"
    )
    print(f"List of N={len(list_of_files)} files created, {from_to}")
    return list_of_files


def save_df(df, fname_out, debug=True, **kwargs):
    """writes df to hdf5 format

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be written to file
    fname_out : string
        path to output file
    debug : bool
        if memory/storage info should be printed
    kwargs : dict
        additional parameters passed to pd.DataFrame::to_hdf()
    Returns
    -------
    None
    """

    if debug:
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
    # df.to_csv('temp.csv')
    df.to_hdf(
        fname_out,
        key=kwargs.get("key", "key"),
        format=kwargs.get("format", "table"),
        complib=kwargs.get("complib", "blosc:zlib"),
        complevel=kwargs.get("complevel", 9),
        data_columns=kwargs.get(
            "data_columns",
            [
                "Jet_Pt",
            ],
        ),
    )
    if debug:
        fsize_mb = os.path.getsize(fname_out) / 1024 / 1024
        print(f"... done. Output file size = {fsize_mb:.2f} MB")


def is_train_sample(df):
    """defines logic for splitting samples into train and test set, currenly jets with odd `entry` are tranining samples"""
    entry_idx = df.index.names.index("entry")
    is_odd = (df.index.get_level_values(entry_idx) % 2).to_numpy(dtype=bool)
    is_train = is_odd
    return is_train


def save_df_train_test(df, fname_out, **kwargs):
    """splits the df into two parts and stores both in files with some suffixes

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be written to file
    fname_out : string
        path to output file
    kwargs : dict
        passed to `save_df()`
    Returns
    -------
    None
    """
    is_train = is_train_sample(df)

    def replace_last(s, old, new):
        return new.join(s.rsplit(old, 1))

    # fname_train = fname_out.replace(".hdf", "_train.hdf")
    fname_train = replace_last(fname_out, ".", "_train.")
    save_df(df[is_train], fname_train, **kwargs)
    # fname_test = fname_out.replace(".hdf", "_test.hdf")
    fname_test = replace_last(fname_out, ".", "_test.")
    save_df(df[~is_train], fname_test, **kwargs)


def get_pythia_weight(fpath, flavour="udsg"):
    """returns weight computed as x-section divided by trials,
    taken from fHistXsectionAfterSel and fHistTrialsAfterSel"""

    def read_hist_integral(fpath, flavour, hist_name):
        froot = uproot3.open(fpath)
        hist_dir_name = [
            obj.decode("utf-8")
            for obj in froot["ChargedJetsHadronCF"].allkeys()
            if f"{flavour}Jets_histos" in obj.decode("utf-8")
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


def save_plot(
    fpath,
    comet_exp=None,
    comet_plotname=None,
    store_png=True,
    store_svg=True,
    store_pickle=True,
):
    """saves plot in multiple formats/places: in comet_ml, png, svg, pickle

    svg is compressed to .svgz and pickle is compressed using lmza to .xz
    to open pickled one:

    ```
    %matplotlib notebook
    with lzma.open('plotname.pickle.xz', "rb") as f:
        ax = pickle.load(f)
        # ax.get_lines()
        # ax.patches
        # ax.collections[0].get_offsets().data
        # ax.get_legend()

    ```

    Parameters
    ----------
    fpath : str
        path where plot should be stored, extension is dropped
    comet_exp : string or comet_ml.[Existing]Experiment
        comet_ml experiment (faster)
        or string based on which the ExistingExperiment is created internally and then closed (slower)
        if None then logging to comet_ml is skipped
    comet_plotname : string
        file name in comet_ml, if not provided it is file name extracted from `fpath`
    store_png/svg/pickle : bool
        if given format should be stored

    Returns
    -------
    None
    """
    if fpath.split(".")[-1] in ["png", "svg", "pickle"]:
        fpath = ".".join(fpath.split(".")[:-1])

    def check_min_size(fname, min_size_kb=3):
        fsize_kb = os.path.getsize(fname) / 1024
        if fsize_kb < min_size_kb:
            print(
                f"Warning: something may be wrong. Size of plot {fname} is only {fsize_kb:.1f} KB - below threshold of {min_size_kb} KB. Check if the file is not empty."
            )

    pathlib.Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    if store_png:
        plt.savefig(fpath + ".png")
        check_min_size(fpath + ".png", 3)
    if store_svg:
        plt.savefig(fpath + ".svgz", format="svgz")
        check_min_size(fpath + ".svgz", 3)
    if store_pickle:
        with lzma.open(fpath + ".pickle.xz", "wb") as f:
            pickle.dump(plt.gca(), f)
        check_min_size(fpath + ".pickle.xz", 15)

    if comet_exp:
        if not comet_plotname:
            comet_plotname = os.path.basename(fpath)
        do_close_exp = False
        if isinstance(comet_exp, str):
            comet_exp = comet_ml.ExistingExperiment(previous_experiment=comet_exp)
            do_close_exp = True
        comet_exp.log_figure(comet_plotname)
        if do_close_exp:
            comet_exp.end()


def save_root(root_obj, fpath, save_png=True, logy=False, draw_option=None):
    if fpath.split(".")[-1] in ["root", "png", "svg", "pickle"]:
        fpath = ".".join(fpath.split(".")[:-1])
    pathlib.Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)

    if isinstance(root_obj, ROOT.TCanvas):
        root_obj.SaveAs(fpath + ".root")
        if save_png:
            root_obj.SaveAs(fpath + ".png")
            if logy:
                root_obj.cd(1)
                primitives = ROOT.gPad.GetListOfPrimitives()
                for i in range(primitives.GetEntries()):
                    p = primitives.At(i)
                    if hasattr(p, "GetMinimum") and p.GetMinimum() <= 0:
                        p.SetMinimum(1e-2)
                ROOT.gPad.SetLogy()
                root_obj.SaveAs(fpath + "_logy.png")
    else:
        root_obj.SaveAs(fpath + ".root")

        if save_png:
            c = ROOT.TCanvas()
            if draw_option is None:
                draw_option = "colz" if isinstance(root_obj, ROOT.TCanvas) else "e"
            root_obj.Draw(draw_option)
            c.Draw()
            c.SaveAs(fpath + ".png")
            if logy:
                c.SetLogy()
                c.SaveAs(fpath + "_logy.png")
