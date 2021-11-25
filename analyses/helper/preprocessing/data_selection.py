import uproot
import pandas as pd


def create_index(fname, tree_name):
    """creates pd.MultiIndex consisting of path to ROOT file, tree name and entry (without any filtering)"""
    froot = uproot.open(fname)
    jet_id = (
        froot[tree_name].pandas.df(flatten=False, branches=[]).index
    )  # branches=[] reads no branches, branches=None reads all
    multi_index = pd.DataFrame(None, index=jet_id)
    multi_index["fpath"] = fname
    multi_index["tree"] = tree_name
    multi_index[["fpath", "tree"]] = multi_index[["fpath", "tree"]]
    multi_index.reset_index(inplace=True)
    multi_index.set_index(["fpath", "tree", "entry"], inplace=True)
    return multi_index.index


def get_hdf5_nrows(fpath, key="key"):
    try:
        with pd.HDFStore(fpath) as store:
            return store.get_storer(key).nrows
    except Exception as e:
        print(f"ERROR: file {fpath} cannot be opened.\n{e}")
        return 0


def calc_njets(input_files, n_b=None, trainset_frac_b=None, ratio_c_to_udsg=0.1):
    if n_b and trainset_frac_b:
        raise ValueError(
            "One cannot provide both parameters: `n_b` and `trainset_frac_b`"
        )
    if not n_b and not trainset_frac_b:
        raise ValueError(
            "Provide one out of these two parameters: `n_b` and `trainset_frac_b`"
        )

    n_avail_b, n_avail_c, n_avail_udsg = 0, 0, 0
    for f in input_files:
        if f.endswith(".root"):
            try:
                froot = uproot.open(f)
            except FileNotFoundError as e:
                print(f"WARNING: file {f} not found!")
                continue
            tree_name_core = "JetTree_AliAnalysisTaskJetExtractor_Jet_AKTChargedR040_tracks_pT0150_E_scheme_"
            n_avail_b += froot[tree_name_core + "bJets"].numentries
            n_avail_c += froot[tree_name_core + "cJets"].numentries
            n_avail_udsg += froot[tree_name_core + "udsgJets"].numentries
        elif f.endswith(".hdf5"):
            n = get_hdf5_nrows(f)
            if "bJets" in f:
                n_avail_b += n
            elif "cJets" in f:
                n_avail_c += n
            elif "udsgJets" in f:
                n_avail_udsg += n
            else:
                raise ValueError(f"filename {f} does not contain flavour!")
        else:
            raise ValueError(
                f'file {f} has unsupported extension! Use ".root" or ".hdf5"'
            )

    #     print(n_avail_b, n_avail_c, n_avail_udsg)

    if not n_b:
        n_b = int(n_avail_b * trainset_frac_b)
    else:
        trainset_frac_b = n_b / n_avail_b  # round_down(n_b / n_avail_b, 4)

    n_c = int(ratio_c_to_udsg * n_b)
    n_udsg = int((1 - ratio_c_to_udsg) * n_b)
    trainset_frac_c = n_c / n_avail_c  # round_down(n_c / n_avail_c, 4)
    trainset_frac_udsg = n_udsg / n_avail_udsg  # round_down(n_udsg / n_avail_udsg, 4)

    d = {
        "b": [n_avail_b, n_b, trainset_frac_b],
        "c": [n_avail_c, n_c, trainset_frac_c],
        "udsg": [n_avail_udsg, n_udsg, trainset_frac_udsg],
    }
    df = pd.DataFrame(d)
    df.index = ["n available", "n trainset", "trainset fraction"]
    if any(df.loc["trainset fraction"] > 1):
        print("ERROR: not enough jets")
    return df
