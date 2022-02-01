import yaml
from pprint import pprint
import importlib

import uproot
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sys

sys.path.insert(0, "/eos/user/s/sbysiak/SWAN_projects/HF-jets/analyses/")
from helper.utils import (
    convert_float64_to_float32,
    infer_tree_name,
    save_df,
    save_df_train_test,
    get_pythia_weight,
    process_files_list,
)
from helper.preprocessing import (
    add_sorting_index,
    add_sorted_col,
    add_nth_val,
    apply_cut,
    create_index,
    extract_features,
)


def fname_input2output(fname_in):
    fname_out = fname_in.replace(input_fname_root, output_fname_root)


config_fname = r"ana_config.yaml"
print(f"\nReading config file: {config_fname}")
with open(config_fname) as file:
    cfg_full = yaml.load(file, Loader=yaml.FullLoader)
cfg = cfg_full["ROOT2pd"]
print("Running with following parameters:")
pprint(cfg["parameters"], width=50)
input_files_param = cfg["parameters"]["list_of_input_root_files"]
branches_to_read = cfg["parameters"]["branches_to_read"]
features_sv = cfg["parameters"]["features_sv"]
features_tracks = cfg["parameters"]["features_tracks"]
n_sv = cfg["parameters"]["n_sv"]
n_tracks = cfg["parameters"]["n_tracks"]
sortby_sv = cfg["parameters"]["sortby_sv"]
sortby_tracks = cfg["parameters"]["sortby_tracks"]
sorting_mode_sv = cfg["parameters"]["sorting_mode_sv"]
sorting_mode_tracks = cfg["parameters"]["sorting_mode_tracks"]
output_fname = cfg["parameters"]["output_fname"]
input_fname_root = cfg["parameters"]["input_fname_root"]
output_fname_root = cfg["parameters"]["output_fname_root"]
data_type = cfg["parameters"]["data_type"].lower()


list_of_root_files = process_files_list(input_files_param)

print("\nStart ROOT files processing ...")
df_merged = None
for fname_in in tqdm(list_of_root_files):
    print("\n", fname_in)
    froot = uproot.open(fname_in)
    flavours = (
        ["b", "c", "udsg"]
        if data_type == "mc"
        else [
            "all",
        ]
    )
    for flavour in flavours:
        tree_name = infer_tree_name(fname_in, flavour)
        # df = froot[tree_name].pandas.df(flatten=False, branches=branches_to_read)
        # df.index = create_index(fname_in, tree_name)
        # df_after = extract_features(
        #     df,
        #     features_tracks=features_tracks,
        #     features_sv=features_sv,
        #     n_tracks=n_tracks,
        #     n_sv=n_sv,
        #     sortby_tracks=sortby_tracks,
        #     sortby_sv=sortby_sv,
        #     sorting_mode_tracks=sorting_mode_tracks,
        #     sorting_mode_sv=sorting_mode_sv,
        # )
        # df_after = convert_float64_to_float32(df_after)

        index = create_index(fname_in, tree_name)
        if data_type == "mc":
            weight = get_pythia_weight(fname_in)

        df_after = None
        # entrysteps = int(1e4)
        # i = 0
        num_entries = froot[tree_name].numentries
        step = int(1e6)
        iters = list(np.arange(0, num_entries, step)) + [
            num_entries,
        ]
        for i, (start, stop) in enumerate(zip(iters[:-1], iters[1:])):
            # import gc
            # gc.collect()
            df = froot[tree_name].pandas.df(
                flatten=False,
                branches=branches_to_read,
                entrystart=start,
                entrystop=stop,
            )
            df.index = index[start:stop]
            print(start, "-", stop, "/", num_entries)
            df_after = extract_features(
                df,
                features_tracks=features_tracks,
                features_sv=features_sv,
                n_tracks=n_tracks,
                n_sv=n_sv,
                sortby_tracks=sortby_tracks,
                sortby_sv=sortby_sv,
                sorting_mode_tracks=sorting_mode_tracks,
                sorting_mode_sv=sorting_mode_sv,
            )
            if data_type == "mc":
                df_after.insert(loc=0, column="weight_pythia", value=weight)
            df_after = convert_float64_to_float32(df_after)
            # df_after = df_after_cur if df_after is None else pd.concat([df_after, df_after])

            if output_fname.lower() == "split":
                fname_out = fname_in.replace(
                    input_fname_root, output_fname_root
                ).replace(
                    "AnalysisResults.root", f"jetDataFrame_{flavour}Jets_i{i}.hdf5"
                )
                if data_type == "mc":
                    save_df_train_test(df_after, fname_out, data_columns=["Jet_Pt"])
                else:
                    save_df(df_after, fname_out, data_columns=["Jet_Pt"])
            else:
                if df_merged is None:
                    df_merged = df_after
                else:
                    df_merged = pd.concat([df_merged, df_after])

if output_fname.lower() != "split":
    if data_type == "mc":
        save_df_train_test(df_merged, output_fname, data_columns=["Jet_Pt"])
    else:
        save_df(df_after, fname_out, data_columns=["Jet_Pt"])
