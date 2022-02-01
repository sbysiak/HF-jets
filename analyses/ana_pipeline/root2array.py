import yaml
import os
import importlib
from pprint import pprint

import uproot
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sys

sys.path.insert(0, "/eos/user/s/sbysiak/.local/lib/python3.7/site-packages/")
sys.path.insert(0, "/eos/user/s/sbysiak/SWAN_projects/HF-jets/analyses/")
from helper.utils import (
    convert_float64_to_float32,
    infer_tree_name,
    save_df,
    process_files_list,
)
from helper.preprocessing import create_index


config_fname = r"ana_config.yaml"
print(f"\nReading config file: {config_fname}")
with open(config_fname) as file:
    cfg_full = yaml.load(file, Loader=yaml.FullLoader)

cfg = cfg_full["ROOT2array"]
print("Running with following parameters:")
pprint(cfg["parameters"], width=50)
input_files_param = cfg["parameters"]["list_of_input_root_files"]
branches_to_read = cfg["parameters"]["branches_to_read"]
get_feature_fname = cfg["parameters"]["get_feature_code"]
output_fname = cfg["parameters"]["output_fname"]
data_type = cfg["parameters"]["data_type"].lower()


list_of_root_files = process_files_list(input_files_param)

print("\nImporting feature-extracting code...")
get_feature_module = importlib.import_module(
    get_feature_fname.replace("/", ".").replace(".py", "")
)
get_feature = get_feature_module.get_feature
print("... import done.")


# print("\nStart ROOT files processing ...")
# df_after_lst = []
# progbar = tqdm(list_of_root_files)
# i = 0
# for fname in progbar:
#     progbar.set_description(f' processing ...{fname[-50:]}')
#     froot = uproot.open(fname)
#     flavours = ["b", "c", "udsg"] if data_type == 'mc' else ['all',]
#     for flavour in flavours:
#         tree_name = infer_tree_name(fname, flavour)
#         df = froot[tree_name].pandas.df(flatten=False, branches=branches_to_read)
#         df.index = create_index(fname, tree_name)
#         df_after = get_feature(df)
#         if "Jet_Pt" not in df_after.columns:
#             df_after = pd.concat([df["Jet_Pt"], df_after], axis=1)
#         df_after = convert_float64_to_float32(df_after)
#         df_after_lst.append(df_after)
#     if len(df_after_lst) >= 30:
#         df_merged = pd.concat(df_after_lst)
#         # save_df(df_merged, output_fname.replace('.hdf', f'__j{i}.hdf'))
#         import pickle
#         with open(output_fname.replace('.hdf', f'__k{i}.pickle'), 'wb') as f:
#             pickle.dump(df_merged, f)
#         print(output_fname.replace('.hdf', f'__k{i}.pickle'), 'was saved')
#         i += 1
#         df_after_lst = []
# print('OK')
# exit()

print("\nStart ROOT files processing ...")
df_after_lst = []
progbar = tqdm(list_of_root_files)
for fname in progbar:
    progbar.set_description(f" processing ...{fname[-50:]}")
    froot = uproot.open(fname)
    flavours = (
        ["b", "c", "udsg"]
        if data_type == "mc"
        else [
            "all",
        ]
    )
    for flavour in flavours:
        tree_name = infer_tree_name(fname, flavour)
        df = froot[tree_name].pandas.df(flatten=False, branches=branches_to_read)
        df.index = create_index(fname, tree_name)
        df_after = get_feature(df)
        if "Jet_Pt" not in df_after.columns:
            df_after = pd.concat([df["Jet_Pt"], df_after], axis=1)
        df_after = convert_float64_to_float32(df_after)
        df_after_lst.append(df_after)
print("Reading from ROOT done, now merging ...")
df_merged = pd.concat(df_after_lst)


# print("\nStart ROOT files processing ...")
# df_merged = None
# progbar = tqdm(list_of_root_files)
# for fname in progbar:
#     progbar.set_description(f' processing ...{fname[-50:]}')
#     froot = uproot.open(fname)
#     flavours = ["b", "c", "udsg"] if data_type == 'mc' else ['all',]
#     for flavour in flavours:
#         tree_name = infer_tree_name(fname, flavour)
#         df = froot[tree_name].pandas.df(flatten=False, branches=branches_to_read)
#         df.index = create_index(fname, tree_name)
#         df_after = get_feature(df)
#         if "Jet_Pt" not in df_after.columns:
#             df_after = pd.concat([df["Jet_Pt"], df_after], axis=1)
#         if df_merged is None:
#             df_merged = df_after
#         else:
#             df_merged = pd.concat([df_merged, df_after])

print("Merging done, now saving to HDF ...")
df_merged = convert_float64_to_float32(df_merged)
mem_usage_mb = df_merged.memory_usage(deep=True).sum() / 1024 / 1024
mem_usage_idx_mb = df_merged.memory_usage(deep=True)["Index"] / 1024 / 1024
print(
    f"... processing done. Memory usage = {mem_usage_mb:.2f} MB (index = {mem_usage_idx_mb:.2f} MB)"
)

df_merged = convert_float64_to_float32(df_merged)
save_df(df_merged, output_fname, format="fixed")
print("OK")
