import sys

sys.path.insert(0, "/eos/user/s/sbysiak/.local/lib/python3.7/site-packages/")
sys.path.insert(0, "/eos/user/s/sbysiak/SWAN_projects/HF-jets/analyses/")

import yaml
from pprint import pprint

import pandas as pd
from xgboost import XGBClassifier
from tqdm.auto import tqdm

from helper.utils import (
    save_df,
    process_files_list,
)
from helper.model import MultiModel


config_fname = r"ana_config.yaml"
print(f"\nReading config file: {config_fname}")
with open(config_fname) as file:
    cfg_full = yaml.load(file, Loader=yaml.FullLoader)

cfg = cfg_full["apply_model"]
print("Running with following parameters:")
pprint(cfg["parameters"], width=50)
list_of_files = process_files_list(cfg["parameters"]["list_of_input_files"])
model_fpath = cfg["parameters"]["model_fpath"]
# output_fpath = cfg["parameters"]["output_fpath"]
input_fname_root = cfg["parameters"]["input_fname_root"]
output_fname_root = cfg["parameters"]["output_fname_root"]

### load model ###
print("\nloading model ...")
if model_fpath.endswith(".model"):
    clf = XGBClassifier(use_label_encoder=False)
    clf.load_model(model_fpath)
    columns_to_read = clf.get_booster().feature_names
elif model_fpath.endswith(".zip"):
    clf = MultiModel()
    clf.load_model(model_fpath)
    columns_to_read = clf.feature_names
else:
    raise TypeError(f"Unsupported model extension: {model_fpath}")
print(columns_to_read)

### make and store predictions ###
print("\nmaking predictions ...")
df_pred = None
# mem_max_mb = 50 # keep small, loading is heavy
# i_out = 0
n_10perc = int(len(list_of_files) * 0.1)
progbar = tqdm(list_of_files)
for f in progbar:
    progbar.set_description(f" processing ...{f[-50:]}")
    output_fpath = f.replace(input_fname_root, output_fname_root).replace(
        "jetDataFrame_", "pred_"
    )
    # print(f'processing {f}')
    try:
        df_cur = pd.read_hdf(
            f,
            columns=columns_to_read
            + [
                "weight_pythia",
            ],
        )
    except Exception as e:
        print(f"apply_model.py: Error! Cannot read {f}, due to: {e}")
    y_proba_cur = clf.predict_proba(
        df_cur.drop(["weight_pythia"], axis=1, errors="ignore")
    )[:, 1]
    df_pred_cur = (
        df_cur[["Jet_Pt", "weight_pythia"]]
        if "weight_pythia" in df_cur.columns
        else df_cur[
            [
                "Jet_Pt",
            ]
        ]
    )
    df_pred_cur["proba"] = y_proba_cur
    # save_df(df_pred_cur, output_fpath, data_columns=['proba', 'Jet_Pt'], debug=(not list_of_files.index(f) % 10))
    save_df(
        df_pred_cur,
        output_fpath,
        format="fixed",
        debug=(not list_of_files.index(f) % n_10perc),
    )
    # df_pred = df_pred_cur if df_pred is None else pd.concat([df_pred, df_pred_cur])
    # mem_usage_mb = df_pred.memory_usage(deep=True).sum()/1024/1024
    # if mem_usage_mb > mem_max_mb:
    # save_df(df_pred, output_fpath.replace('.hdf', f'_i{i_out}.hdf'), data_columns=['proba', 'Jet_Pt'])
    # df_pred = None
    # i_out += 1
# save_df(df_pred, output_fpath.replace('.hdf', f'_i{i_out}.hdf'), data_columns=['proba', 'Jet_Pt'])
print("apply_model.py: ok")
