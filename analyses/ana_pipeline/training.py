import comet_ml

import sys

sys.path.insert(0, "/eos/user/s/sbysiak/.local/lib/python3.7/site-packages/")
sys.path.insert(0, "/eos/user/s/sbysiak/SWAN_projects/HF-jets/analyses/")

import yaml
import numpy as np
import pandas as pd
from pprint import pprint
import subprocess
import os
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from helper.utils import process_files_list, save_plot, signal_eff
from helper.preprocessing import calc_njets, get_hdf5_nrows
from helper.model import MultiModel
from helper.plotting import plot_tagging_eff, plot_score_distr


def basic_perf_plots(clf, X_train, y_train, X_test, y_test, plot_path_root):
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax = plot_tagging_eff(
        y_train, y_train_proba, label="$b$ vs $c+udsg$ train", color="b", ax=ax
    )
    ax = plot_tagging_eff(
        y_test, y_test_proba, label="$b$ vs $c+udsg$ test", color="r", ax=ax
    )
    ax.set_ylim(bottom=1e-5)
    mistag_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
    for mistag_rate in mistag_rates:
        eff = signal_eff(y_test, y_test_proba, mistag_rate)
        ax.text(eff + 0.05, mistag_rate, f"eff @mistag={mistag_rate:.0e}: {eff:.3f}")
    #         exp.log_metric(f'tagEff@mistag_{mistag_rate:.0e}', eff)
    save_plot(os.path.join(plot_path_root, "tagging_eff_vs_mistag_rate"))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax = plot_score_distr(
        y_train,
        y_train_proba,
        linestyle=":",
        ax=ax,
        mistag_thresholds=None,
        label="train",
        nbins=50,
    )
    ax = plot_score_distr(
        y_test,
        y_test_proba,
        linestyle="-",
        ax=ax,
        lw=2,
        mistag_thresholds=None,
        label="test",
        nbins=50,
    )
    ax.legend(loc="upper center")
    ax.set_yscale("linear")
    save_plot(os.path.join(plot_path_root, "score_distr"))

    fig, ax = plt.subplots(figsize=(5, 3))
    max_ntrees = 0
    if isinstance(clf, XGBClassifier):
        max_ntrees = clf.get_params()["n_estimators"]
    elif isinstance(clf, MultiModel):
        max_ntrees = clf.models[list(clf.models)[0]].get_params()["n_estimators"]
    ntrees = [1, 3, 6, 10, 20, 30, 40, 60, 80, 100]
    ntrees += [int(max_ntrees * frac) for frac in np.linspace(0, 1, 11)]
    ntrees = [n for n in ntrees if n <= max_ntrees and n > 0]  # n=0 -> use all trees
    ntrees = sorted(set(ntrees))
    scores_train = []
    scores_test = []
    for n in ntrees:
        y_train_proba_ntrees = clf.predict_proba(X_train, iteration_range=(0, n))[:, 1]
        y_test_proba_ntrees = clf.predict_proba(X_test, iteration_range=(0, n))[:, 1]
        scores_train.append(roc_auc_score(y_train, y_train_proba_ntrees))
        scores_test.append(roc_auc_score(y_test, y_test_proba_ntrees))
    ax.plot(ntrees, scores_train, ".-", label="train", color="b")
    ax.plot(ntrees, scores_test, ".-", label="test", color="r")
    ax.legend()
    ax.set_xlabel("num. trees")
    ax.set_ylabel("ROC AUC")
    ax.grid()
    save_plot(os.path.join(plot_path_root, "learning_curve"))


def loading_test(clf, fname_model, X_test):
    if isinstance(clf, XGBClassifier):
        clf_loaded = xgb.XGBClassifier(use_label_encoder=False)
    elif isinstance(clf, MultiModel):
        clf_loaded = MultiModel()
    else:
        raise TypeError(f"Unsupported model type: {clf.__class__}")
    clf_loaded.load_model(fname_model)
    assert all(
        clf.predict_proba(X_test)[:, 1] == clf_loaded.predict_proba(X_test)[:, 1]
    )


config_fname = r"ana_config.yaml"
print(f"\nReading config file: {config_fname}")
with open(config_fname) as file:
    cfg_full = yaml.load(file, Loader=yaml.FullLoader)
cfg = cfg_full["training"]
print("Running with following parameters:")
pprint(cfg["parameters"], width=150)

list_of_train_files = process_files_list(cfg["parameters"]["list_of_input_files_train"])
list_of_test_files = process_files_list(cfg["parameters"]["list_of_input_files_test"])
output_dir = cfg["parameters"]["output_dir"]
xgb_params = cfg["parameters"]["xgboost"]

### memory_method: ###
memory_method = cfg["parameters"]["memory_method"]
if memory_method == "internal":
    pass
elif memory_method == "external":
    raise NotImplementedError
elif memory_method == "incremental":
    raise NotImplementedError
else:
    raise ValueError
### weights: ###
training_weights = cfg["parameters"]["training_weights"].lower()
if training_weights == "pythia":
    pass
elif training_weights is None or training_weights == "none" or training_weights == "no":
    training_weights = None
else:
    raise NotImplementedError
### num models: ###
num_models = cfg["parameters"]["num_models"]
if num_models == "single":
    pass
else:
    multimodel_pt_bins = num_models
### fractions: ###
fractions = cfg["parameters"]["input_data"]["fractions"]
if not fractions:
    print("\nCounting jets")
    ratio_b_to_rest = cfg["parameters"]["input_data"]["ratio_b_to_rest"]
    ratio_c_to_udsg = cfg["parameters"]["input_data"]["ratio_c_to_udsg"]
    n_b = cfg["parameters"]["input_data"]["n_b"]
    if n_b <= 1:
        njets = calc_njets(
            list_of_train_files,
            trainset_frac_b=n_b,
            ratio_c_to_udsg=ratio_c_to_udsg,
        )
        fractions = njets.loc["trainset fraction"].to_dict()
    else:
        njets = calc_njets(
            list_of_train_files, n_b=n_b, ratio_c_to_udsg=ratio_c_to_udsg
        )
        fractions = njets.loc["trainset fraction"].to_dict()

    fractions["c"] /= ratio_b_to_rest
    fractions["udsg"] /= ratio_b_to_rest
    print(f"\n{njets}")
print(f"\n\nFractions: {fractions}")
### columns: ###
n_sv = cfg["parameters"]["input_data"]["n_sv"]
n_tracks = cfg["parameters"]["input_data"]["n_tracks"]
training_columns_jet = cfg["parameters"]["input_data"]["training_columns_jet"]
training_columns_tracks = cfg["parameters"]["input_data"]["training_columns_tracks"]
training_columns_sv = cfg["parameters"]["input_data"]["training_columns_sv"]

training_columns_jet = training_columns_jet if training_columns_jet else []
training_columns_sv = training_columns_sv if training_columns_sv else []
training_columns_tracks = training_columns_tracks if training_columns_tracks else []

columns_to_read = training_columns_jet
if training_weights == "pythia":
    columns_to_read.insert(0, "weight_pythia")
available_columns = pd.read_hdf(list_of_train_files[0], stop=1).columns

patterns = []
for col in training_columns_tracks + training_columns_sv:
    col = col.replace("Jet_Track_", "").replace("Jet_SecVtx_", "")
    patterns.append(col + "__sortby")
for col in available_columns:
    for pat in patterns:
        if pat in col:
            idx = int(col.split("_")[2])
            if ("Jet_Track" in col and idx < n_tracks) or (
                "Jet_SecVtx" in col and idx < n_sv
            ):
                columns_to_read.append(col)
print(f"\n\n columns to be read (N={len(columns_to_read)}):")
pprint(columns_to_read)

print("Config done.\n\nReading data files:")
df_train = None
labels_train = []
for f in tqdm(list_of_train_files):
    nrows = get_hdf5_nrows(f)
    if not nrows:
        continue
    if "bJets" in f:
        n_read = int(nrows * fractions["b"])
    elif "cJets" in f:
        n_read = int(nrows * fractions["c"])
    elif "udsgJets" in f:
        n_read = int(nrows * fractions["udsg"])
    # print(f"reading {n_read} rows from {f}")
    try:
        df_cur = pd.read_hdf(f, stop=n_read, columns=columns_to_read)
        labels_train += [
            1 if "bJets" in f else 0,
        ] * n_read
    except:
        print("\nERROR: cannot read ", f)
    if df_train is None:
        df_train = df_cur
    else:
        df_train = pd.concat([df_train, df_cur])

print("^^ train reading done,\n now test:")
df_test = None
labels_test = []
for f in tqdm(list_of_test_files):
    nrows = get_hdf5_nrows(f)
    if not nrows:
        continue
    if "bJets" in f:
        n_read = int(nrows * fractions["b"])
    elif "cJets" in f:
        n_read = int(nrows * fractions["c"])
    elif "udsgJets" in f:
        n_read = int(nrows * fractions["udsg"])
    # print(f"reading {n_read} rows from {f}")
    try:
        df_cur = pd.read_hdf(f, stop=n_read, columns=columns_to_read)
        labels_test += [
            1 if "bJets" in f else 0,
        ] * n_read
    except:
        print("\nERROR: cannot read ", f)
    if df_test is None:
        df_test = df_cur
    else:
        df_test = pd.concat([df_test, df_cur])
df_train["label"] = labels_train
df_test["label"] = labels_test

size_train_mb = df_train.memory_usage(deep=True).sum() / 1024 / 1024
size_test_mb = df_test.memory_usage(deep=True).sum() / 1024 / 1024
n_pos_train = np.sum(labels_train)
n_pos_test = np.sum(labels_test)
n_tot_train = len(labels_train)
n_tot_test = len(labels_test)
print(
    f"data reading done.\nSize of df: train={size_train_mb:.1f} MB, test={size_test_mb:.1f} MB\n#positive samples / #total samples: train: {n_pos_train}/{n_tot_train}, test: {n_pos_test}/{n_tot_test}"
)
print("Creating X,y for training")


y_train = df_train["label"].to_numpy()
X_train = df_train.drop(["label", "weight_pythia"], axis=1, errors="ignore")
if training_weights == "pythia":
    w_train = df_train["weight_pythia"]
elif training_weights is None:
    w_train = np.ones_like(y_train)
else:
    raise NotImplementedError

y_test = df_test["label"].to_numpy()
X_test = df_test.drop(["label", "weight_pythia"], axis=1, errors="ignore")
if training_weights == "pythia":
    w_test = df_test["weight_pythia"]
elif training_weights is None:
    w_test = np.ones_like(y_test)
else:
    raise NotImplementedError

print("\nTraining model")
if num_models == "single":
    clf = XGBClassifier(**xgb_params, verbosity=0, silent=0, use_label_encoder=False)
    clf.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[w_train, w_test],
        verbose=True,
    )
else:
    clf_base = XGBClassifier(
        **xgb_params, verbosity=0, silent=0, use_label_encoder=False
    )
    clf = MultiModel(clf_base, multimodel_pt_bins)
    clf.fit(X_train, y_train, sample_weight=w_train)

print("\nSaving model and generating plots")
if hasattr(clf, "feature_names"):
    assert not clf.feature_names is None
else:
    assert not clf.get_booster().feature_names is None
fname_model = os.path.join(output_dir, "b_tagger.model")
clf.save_model(fname_model)
loading_test(clf, fname_model, X_test)
basic_perf_plots(
    clf, X_train, y_train, X_test, y_test, os.path.join(output_dir, "plots")
)
print("training.py: ok")
