from sklearn.base import BaseEstimator
from sklearn.base import clone
import numpy as np
import pandas as pd
import pickle
import os
import zipfile
from xgboost import XGBClassifier


class MultiModel(BaseEstimator):
    """
    class wrapping multiple XGBoost models to be applied in various jet pt bins

    Parameters
    ----------
    init_models : xgboost.XGBClassifiers or list of xgboost.XGBClassifiers
        if single model is passed it clones it - traning information is lost, only hyperparameters survive
        if list of models is passed then the internal models are initialized with those (potentially) trained models
    pt_bins : list
        pt binning
    """

    def __init__(self, init_models=None, pt_bins=None):
        self.pt_bins = pt_bins
        self.models = {}
        self.feature_names = None
        if pt_bins is None and init_models is None:
            return
        if hasattr(init_models, "__iter__"):
            # init_models is array of models
            for pt_low, pt_high, model in zip(
                self.pt_bins[:-1], self.pt_bins[1:], init_models
            ):
                self.models[(pt_low, pt_high)] = model
                if self.feature_names is None:
                    self.feature_names = model.get_booster().feature_names
                else:
                    assert self.feature_names == model.get_booster().feature_names
        else:
            # init_models is an unfitted model
            for pt_low, pt_high in zip(self.pt_bins[:-1], self.pt_bins[1:]):
                self.models[(pt_low, pt_high)] = clone(init_models)

    def validate_pt(self, X_pt):
        if any(X_pt <= self.pt_bins[0]) or any(X_pt >= self.pt_bins[-1]):
            print("WARNING: there are jets with pT beyond assumed bins")
            print("for training: they will not be used")
            print("for inference: they will return dummy values (-1)")

    def validate_features(self, df):
        if self.feature_names is None:
            print(
                "Cannot validate feature names. No feature names saved during training."
            )
        elif self.feature_names != df.columns.to_list():
            raise ValueError(
                f"Error: Mismatch in feature names. In model: {self.feature_names}\nwhile in data: {df.columns.to_list()}"
            )

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, np.ndarray):
            # assume Jet_Pt is first column
            X_pt = X[:, 0]
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.to_list()
            X_pt = X["Jet_Pt"].to_numpy()
            X = X.to_numpy()
        else:
            raise TypeError
        self.validate_pt(X_pt)

        for ptbin in self.models.keys():
            mask = (X_pt >= ptbin[0]) & (X_pt < ptbin[1])
            print(f"training model for bin {ptbin} on {sum(mask)} samples")
            X_sel = X[mask, :]
            y_sel = y[mask]
            w_sel = None if sample_weight is None else sample_weight[mask]
            self.models[ptbin].fit(X_sel, y_sel, sample_weight=w_sel)

    def predict(self, X):
        raise NotImplementedError("same stuff missing compared to predict_proba()")
        if isinstance(X, np.ndarray):
            # assume Jet_Pt is first column
            X_pt = X[:, 0]
        elif isinstance(X, pd.DataFrame):
            self.validate_features(X)
            X_pt = X["Jet_Pt"].to_numpy()
            X = X.to_numpy()
        else:
            raise TypeError
        self.validate_pt(X_pt)

        y_pred = np.ones([len(X), 1]) * -1

        for ptbin in self.models.keys():
            mask = (X_pt >= ptbin[0]) & (X_pt < ptbin[1])
            X_sel = X[mask, :]
            print(len(X_sel))
            y_pred_sel = self.models[ptbin].predict(X_sel).reshape(len(X_sel), 1)
            y_pred[mask] = y_pred_sel
        return y_pred

    def predict_proba(self, X, iteration_range=None):
        if isinstance(X, np.ndarray):
            # assume Jet_Pt is first column
            X_pt = X[:, 0]
        elif isinstance(X, pd.DataFrame):
            self.validate_features(X)
            X_pt = X["Jet_Pt"].to_numpy()
            X = X.to_numpy()
        else:
            raise TypeError
        self.validate_pt(X_pt)
        y_proba = np.ones([len(X), 2]) * -1

        for ptbin in self.models.keys():
            mask = (X_pt >= ptbin[0]) & (X_pt < ptbin[1])
            if np.sum(mask) == 0:
                continue
            X_sel = X[mask, :]
            y_proba_sel = self.models[ptbin].predict_proba(
                X_sel, iteration_range=iteration_range
            )
            y_proba[mask, :] = y_proba_sel
        return y_proba

    def save_model(self, fname):
        fnames = []
        fname_noext = fname.rsplit(".", 1)[0]
        for ptbin, model in self.models.items():
            fname_model = fname_noext + f"_pt-{ptbin[0]}-{ptbin[1]}.model"
            model.save_model(fname_model)
            fnames.append(fname_model)
        fname_meta = fname_noext + "_meta.pickle"
        with open(fname_meta, "wb") as f:
            meta_dict = {"pt_bins": self.pt_bins, "feature_names": self.feature_names}
            pickle.dump(meta_dict, f)
        fnames.append(fname_meta)
        with zipfile.ZipFile(fname_noext + ".zip", "w") as zipf:
            for f in fnames:
                zipf.write(f)
                os.remove(f)

    def load_model(self, fname):
        fname_noext = fname.rsplit(".", 1)[0]
        fname_zip = fname_noext + ".zip"
        with zipfile.ZipFile(fname_zip, "r") as zipf:
            zipped_files = zipf.namelist()
            zipf.extractall()
        fname_meta = [f for f in zipped_files if "meta.pickle" in f][0]
        zipped_files.remove(fname_meta)
        with open(fname_meta, "rb") as f:
            meta_dict = pickle.load(f)
            self.pt_bins = meta_dict["pt_bins"]
            self.feature_names = meta_dict["feature_names"]
        os.remove(fname_meta)
        for fname_model in zipped_files:
            ptbin = tuple(
                [int(pt) for pt in fname_model.replace(".model", "").split("-")[-2:]]
            )
            with open(fname_model, "rb") as f:
                clf_loaded = XGBClassifier(use_label_encoder=False)
                clf_loaded.load_model(fname_model)
                self.models[ptbin] = clf_loaded
            os.remove(fname_model)
        try:
            pt_bins_from_models = sorted(
                list(set([pt for ptbin in self.models.keys() for pt in ptbin]))
            )
            assert self.pt_bins == pt_bins_from_models
        except:
            raise ValueError(
                "Error: mismatch between ptbins", self.pt_bins, pt_bins_from_models
            )
