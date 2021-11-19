import comet_ml
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def get_exp_metadata(exp_id):
    exp = comet_ml.API().get(exp_id)
    idx_metadata = ['metadata' in e['fileName'] for e in exp.get_asset_list()].index(True)
    asset_id_metadata = exp.get_asset_list()[idx_metadata]['assetId']
    metadata = pickle.loads(exp.get_asset(asset_id_metadata))
    return metadata



def get_exp_booster(exp_id):
    exp = comet_ml.API().get(exp_id)
    idx_booster = ['booster' in e['fileName'] for e in exp.get_asset_list()].index(True)
    asset_id_booster = exp.get_asset_list()[idx_booster]['assetId']
    with open('tmp/tmp.model', 'wb') as f:
        f.write(exp.get_asset(asset_id_booster))
    clf = XGBClassifier()
    clf.load_model('tmp/tmp.model')
    clf.classes_ = np.array([0,1])
    # https://github.com/dmlc/xgboost/issues/2073
    clf._le = LabelEncoder().fit([0,1])
    return clf
