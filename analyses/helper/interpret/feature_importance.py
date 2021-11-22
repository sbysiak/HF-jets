import numpy as np


def feature_importance_report(
    feat_importances, feat_names, print_raw=10, importance_type="", norm=True
):
    """prints feature importances: raw and aggregated by obj type (SV/track/jet)

    Parameters
    ----------
    feat_importances : iterable of numbers
        features' importances
    feat_names : iterable of strings
        array of features' names used in training
    print_raw : int
        how many unaggregated features should be printed
        (table with aggregations is printed anyway)
    importance_type : str
        description of importance type, e.g. \"permutation imp\" or \"XGB's weigth\"
    norm : bool
        if importances should be normalized to 1

    Returns
    -------
    None
    """
    ## TODO: add "Jet_Shape_" as category

    def featname2objType(featname):
        if featname.startswith("Jet_SecVtx"):
            return "SV"
        if featname.startswith("Jet_Track"):
            return "track"
        else:
            return "jet"

    def featname2iObj(featname):
        objType = featname2objType(featname)
        if objType == "jet":
            return None
        else:
            return objType + "_" + featname.split("_")[2]

    def featname2observable(featname):
        objType = featname2objType(featname)
        if objType == "jet":
            return None
        else:
            return objType + "_" + featname.split("_")[3]

    feature_importances = {}
    imp_total = np.sum(list(feat_importances)) if norm else 1
    if print_raw:
        print(f"*** Raw {importance_type}: ***")
    for i, (feat_imp, feat_name) in enumerate(
        sorted(
            zip(map(lambda x: round(x, 4), feat_importances), feat_names), reverse=True
        )
    ):
        if i < print_raw:
            print(f"{feat_name:<55s}| {feat_imp / imp_total:.4f}")
        for k in [
            featname2objType(feat_name),
            featname2iObj(feat_name),
            featname2observable(feat_name),
        ]:
            if k == None:
                continue
            if k in feature_importances.keys():
                feature_importances[k] += feat_imp / imp_total
            else:
                feature_importances[k] = feat_imp / imp_total
    if print_raw:
        print("\n\n")

    print(f"*** Aggregated features: ***")
    print("{:<20s} | ".format("category") + importance_type)
    max_sv = max(
        [int(name.split("_")[2]) for name in feat_names if "Jet_SecVtx_" in name],
        default=0,
    )
    max_track = max(
        [int(name.split("_")[2]) for name in feat_names if "Jet_Track_" in name],
        default=0,
    )
    for k, v in sorted(feature_importances.items()):
        if "_" not in k:
            print("--" * 13)
        print(f"{k:<20s} | {v:.3f}")
        if "_" not in k or k == f"SV_{max_sv}" or k == f"track_{max_track}":
            print(" " * 20 + " | ")
    print("--" * 13 + "\n\n\n")
