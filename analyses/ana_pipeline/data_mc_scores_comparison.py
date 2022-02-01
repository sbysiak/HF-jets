import sys

sys.path.insert(0, "/eos/user/s/sbysiak/.local/lib/python3.7/site-packages/")
sys.path.insert(0, "/eos/user/s/sbysiak/SWAN_projects/HF-jets/analyses/")

import yaml
import os
import pathlib
import gc
from tqdm.auto import tqdm
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ROOT

# from scipy.stats import pearsonr
from array import array

from helper.utils import process_files_list, save_plot
from helper.preprocessing import get_hdf5_nrows


config_fname = r"ana_config.yaml"
print(f"\nReading config file: {config_fname}")
with open(config_fname) as file:
    cfg_full = yaml.load(file, Loader=yaml.FullLoader)

cfg = cfg_full["data_mc_scores_comparison"]
print("Running with following parameters:")
pprint(cfg["parameters"], width=50)
list_of_files_mc = process_files_list(cfg["parameters"]["list_of_input_files_mc"])
list_of_files_data = process_files_list(cfg["parameters"]["list_of_input_files_data"])
output_dir = cfg["parameters"]["output_dir"]
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
cut_variation_args_lst = cfg["parameters"]["cut_variation_args"]
if not hasattr(cut_variation_args_lst[-1], "__iter__"):
    cut_variation_args_lst = [
        cut_variation_args_lst,
    ]
pt_binning = cfg["parameters"]["pt_binning"]
nbins_proba = cfg["parameters"]["nbins_proba"]

### read scores ###
print("Reading scores from files:")
df_data_list = []
for f in tqdm(list_of_files_data):
    df_cur = pd.read_hdf(f)
    df_data_list.append(df_cur)
df_data = pd.concat(df_data_list)
del df_data_list
df_mc_list = []
for f in tqdm(list_of_files_mc):
    df_cur = pd.read_hdf(f)
    df_mc_list.append(df_cur)
df_mc = pd.concat(df_mc_list)
del df_mc_list


### np -> ROOT ###
print("Filling ROOT histos:")

ROOT.gStyle.SetOptStat(11)
ROOT.gStyle.SetStatY(0.9)
ROOT.gStyle.SetStatX(0.9)
ROOT.gStyle.SetStatW(0.2)
ROOT.gStyle.SetStatH(0.5)

nbins_pt = 100
h2data = ROOT.TH2F("h2data", "h2data", nbins_proba, 0, 1, nbins_pt, 0, 100)
h2mc = ROOT.TH2F("h2mc", "h2mc", nbins_proba, 0, 1, nbins_pt, 0, 100)
h2mc.Sumw2()
h2data.Sumw2()

h2data.FillN(
    len(df_data),
    array("d", df_data[["proba"]].to_numpy()),
    array("d", df_data[["Jet_Pt"]].to_numpy()),
    array("d", np.ones_like(df_data[["proba"]].to_numpy())),
)

h2mc.FillN(
    len(df_mc),
    array("d", df_mc[["proba"]].to_numpy()),
    array("d", df_mc[["Jet_Pt"]].to_numpy()),
    array("d", df_mc[["weight_pythia"]].to_numpy()),
)

# for p,pt in tqdm(df_data[['proba', 'Jet_Pt']].to_numpy()):
# h2data.Fill(p,pt)
# for p,pt,w in tqdm(df_mc[['proba','Jet_Pt', 'weight_pythia']].to_numpy()):
# h2mc.Fill(p,pt,w)
h2data.SaveAs(os.path.join(output_dir, f"h2probaVsPt_data.root"))
h2mc.SaveAs(os.path.join(output_dir, f"h2probaVsPt_mc.root"))

### plot scores distr ###
def make_pads(i):
    irow = i // ncols
    icol = i % ncols
    pad1 = ROOT.TPad(
        f"pad_{i}",
        f"pad_{i}",
        icol * colw,
        (nrows - irow - 1 + 0.4) * rowh,
        (icol + 1) * colw,
        (nrows - irow) * rowh,
    )
    pad2 = ROOT.TPad(
        f"pad_{i}_ratio",
        f"pad_{i}_ratio",
        icol * colw,
        (nrows - irow - 1) * rowh,
        (icol + 1) * colw,
        (nrows - irow - 1 + 0.4) * rowh,
    )
    pad1.SetBottomMargin(0.0)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.25)
    return pad1, pad2


ptbins = zip(pt_binning[:-1], pt_binning[1:])
n_ptbins = len(pt_binning) - 1
ncols = 3 if (n_ptbins) > 4 else 2
nrows = int(np.ceil(n_ptbins / ncols))
colw = 1 / ncols
rowh = 1 / nrows

c = ROOT.TCanvas("c", "c", 600 * ncols, 300 * nrows)
line = ROOT.TLine(0, 1, 1, 1)
line.SetLineStyle(ROOT.kDashed)
for i, ptbin in enumerate(ptbins):
    c.cd()
    pad1, pad2 = make_pads(i)
    pad1.Draw()
    pad2.Draw()

    bin_low, bin_high = h2mc.GetYaxis().FindBin(ptbin[0]), h2mc.GetYaxis().FindBin(
        ptbin[1] - 1e-8
    )
    hmc = h2mc.ProjectionX("_px", bin_low, bin_high, "e")
    hdata = h2data.ProjectionX("_px", bin_low, bin_high, "e")
    hmc.Scale(1 / hmc.Integral())
    hdata.Scale(1 / hdata.Integral())
    hratio = hmc.Clone("hratio")
    hratio.Divide(hdata)

    hmc.SetTitle("")
    hmc.SetLineColor(ROOT.kBlue)
    hmc.SetLineWidth(3)
    hdata.SetTitle("")
    hdata.SetStats(0)
    hdata.SetLineColor(ROOT.kBlack)
    hdata.SetLineWidth(3)
    hdata.GetYaxis().SetLabelSize(0.08)
    hdata.GetYaxis().SetTitleSize(0.08)
    hdata.GetYaxis().SetTitleOffset(0.6)
    hdata.GetYaxis().SetTitle("probability density")
    hdata.SetMaximum(max(hdata.GetMaximum(), hmc.GetMaximum()) * 1.1)
    hratio.SetTitle("")
    hratio.SetStats(0)
    hratio.SetLineWidth(3)
    hratio.GetXaxis().SetTitle("BDT score")
    hratio.GetYaxis().SetTitle("ratio to data")
    hratio.GetXaxis().SetLabelSize(0.12)
    hratio.GetXaxis().SetTitleSize(0.12)
    hratio.GetYaxis().SetLabelSize(0.12)
    hratio.GetYaxis().SetTitleSize(0.12)
    hratio.GetYaxis().SetTitleOffset(0.3)

    pad1.cd()
    hdata.DrawCopy()
    hmc.DrawCopy("same")
    leg = pad1.BuildLegend(0.75, 0.6, 0.90, 0.89)
    leg.SetLineWidth(0)
    leg.Clear()
    leg.AddEntry(hdata, "data", "l")
    leg.AddEntry(hmc, "MC", "l")
    txt_pt = f"{ptbin[0]} < " + "#it{p}_{T}^{jet}" + f" < {ptbin[1]} " + " GeV/#it{c}"
    ks_dist = hmc.KolmogorovTest(hdata, "M")
    txt_ks = f"KS = {ks_dist:.3f}"
    latex = ROOT.TLatex()
    latex.SetTextSize(0.1)
    latex.SetTextAlign(31)
    latex.DrawLatexNDC(0.65, 0.78, txt_pt)
    latex.DrawLatexNDC(0.65, 0.65, txt_ks)

    pad2.cd()
    hratio.DrawCopy("axis")
    grshade = ROOT.TGraph()
    m = 0.2
    grshade.SetFillStyle(3001)
    grshade.SetFillColor(ROOT.kGray + 1)
    grshade.DrawGraph(
        4, array("d", [0, 1, 1, 0]), array("d", [1 + m, 1 + m, 1 - m, 1 - m]), "f"
    )
    line.DrawLine(0, 1, 1, 1)
    hratio.DrawCopy("same")

    # hdata.SaveAs(os.path.join(output_dir, f'scores_dist_data_pt{ptbin[0]}-{ptbin[1]}.root'))
    # hmc.SaveAs(os.path.join(output_dir, f'scores_dist_mc_pt{ptbin[0]}-{ptbin[1]}.root'))
c.Draw()
c.SaveAs(os.path.join(output_dir, f"scores_dist.root"))
c.SaveAs(os.path.join(output_dir, f"scores_dist.png"))


### cut variation aka Barlow test ###


def calc_metrics(ts, y_proba, y_true, weights=None):
    if not hasattr(ts, "__iter__"):
        ts = [
            ts,
        ]
    if weights is None:
        weights = np.ones_like(y_proba)
    purities = []
    fprs = []
    tprs = []
    n_tot_sig = np.sum(weights[y_true == 1])
    n_tot_bckg = np.sum(weights[y_true == 0])
    for t in ts:
        n_pass_sig = np.sum(weights[(y_true == 1) & (y_proba >= t)])
        n_pass_bckg = np.sum(weights[(y_true == 0) & (y_proba >= t)])
        tprs.append(n_pass_sig / n_tot_sig)
        fprs.append(n_pass_bckg / n_tot_bckg)
        purities.append(n_pass_sig / (n_pass_sig + n_pass_bckg))
    if len(ts) == 1:
        return ts[0], tprs[0], fprs[0], purities[0]
    return np.array(ts), np.array(tprs), np.array(fprs), np.array(purities)


def cut_variation_test(
    y_proba_data,
    y_true_mc,
    y_proba_mc,
    weights_mc=None,
    mode="threshold",
    x_vars=None,
    cut_ref=0.4,
    delta_cut=0.1,
    n_inter_points=5,
    title_suffix="",
    verbose=False,
    add_fits=False,
    axes=None,
    results=None,
    res_prefix="",
    n_thresholds=50,
):

    ### MODE: threshold, eff, fpr, purity

    ###
    ### roc vs precision-recall curves: https://stackoverflow.com/a/30084402/5770251

    #     from ..apply import calc_mc_correction_uncert

    def calc_mc_correction(y_true, y_proba, threshold, sample_weight=None):
        y_pred = y_proba > threshold
        if sample_weight is None:
            return np.sum(y_true) / np.sum(y_pred)
        else:
            return np.dot(y_true, sample_weight) / np.dot(y_pred, sample_weight)

    def describe_point(idx):
        return f"({thresholds[idx]:.5f}, {tprs[idx]:.3f}, {fprs[idx]:.4f}, {purities[idx]:.3f})"

    ts = np.hstack(
        [
            np.linspace(0, 0.5, int(n_thresholds * 0.1)),
            np.linspace(0.5, 0.8, int(n_thresholds * 0.1)),
            np.linspace(0.8, 0.95, int(n_thresholds * 0.4)),
            np.linspace(0.95, 1, int(n_thresholds * 0.4)),
        ]
    )
    ts = sorted(ts)
    thresholds, tprs, fprs, purities = calc_metrics(
        ts, y_proba_mc, y_true_mc, weights_mc
    )

    if mode == "threshold":
        all_vals = thresholds
        label = "score threshold"
    elif mode == "fpr":
        all_vals = fprs
        label = "mistagging rate"
    elif mode == "tpr":
        all_vals = tprs
        label = "efficiency"
    elif mode == "purity":
        all_vals = purities
        label = "purity"
    else:
        raise ValueError(
            '`mode` has to be one of the following: "threshold", "fpr", "tpr", "purity"'
        )

    if mode == "fpr":
        cut_low, cut_high = cut_ref / delta_cut, cut_ref * delta_cut
        cut_values = (
            list(
                np.logspace(np.log10(cut_low), np.log10(cut_ref), n_inter_points + 1)[
                    :-1
                ]
            )
            + [
                cut_ref,
            ]
            + list(
                np.logspace(np.log10(cut_ref), np.log10(cut_high), n_inter_points + 1)[
                    1:
                ]
            )
        )
    else:
        cut_low, cut_high = cut_ref - delta_cut, cut_ref + delta_cut
        cut_values = (
            list(np.linspace(cut_low, cut_ref, n_inter_points + 1)[:-1])
            + [
                cut_ref,
            ]
            + list(np.linspace(cut_ref, cut_high, n_inter_points + 1)[1:])
        )

    indices = []
    for cut in cut_values:
        indices.append(np.nanargmin(abs(all_vals - cut)))
    indices = np.array(indices)

    threshold_values = thresholds[indices]
    threshold_ref = threshold_values[n_inter_points]

    ratio_arr = []
    pt_arr = []
    err = []
    sigma_uncorr_arr = []
    n_bjets_arr = []

    n_bjets_ref_raw = np.sum(y_proba_data > threshold_ref)
    #     mc_correction_ref, mc_correction_ref_err = calc_mc_correction_uncert(y_true_mc, y_proba_mc, threshold_ref, sample_weight=weights_mc, N_bootstrap=10)
    mc_correction_ref = calc_mc_correction(
        y_true_mc, y_proba_mc, threshold_ref, sample_weight=weights_mc
    )
    n_bjets_ref_corr = n_bjets_ref_raw * mc_correction_ref

    #     sigma_ref = np.sqrt(  (np.sqrt(n_bjets_ref_raw)*mc_correction_ref)**2  + (mc_correction_ref_err*n_bjets_ref_raw)**2   )
    sigma_ref = np.sqrt(n_bjets_ref_raw) * mc_correction_ref
    #     if verbose: print(f'sigma ref = {sigma_ref:.3f} \t\t data stat err ={(np.sqrt(n_bjets_ref_raw)*mc_correction_ref):.3f}, MC correction err={(mc_correction_ref_err*n_bjets_ref_raw):.3f}')
    if verbose:
        print(
            f"sigma ref = {sigma_ref:.3f} \t\t data stat err ={(np.sqrt(n_bjets_ref_raw)*mc_correction_ref):.3f}"
        )

    for idx, threshold in zip(indices, threshold_values):
        n_bjets_cur_raw = np.sum(y_proba_data > threshold)
        #         mc_correction, mc_correction_err = calc_mc_correction_uncert(y_true_mc, y_proba_mc, threshold, sample_weight=weights_mc)
        mc_correction = calc_mc_correction(
            y_true_mc, y_proba_mc, threshold, sample_weight=weights_mc
        )
        n_bjets_cur_corr = n_bjets_cur_raw * mc_correction

        #         sigma_cur = np.sqrt(  (np.sqrt(n_bjets_cur_raw)*mc_correction)**2  + (mc_correction_err*n_bjets_cur_raw)**2   )
        sigma_cur = np.sqrt(n_bjets_cur_raw) * mc_correction
        sigma_uncorr = np.sqrt(np.abs(sigma_ref**2 - sigma_cur**2))

        n_bjets_arr.append(n_bjets_cur_corr)
        sigma_uncorr_arr.append(sigma_uncorr)

        #         if verbose: print(f'{describe_point(idx)}:  {n_bjets_cur_raw} (+/-{np.sqrt(n_bjets_cur_raw):.2f}) * {mc_correction:.3f} (+/-{mc_correction_err:.3f}) = {n_bjets_cur_corr:.1f} ({sigma_cur:.2f}), uncorr. err: {sigma_uncorr:.2f} {"<-" if abs(threshold-threshold_ref)<1e-8 else ""}')
        if verbose:
            print(
                f'{describe_point(idx)}:  {n_bjets_cur_raw} (+/-{np.sqrt(n_bjets_cur_raw):.2f}) * {mc_correction:.3f} = {n_bjets_cur_corr:.1f} ({sigma_cur:.2f}), uncorr. err: {sigma_uncorr:.2f} {"<-" if abs(threshold-threshold_ref)<1e-8 else ""}'
            )

    if x_vars is None:
        x_vars = (mode,)
    if isinstance(x_vars, str):
        x_vars = (x_vars,)

    if axes is None:
        fig, axes = plt.subplots(
            nrows=len(x_vars), figsize=(10 if add_fits else 7, 5 + 3 * len(x_vars))
        )
    if not hasattr(axes, "__iter__"):
        axes = [
            axes,
        ]

    for ax, x_var in zip(axes, x_vars):
        if x_var == "threshold":
            x_vals = thresholds[indices]
            xlabel = "score threshold"
        elif x_var == "fpr":
            x_vals = fprs[indices]
            xlabel = "mistagging rate"
            ax.set_xscale("log")
        elif x_var == "tpr":
            x_vals = tprs[indices]
            xlabel = "efficiency"
        elif x_var == "purity":
            x_vals = purities[indices]
            xlabel = "purity"
        else:
            raise ValueError(
                '`x_var` has to contain one of the following: "threshold", "fpr", "tpr", "purity" or be None'
            )

        x_ref = x_vals[n_inter_points]
        ax.errorbar(
            x_vals,
            n_bjets_arr,
            yerr=sigma_uncorr_arr,
            fmt="o",
            ms=4,
            lw=3,
            label="uncorr. error",
        )
        ax.errorbar(x_ref, n_bjets_ref_corr, sigma_ref, fmt="o", color="k", ms=8, lw=3)

        ax.legend()
        # ax.set_ylabel('b / incl.')
        ax.set_ylabel("number of b-jets in data")
        ax.set_xlabel("model " + xlabel)
        if x_var == x_vars[0]:
            ax.set_title(
                f"left: {describe_point(indices[0])}\ncentre: {describe_point(indices[n_inter_points])}\nright: {describe_point(indices[-1])}\n"
                + title_suffix,
                fontsize=12,
            )
        ax.grid()

        if results is not None:
            #             res_prefix = f'{res_prefix}_{mode}_{cut_ref}_{delta_cut}'
            if len(x_vars) > 1:
                res_presfix_cur = f"{res_prefix}_{x_var}"
            else:
                res_prefix_cur = res_prefix
            results[f"{res_prefix_cur}_ref_x"] = x_ref
            results[f"{res_prefix_cur}_ref_y"] = n_bjets_ref_corr
            results[f"{res_prefix_cur}_ref_err"] = sigma_ref
            results[f"{res_prefix_cur}_x"] = x_vals
            results[f"{res_prefix_cur}_y"] = n_bjets_arr
            results[f"{res_prefix_cur}_y_mean"] = np.mean(n_bjets_arr)
            results[f"{res_prefix_cur}_y_std"] = np.std(n_bjets_arr)
            results[f"{res_prefix_cur}_y_minmax"] = np.max(n_bjets_arr) - np.min(
                n_bjets_arr
            )

        if add_fits:
            ### SLOPE
            x = x_vals
            y = n_bjets_arr
            #     coef = np.polyfit(x,y,1, w=1/np.array(sigma_uncorr_arr))
            try:
                coef = np.polyfit(x, y, 1)
                poly1d_fn = np.poly1d(coef)
                # poly1d_fn is now a function which takes in x and returns an estimate for y
                ax.plot(x, poly1d_fn(x), "--k")
                ax.set_ylim(
                    bottom=ax.get_ylim()[0]
                    - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                )
                ax.text(
                    0.05,
                    0.05,
                    f"linear fit (w/o errorbars):\n{coef[0]:.2f} * x + {coef[1]:.2f}",
                    transform=ax.transAxes,
                )

                if results is not None:
                    results[f"{res_prefix_cur}_fit_lin_a"] = coef[0]
                    results[f"{res_prefix_cur}_fit_lin_b"] = coef[1]
            except Exception as e:
                print(f"Fit failed ({e})")
                results[f"{res_prefix_cur}_fit_lin_a"] = -1
                results[f"{res_prefix_cur}_fit_lin_b"] = -1
    plt.tight_layout()
    #     fig.patch.set_facecolor('white')
    return axes if len(axes) > 1 else axes[0]


def stability_report(res, ptbinning, mode=""):
    x = [(pt[0] + pt[1]) / 2 for pt in zip(ptbinning[:-1], ptbinning[1:])]
    means = [
        res[f"pT{pt[0]}-{pt[1]}{mode}_y_mean"]
        for pt in zip(ptbinning[:-1], ptbinning[1:])
    ]
    stds = [
        res[f"pT{pt[0]}-{pt[1]}{mode}_y_std"]
        for pt in zip(ptbinning[:-1], ptbinning[1:])
    ]
    stat_errs = [
        res[f"pT{pt[0]}-{pt[1]}{mode}_ref_err"]
        for pt in zip(ptbinning[:-1], ptbinning[1:])
    ]
    a_fits = [
        res[f"pT{pt[0]}-{pt[1]}{mode}_fit_lin_a"]
        for pt in zip(ptbinning[:-1], ptbinning[1:])
    ]
    minmax = [
        res[f"pT{pt[0]}-{pt[1]}{mode}_y_minmax"]
        for pt in zip(ptbinning[:-1], ptbinning[1:])
    ]

    std_over_mean = np.array(stds) / np.array(means)
    a_over_mean = abs(np.array(a_fits) / np.array(means))
    minmax_over_mean = np.array(minmax) / np.array(means)
    std_over_stat_err = np.array(stds) / np.array(stat_errs)
    a_over_stat_err = abs(np.array(a_fits)) / np.array(stat_errs)
    minmax_over_stat_err = np.array(minmax) / np.array(stat_errs)

    #     def nanpearsonr(x,y):
    #         try:
    #             return pearsonr(x,y)
    #         except:
    #             print('Catched error in pearsonr()')
    #             return 0,0
    #     corr = [ abs(nanpearsonr(res[f'pT{pt[0]}-{pt[1]}{mode}_x'], res[f'pT{pt[0]}-{pt[1]}{mode}_y'])[0])
    #             for pt in zip(ptbinning[:-1], ptbinning[1:])]
    #     pval = [ nanpearsonr(res[f'pT{pt[0]}-{pt[1]}{mode}_x'], res[f'pT{pt[0]}-{pt[1]}{mode}_y'])[1]
    #             for pt in zip(ptbinning[:-1], ptbinning[1:])]

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(13, 6))
    #     for i, (y,ylabel) in enumerate([(std_over_mean, 'stddev / mean'), (a_over_mean, 'slope / mean'), (corr, 'correlation'),
    #                  (std_over_stat_err, 'stddev / $\sigma_{stat}$'), (a_over_stat_err, 'slope / $\sigma_{stat}$'), (pval, 'pearson p-val')]):
    for i, (y, ylabel) in enumerate(
        [
            (std_over_mean, "stddev / mean"),
            (a_over_mean, "slope / mean"),
            (minmax_over_mean, "max-min / mean"),
            (std_over_stat_err, "stddev / $\sigma_{stat}$"),
            (a_over_stat_err, "slope / $\sigma_{stat}$"),
            (minmax_over_stat_err, "max-min / $\sigma_{stat}$"),
        ]
    ):
        ax = axes.flatten()[i]
        #         if 'p-val' in ylabel:
        #             ax.set_yscale('log')
        #             y = np.array(y) + 1e-4
        #             ax.axhline(0.054, ls='--', lw=3, color='b')
        #             ax.axhline(0.046, ls='--', lw=3, color='r')
        if "sigma" in ylabel:
            ax.set_yscale("log")
        #         if 'corr' in ylabel:
        #             ax.set_ylim(0,1.05)
        ax.hlines(y, ptbinning[:-1], ptbinning[1:], lw=2, color="k")
        ax.plot(x, y, "o", color="k")
        ax.set_xlabel("$p_{\mathrm{T}}^{\mathrm{jet}}$ (GeV/$c$)")
        ax.set_ylabel(ylabel)
        ax.grid(which="both")

    plt.tight_layout()
    return axes


y_proba = df_mc["proba"]
y_true = np.array(
    [1 if "bJet" in e else 0 for e in df_mc.index.get_level_values(1).to_numpy()]
)
df_mc["label"] = y_true

for mode, cut_ref, delta_cut, n_inter_points in cut_variation_args_lst:
    ncols = 3 if (n_ptbins) > 4 else 2
    nrows = int(np.ceil(n_ptbins / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(2 + ncols * 6, 1 + nrows * 4)
    )
    res = {}
    figname_suffix = f"_{mode}-{cut_ref}-{delta_cut}-{n_inter_points}"
    print(figname_suffix)
    gc.collect()
    for pt, ax in zip(zip(pt_binning[:-1], pt_binning[1:]), axes.flatten()):
        q = f"{pt[0]} < Jet_Pt < {pt[1]}"
        print(f"\t\t\t------ {q} ---------")
        df_mc_sel = df_mc.query(q)
        cut_variation_test(
            y_proba_data=df_data.query(q)["proba"],
            y_true_mc=df_mc_sel["label"],
            y_proba_mc=df_mc_sel["proba"],
            #         weights_mc=None,
            weights_mc=df_mc_sel["weight_pythia"].to_numpy(),
            mode=mode,
            x_vars=None,
            cut_ref=cut_ref,
            delta_cut=delta_cut,
            n_inter_points=n_inter_points,
            title_suffix="$\\bf{" + f"{pt[0]}-{pt[1]}" + "}$ GeV/$c$",
            verbose=True,
            add_fits=True,
            results=res,
            res_prefix=f'pT{q.replace(" < Jet_Pt < ", "-")}',
            axes=ax,
            n_thresholds=200,
        )
        gc.collect()

    figname_suffix = f"_{mode}-{cut_ref}-{delta_cut}"
    save_plot(os.path.join(output_dir, "cut_var_test" + figname_suffix))
    #     exp.log_figure('cut_var_test' + figname_suffix)
    axes_report = stability_report(res, pt_binning)
    #     exp.log_figure('cut_var_report' + figname_suffix)
    save_plot(os.path.join(output_dir, "cut_var_report" + figname_suffix))


print("data_mc_scores_comparison.py: ok")
