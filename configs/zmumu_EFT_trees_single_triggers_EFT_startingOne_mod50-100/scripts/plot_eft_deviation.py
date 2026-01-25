import os, sys
import zlib
import pickle
import copy
import json
import numpy as np
import multiprocessing as mp
from functools import partial
import hist
import argparse
from glob import glob
import matplotlib
matplotlib.use("Agg")  # fast, no GUI
import matplotlib.pyplot as plt
from copy import deepcopy

import mplhep as hep
plt.style.use(hep.style.CMS)

# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    parser.add_argument("-o", "--output", default="plots", help="Output folder")
    parser.add_argument("-op", "--operator", required=True, help="One operator to plot and coeff value for example <-op k_cpwb=0.1>")
    parser.add_argument("-f", "--file", type=str, required=True, help="Pickled file ")
    parser.add_argument("-j", "--nworkers", type=int, default=4, help="Number of workers")

    return parser.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file(path):
    with open(path, "rb") as f:
        global_histos = pickle.load(f)
    return global_histos

def get_shape_name(names):
    shape1, shape2 = names.split("x") 
    #print(shape1, shape2 )
    ops = []
    for sh__ in [shape1, shape2]:
        if sh__.startswith("w1_"): ops.append(sh__.split("w1_")[1])
        elif sh__.startswith("wm1_"): ops.append(sh__.split("wm1_")[1] + "_m1")
        elif sh__.startswith("w11_"): ops.append(sh__.split("w11_")[1])
        elif sh__ == "sm": ops.append(sh__)

    #print(ops)
    # diagonal
    if shape1 == shape2:
        if shape1 == "sm": 
            return "sm_variance"
        elif shape1.startswith("w1_"): 
            op__ = shape1.split("w1_")[1]
            return f"{op__}_plus_variance"
        elif shape1.startswith("wm1_"):
            op__ = shape1.split("wm1_")[1]
            return f"{op__}_minus_variance"
        elif shape1.startswith("w11_"): 
            op1_op2 = shape1.split("w11_")[1]
            return f"{op1_op2}_variance"
        
    # off diagonal 
    return ops, "_mixed_variance"

# -----------------------------
# Plot histograms
# -----------------------------
def plot_sm_hist_worker(args):
    h_sm, other_histos, region, operator, value, var, variables, output_folder = args
    
    h_sm = h_sm
    h_p1 = other_histos["w1_"+operator]
    h_m1 = other_histos["wm1_"+operator]
    # EFT = sm + k/2 (h_p1 - h_m1) + k**2/2 (h_p1 + h_m1 - 2*sm)
    k = float(value)
    
    eft = h_sm.copy()
    
    
    
    eft.view().value[:] = (
        (1-k**2)*h_sm.view().value[:] +
        (k/2*(1+k))*h_p1.view().value[:] +
        (k/2*(k-1))*h_m1.view().value[:]
    )
    
    print("---> EFT")
    
    # variances are trickier 
    
    # variance_EFT(k) = (1-k**2)**2 * variance_SM  
    #                   + (k/2*(1+k))**2 * variance_Plus1
    #                   + (k/2*(k-1))**2 * variance_Minus1
    #                   + 2(1-k**2)*(k/2*(1+k)) * covariance(SM, Plus1)
    #                   + 2(1-k**2)*(k/2*(k-1)) * covariance(SM, Minus1)
    #                   + 2(k/2*(1+k))*(k/2*(k-1)) * covariance(Plus1, Minus1)
    
    
    # dict_keys(['w1_cpd', 'wm1_cpd', 'sm_variance', 'sm_cpd_mixed_variance', 'sm_cpd_m1_mixed_variance', 'cpd_plus_variance', 'cpd_cpd_m1_mixed_variance', 'cpd_minus_variance'])
    # retrieve the pieces 
    var_sm = other_histos["sm_variance"]
    var_w1 = other_histos[f"{operator}_plus_variance"]
    var_wm1 = other_histos[f"{operator}_minus_variance"]
    var_sm_w1 = other_histos[f"sm_{operator}_mixed_variance"]
    var_sm_wm1 = other_histos[f"sm_{operator}_m1_mixed_variance"]
    var_w1_wm1 = other_histos[f"{operator}_{operator}_m1_mixed_variance"]

    eft.view().variance[:] = (1-k**2)**2 * var_sm.values() + \
                            (k/2*(1+k))**2 * var_w1.values() + \
                            (k/2*(k-1))**2 * var_wm1.values() + \
                            (1-k**2)*(k/2*(1+k)) * var_sm_w1.values() + \
                            (1-k**2)*(k/2*(k-1)) * var_sm_wm1.values() + \
                            (k/2*(1+k))*(k/2*(k-1)) * var_w1_wm1.values()   
                            
    print(eft.view().variance[:]) 

    # scale eft around SM  
    eft.view().value[:] = eft.view().value[:] / h_sm.view().value[:]
    eft.view().variance[:] = eft.view().variance[:] / (h_sm.view().value[:]**2)
    

    # --- Set up figure with ratio plot ---
    figsize = variables[var]["figsize"] if "figsize" in variables[var] and variables[var]["figsize"] else (8,8)
    fig = plt.figure(figsize=figsize)

    # --- SM histogram ---
    bin_edges = eft.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    values = eft.values()
    errors = np.sqrt(eft.variances()) if eft.variances() is not None else np.zeros_like(values)

    plt.step(bin_edges[:-1], values, where='post', color='black', label="EFT {}={}".format(operator, value))
    plt.fill_between(bin_edges[:-1], values-errors, values+errors, step='post', alpha=0.3, color='black')
    
    # horizontal line at 1
    plt.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=0, 
        label = "SM"
    )
    
    plt.legend()
    plt.xlabel(variables[var]["xaxis"])
    plt.ylabel("X / SM")
    if "logx" in variables[var] and variables[var]["logx"]: plt.xscale("log")
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.ylim(0.8, 1.2)
    hep.cms.label(ax=plt.gca(), year=2018, com=13, data=False)

    # --- Save figure ---
    os.makedirs(output_folder, exist_ok=True)
    outname_png = os.path.join(output_folder, f"{region}_{var}_{operator}_{value}.png")
    outname_pdf = os.path.join(output_folder, f"{region}_{var}_{operator}_{value}.pdf")
    plt.savefig(outname_png, bbox_inches="tight")
    plt.savefig(outname_pdf, bbox_inches="tight")
    plt.close(fig)
    
def plot_single_components_worker(args):

    h_comp, region, operator, var, variables, output_folder = args
    h_comp = h_comp["histo"]

    print(region, operator, var, output_folder)

    # Figure size
    figsize = variables[var]["figsize"] if "figsize" in variables[var] and variables[var]["figsize"] else (8,8)

    # Create figure with main + ratio panel
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios":[3,1]}, sharex=True
    )

    # Histogram data
    bin_edges = h_comp.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    values_op = h_comp.values()
    errors_op = np.sqrt(h_comp.variances()) if h_comp.variances() is not None else np.zeros_like(values_op)

    # --- Main histogram ---
    ax_main.step(bin_edges[:-1], values_op, where='post', color='red', label=f"{operator}")
    ax_main.fill_between(
        bin_edges[:-1], values_op - errors_op, values_op + errors_op, step='post', alpha=0.3, color='red'
    )
    ax_main.set_ylabel("Events")
    if "logy" in variables[var] and variables[var]["logy"]:
        if np.all(values_op > 0):
            ax_main.set_yscale("log")
    if "logx" in variables[var] and variables[var]["logx"]:
        ax_main.set_xscale("log")
    ax_main.set_xlim(bin_edges[0], bin_edges[-1])
    ax_main.legend()
    hep.cms.label(ax=ax_main, year=2018, com=13, data=False)

    # --- Ratio panel: histogram / histogram = 1 with uncertainty ---
    ratio = np.ones_like(values_op)
    ratio_err = np.zeros_like(values_op)

    # Avoid division by zero and handle negative bins
    nonzero_mask = values_op != 0
    ratio_err[nonzero_mask] = errors_op[nonzero_mask] / np.abs(values_op[nonzero_mask])
    ratio_err[~nonzero_mask] = 0  # zero bins: uncertainty zero

    ax_ratio.step(bin_edges[:-1], ratio, where='post', color='black')
    ax_ratio.fill_between(
        bin_edges[:-1],
        ratio - ratio_err,
        ratio + ratio_err,
        step='post',
        alpha=0.3,
        color='gray'
    )
    ax_ratio.set_ylim(0.7, 1.3)
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_xlabel(variables[var]["xaxis"])

    # --- Save figure ---
    output_folder = os.path.join(output_folder, "components")
    os.makedirs(output_folder, exist_ok=True)
    outname_png = os.path.join(output_folder, f"component_{region}_{var}_{operator}.png")
    outname_pdf = os.path.join(output_folder, f"component_{region}_{var}_{operator}.pdf")

    plt.savefig(outname_png, bbox_inches="tight")
    plt.savefig(outname_pdf, bbox_inches="tight")
    plt.close(fig)
    
# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    mkdir(args.output)
    
    shape = "all"
    
    f = read_file(args.file)
    
    plot_tasks = []
    
    gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1500, 15000]
    gen_mll_bins_2 = [400, 600, 800, 1000, 1400, 15000]
    mll_medium_bins = [50,58,64,72,78,84,90,96, 200,210,220,230,240,255,270,285,300,325,350,375,
                400,450,500]
    gen_mll_optimized = [50, 92, 94, 96, 98, 106, 112, 160, 265, 275, 295, 315, 430, 440, 450, 460, 480, 490, 500, 510, 520, 540, 560, 580, 600, 636, 672, 708, 744, 800, 908, 3000]

    costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    etaZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]
    
    variables = {
        "mll": {"binning": (50, 800, 500), "xaxis": r"$m_{\ell\ell}$ [GeV]", "figsize": (8,8), "logy": True, "logx": True},
        "costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]", "figsize": (8,8), "logy": True},
        "triple_diff_mll400": {"axis": [hist.axis.Variable(gen_mll_bins_2, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]", "figsize": (8,8), "logy": True},
        "triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        "triple_diff_medium": {"axis": [hist.axis.Variable(mll_medium_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        "triple_diff_optimized": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        "triple_diff_mll_1500_3000": {"axis": [hist.axis.Variable(np.linspace(1500, 3000, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_1000_1500": {"axis": [hist.axis.Variable(np.linspace(1000, 1500, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_800_1000": {"axis": [hist.axis.Variable(np.linspace(800, 1000, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_600_800": {"axis": [hist.axis.Variable(np.linspace(600, 800, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_400_600": {"axis": [hist.axis.Variable(np.linspace(400, 600, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_200_400": {"axis": [hist.axis.Variable(np.linspace(200, 400, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_120_200": {"axis": [hist.axis.Variable(np.linspace(120, 200, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_mll_50_120": {"axis": [hist.axis.Variable(np.linspace(120, 200, 10), name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
    }
     
    operator, value = args.operator.split("=")
    for region in f.keys():
        # plot +1 and -1 with sm
        for var in f[region][operator].keys():
            h_sm = f[region]["sm"][var][shape]["histo"]
            
            other = {
                "w1_"+operator: f[region][operator][var][shape]["histo"],
                "wm1_"+operator: f[region][operator + "_m1"][var][shape]["histo"],
            }
            #h_p1 = f[region][operator][var][shape]
            #h_m1 = f[region][operator + "_m1"][var][shape]

            categories = ["sm", f"w1_{operator}", f"wm1_{operator}"] # ["sm", "w1_cw", "wm1_cw", ..., "w11_cw_chl1"]

            labels = np.array([[f"{a}x{b}" for b in categories] for a in categories])
            
            for i in range(0, labels.shape[0]):
                for j in range(i, labels.shape[1]):
                    sn__ = get_shape_name(labels[i,j])
                    print(sn__)
                     
                    if isinstance(sn__, str):
                        other[sn__] = f[region][sn__][var][shape]["histo"]
                    else:
                        perm_name = "_".join(sn__[0]) + sn__[1]
                        if perm_name not in f[region].keys():
                            perm_name = "_".join(sn__[0][::-1]) + sn__[1]
                            
                        #print(len(f[region].keys()))
                        other[perm_name] = f[region][perm_name][var][shape]["histo"]
            
            print(other.keys())
            # need to save cross terms as well 
            
            # h_sm_wp1 = 
            # h_sm_wm1 = 
            # h_wp1_wm1 =

            plot_tasks.append((h_sm, other, region, operator, value, var, variables, args.output))
                
    with mp.Pool(processes=args.nworkers) as pool:
        pool.map(plot_sm_hist_worker, plot_tasks)
    
    
    print(f"Plots saved in {args.output}/")

if __name__ == "__main__":
    main()
