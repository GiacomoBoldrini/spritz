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
    parser.add_argument("-j", "--nworkers", type=int, default=4, help="Number of workers")
    parser.add_argument("-f", "--file", type=str, required=True, help="Pickled file ")
    parser.add_argument("-s", "--sample", type=str, required=False, help="Sample to plot (default: all)", default="all")
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

def divide_bin_width(h, var):
    if isinstance(var["axis"], list):
        b_w = var["axis"][0].widths 
        fact = 1 # repeat b_w for the number of bins we have in other directions
        for ax_ in var["axis"][1:]:
            fact *= ax_.size 
    else:
        b_w = var["axis"].widths 
        fact = 1
    
    b_w = np.tile(b_w, fact)
            
    # scale values 
    h.view().value = h.view().value / b_w
    h.view().variance = h.view().variance / b_w**2
    return h
# -----------------------------
# Plot histograms
# -----------------------------
def plot_sm_hist_worker(args):
    h_sm, other_histos, region, operator, var, variables, output_folder = args
    
    h_sm = h_sm["histo"]
    other_histos = [i["histo"] for i in other_histos]
    
    if "divide_bin_width" in variables[var].keys() and variables[var]["divide_bin_width"]:
        h_sm = divide_bin_width(h_sm, variables[var])
        other_histos = [ divide_bin_width(i, variables[var]) for i in other_histos]
        
    
    print(region, operator, var, output_folder)

    # --- Set up figure with ratio plot ---
    figsize = variables[var]["figsize"] if "figsize" in variables[var] and variables[var]["figsize"] else (8,8)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    # --- SM histogram ---
    bin_edges = h_sm.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    values = h_sm.values()
    errors = np.sqrt(h_sm.variances()) if h_sm.variances() is not None else np.zeros_like(values)

    ax.step(
        bin_edges,
        np.r_[values, values[-1]],
        where='post',
        color='black',
        label="SM"
    )

    ax.fill_between(
        bin_edges,
        np.r_[values - errors, values[-1] - errors[-1]],
        np.r_[values + errors, values[-1] + errors[-1]],
        step='post',
        alpha=0.3,
        color='black'
    )

    #ax.step(bin_edges[:-1], values, where='post', color='black', label="SM")
    #ax.fill_between(bin_edges[:-1], values-errors, values+errors, step='post', alpha=0.3, color='black')

    # --- Other histograms and ratio ---
    for idx, h__ in enumerate(other_histos):
        values_op = h__.values()
        errors_op = np.sqrt(h__.variances()) if h__.variances() is not None else np.zeros_like(values_op)
        label = f"{operator}=1" if idx ==0 else f"{operator}=-1"
        #ax.step(bin_edges[:-1], values_op, where='post', label=label)
        #ax.fill_between(bin_edges[:-1], values_op-errors_op, values_op+errors_op, step='post', alpha=0.3)
        
        ax.step(
            bin_edges,
            np.r_[values_op, values_op[-1]],
            where='post',
            label=label
        )

        ax.fill_between(
            bin_edges,
            np.r_[values_op - errors_op, values_op[-1] - errors_op[-1]],
            np.r_[values_op + errors_op, values_op[-1] + errors_op[-1]],
            step='post',
            alpha=0.3
        )

        # --- Compute ratio and band ---
        ratio = np.zeros_like(values_op)
        ratio_err = np.zeros_like(values_op)
        mask = values > 0
        ratio[mask] = values_op[mask] / values[mask]
        ratio_err[mask] = ratio[mask] * np.sqrt((errors_op[mask]/values_op[mask])**2 + (errors[mask]/values[mask])**2)

        # Draw ratio as step with uncertainty band
        #ax_ratio.step(bin_edges[:-1], ratio, where='post', label=label)
        #ax_ratio.fill_between(bin_edges[:-1],
        #                      ratio - ratio_err,
        #                      ratio + ratio_err,
        #                      step='post', alpha=0.3)
        
        ax_ratio.step(
            bin_edges,
            np.r_[ratio, ratio[-1]],
            where='post',
            label=label
        )

        ax_ratio.fill_between(
            bin_edges,
            np.r_[ratio - ratio_err, ratio[-1] - ratio_err[-1]],
            np.r_[ratio + ratio_err, ratio[-1] + ratio_err[-1]],
            step='post',
            alpha=0.3
        )

    # --- Ratio panel settings ---
    ax_ratio.axhline(1.0, color='gray', linestyle='--')
    ax_ratio.set_ylabel("Ratio to SM")
    ax_ratio.set_xlabel(variables[var]["xaxis"])
    ax_ratio.set_ylim(0.8, 1.2)  # adjust as needed

    # --- Axis options for main plot ---
    if "logy" in variables[var] and variables[var]["logy"]: ax.set_yscale("log")
    if "logx" in variables[var] and variables[var]["logx"]: ax.set_xscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel("")
    ax.set_ylabel("Events")
    if "divide_bin_width" in variables[var].keys() and variables[var]["divide_bin_width"]:
        ax.set_ylabel("Events/<Bin Width>")
    ax.tick_params(labelbottom=False)  # alternative, more robust
    ax.legend()
    hep.cms.label(ax=ax, year=2018, com=13, data=False)

    # --- Save figure ---
    os.makedirs(output_folder, exist_ok=True)
    outname_png = os.path.join(output_folder, f"{region}_{var}_{operator}.png")
    outname_pdf = os.path.join(output_folder, f"{region}_{var}_{operator}.pdf")
    plt.savefig(outname_png, bbox_inches="tight")
    plt.savefig(outname_pdf, bbox_inches="tight")
    plt.close(fig)
    
def plot_single_components_worker(args):

    h_comp, region, operator, var, variables, output_folder = args
    h_comp = h_comp["histo"]
    
    if "divide_bin_width" in variables[var].keys() and variables[var]["divide_bin_width"]:
        h_comp = divide_bin_width(h_comp, variables[var])

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

    ax_main.step(bin_edges, np.r_[values_op, values_op[-1]], where='post', color='red', label=f"{operator}")
    ax_main.fill_between(
        bin_edges, np.r_[values_op - errors_op, values_op[-1] - errors_op[-1]], np.r_[values_op + errors_op, values_op[-1] + errors_op[-1]], step='post', alpha=0.3, color='red'
    )
    # ax_main.fill_between(
    #     bin_edges[:-1], values_op - errors_op, values_op + errors_op, step='post', alpha=0.3, color='red'
    # )
    ax_main.set_ylabel("Events")
    if "divide_bin_width" in variables[var].keys() and variables[var]["divide_bin_width"]:
        ax_main.set_ylabel("Events/<Bin Width>")
        
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

    # ax_ratio.step(bin_edges[:-1], ratio, where='post', color='black')
    ax_ratio.step(bin_edges, np.r_[ratio, ratio[-1]] , where='post', color='black')

    # ax_ratio.fill_between(
    #     bin_edges[:-1],
    #     ratio - ratio_err,
    #     ratio + ratio_err,
    #     step='post',
    #     alpha=0.3,
    #     color='gray'
    # )
    ax_ratio.fill_between(
        bin_edges,
        np.r_[ratio - ratio_err, ratio[-1] - ratio_err[-1]],
        np.r_[ratio + ratio_err, ratio[-1] + ratio_err[-1]],
        step='post',
        alpha=0.3,
        color='gray'
    )
    
    ax_ratio.axhline(1.3, color="black", linestyle="--", linewidth=1)
    ax_ratio.axhline(0.7, color="black", linestyle="--", linewidth=1)
    
    ax_ratio.set_ylim(0.5, 1.5)
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
    
    shape = args.sample
    
    f = read_file(args.file)
    
    plot_tasks = []
    
    gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1400, 15000]
    mll_medium_bins = [50,58,64,72,78,84,90,96,102,108,116,124,132,140,
                148,156,164,172,180,190,200,210,220,230,240,255,270,285,300,325,350,375,
                400,450,500]
    
    costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    costheta_bins_optimized = [-1, -0.5, 0.0, 0.5, 1]

    rapll_abs_bins = [0.0, 0.5, 1.0, 1.5, 2.5]
    rapll_abs_bins_opt = [0.0, 0.5, 1.0, 2.5]

    #gen_mll_optimized = [50, 90, 94, 133, 139, 151, 199, 235, 255, 295, 510, 590, 654, 672, 708, 780, 854, 962, 3000] # 30% on cqe2
    #gen_mll_optimized = [50, 90, 94, 133, 139, 151, 199, 235, 255, 295, 510, 590, 654, 672, 708, 780, 854, 962, 3000]
    gen_mll_optimized = [50, 64, 76, 82, 86, 90, 98, 103, 121, 127, 130, 133, 148, 151, 154, 157, 163, 166, 172, 178, 184, 205, 210, 220, 235, 240, 260, 265, 325, 345, 500, 530, 560, 580, 590, 618, 636, 654, 672, 690, 708, 744, 780, 800, 827, 854, 908, 962, 1040, 3000]
    gen_mll_optimized = [50, 64, 76, 82, 86, 90, 98, 103, 121, 127, 130, 133, 148, 151, 154, 157, 163, 166, 172, 178, 184, 205, 210, 220, 235, 240, 260, 265, 325, 345, 500, 530, 570, 618, 654, 708, 3000] # min 10 ev SM 
    
    minimal_bin_width = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 618, 636, 654, 672, 690, 708, 726, 744, 762, 780, 798, 800, 827, 854, 881, 908, 935, 962, 989, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1500, 1565, 1630, 1695, 1760, 1825, 1890, 1955, 2020, 2085, 2150, 2215, 2280, 2345, 2410, 2475, 2540, 2605, 2670, 2735, 2800, 2865, 2930, 2995, 3000]
    
    variables = {
        "mll": {"axis": hist.axis.Variable(minimal_bin_width, name="mll"), "xaxis": r"$m_{\ell\ell}$ [GeV]", "figsize": (8,8), "logy": True, "logx": True, "divide_bin_width": True},
        #"costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]", "figsize": (8,8), "logy": True},
        #"triple_diff_mll400": {"axis": [hist.axis.Variable(gen_mll_bins_2, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        #"yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]", "figsize": (8,8), "logy": True},
        #"triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        #"triple_diff_medium": {"axis": [hist.axis.Variable(mll_medium_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        #
        "triple_diff_optimized": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(rapll_abs_bins_opt, name="rapll_abs")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True, "divide_bin_width": True},
        
        "triple_diff_rapll_0_0p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_0p5_1p0_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p0_1p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p5_2p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        
        "triple_diff_rapll_0_0p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_0p5_1p0_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p0_1p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, 0.0], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p5_2p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        
        "triple_diff_rapll_0_0p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_0p5_1p0_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p0_1p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p5_2p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        
        "triple_diff_rapll_0_0p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_0p5_1p0_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p0_1p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
        "triple_diff_rapll_1p5_2p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin", "logy": True, "divide_bin_width": True},
    }
     
    for region in f.keys():
        operators = list([i.split("_m1")[0] for i in f[region].keys() if i.endswith("_m1")])
        
        # plot +1 and -1 with sm
        for operator in operators:
            for var in f[region][operator].keys():
                h_sm = f[region]["sm"][var][shape]
                h_p1 = f[region][operator][var][shape]
                h_m1 = f[region][operator + "_m1"][var][shape]

                plot_tasks.append((h_sm, [h_p1, h_m1], region, operator, var, variables, args.output))
                
    with mp.Pool(processes=args.nworkers) as pool:
        pool.map(plot_sm_hist_worker, plot_tasks)
    
    # Now just plot everything 
    plot_tasks = []
    for region in f.keys():
        for operator in f[region].keys():
            if operator.endswith("_lin") or operator == "sm" or operator.endswith("_quad") or operator.endswith("_mix"):
                for var in f[region][operator].keys():
                    plot_tasks.append((f[region][operator][var][shape], region, operator, var, variables, args.output))
                
    with mp.Pool(processes=args.nworkers) as pool:
        pool.map(plot_single_components_worker, plot_tasks)
        

    
    print(f"Plots saved in {args.output}/")

if __name__ == "__main__":
    main()
