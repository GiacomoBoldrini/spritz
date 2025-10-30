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

# -----------------------------
# Plot histograms
# -----------------------------
def plot_sm_hist_worker(args):
    h_sm, other_histos, region, operator, var, variables, output_folder = args
    
    h_sm = h_sm["histo"]
    other_histos = [i["histo"] for i in other_histos]
    
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

    ax.step(bin_edges[:-1], values, where='post', color='black', label="SM")
    ax.fill_between(bin_edges[:-1], values-errors, values+errors, step='post', alpha=0.3, color='black')

    # --- Other histograms and ratio ---
    for idx, h__ in enumerate(other_histos):
        values_op = h__.values()
        errors_op = np.sqrt(h__.variances()) if h__.variances() is not None else np.zeros_like(values_op)
        label = f"{operator}=1" if idx ==0 else f"{operator}=-1"
        ax.step(bin_edges[:-1], values_op, where='post', label=label)
        ax.fill_between(bin_edges[:-1], values_op-errors_op, values_op+errors_op, step='post', alpha=0.3)

        # --- Compute ratio and band ---
        ratio = np.zeros_like(values_op)
        ratio_err = np.zeros_like(values_op)
        mask = values > 0
        ratio[mask] = values_op[mask] / values[mask]
        ratio_err[mask] = ratio[mask] * np.sqrt((errors_op[mask]/values_op[mask])**2 + (errors[mask]/values[mask])**2)

        # Draw ratio as step with uncertainty band
        ax_ratio.step(bin_edges[:-1], ratio, where='post', label=label)
        ax_ratio.fill_between(bin_edges[:-1],
                              ratio - ratio_err,
                              ratio + ratio_err,
                              step='post', alpha=0.3)

    # --- Ratio panel settings ---
    ax_ratio.axhline(1.0, color='gray', linestyle='--')
    ax_ratio.set_ylabel("Ratio to SM")
    ax_ratio.set_xlabel(variables[var]["xaxis"])
    ax_ratio.set_ylim(0.0, 2.0)  # adjust as needed

    # --- Axis options for main plot ---
    if "logy" in variables[var] and variables[var]["logy"]: ax.set_yscale("log")
    if "logx" in variables[var] and variables[var]["logx"]: ax.set_xscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel("")
    ax.set_ylabel("Events")
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
    ax_ratio.set_ylim(0.0, 2.0)
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
    
    gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1500, 15000]
    mll_medium_bins = [50,58,64,72,78,84,90,96, 200,210,220,230,240,255,270,285,300,325,350,375,
                400,450,500]
    gen_mll_optimized = [50, 92, 94, 96, 98, 106, 112, 160, 265, 275, 295, 315, 430, 440, 450, 460, 480, 490, 500, 510, 520, 540, 560, 580, 600, 636, 672, 708, 744, 800, 908, 3000]

    costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    etaZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]
    
    variables = {
        "mll": {"binning": (50, 800, 500), "xaxis": r"$m_{\ell\ell}$ [GeV]", "figsize": (8,8), "logy": True, "logx": True},
        "costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]", "figsize": (8,8), "logy": True},
        "yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]", "figsize": (8,8), "logy": True},
        "triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        "triple_diff_medium": {"axis": [hist.axis.Variable(mll_medium_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
        "triple_diff_optimized": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin", "figsize": (25,8), "logy": True},
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
