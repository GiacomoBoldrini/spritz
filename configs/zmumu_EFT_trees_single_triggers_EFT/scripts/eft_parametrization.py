import os
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

import mplhep as hep
plt.style.use(hep.style.CMS)
from config import datasets, regions, lumi, samples

# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    parser.add_argument("-o", "--output", default="plots", help="Output folder")
    parser.add_argument("-j", "--nworkers", type=int, default=4, help="Number of workers")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with input pickle files")
    parser.add_argument("--lhe-json", type=str, required=True, help="JSON with LHE reweighting weights")
    parser.add_argument("--max-files", dest="max_files", type=int, required=False, help="maximum files to process, by default all", default=-1)
    return parser.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file(path):
    with open(path, "rb") as f:
        compressed = f.read()
        decompressed = zlib.decompress(compressed)
        data = pickle.loads(decompressed)
        return data

def read_inputs(inputs: list):
    all_chunks = []
    for input_file in inputs:
        job_result = read_file(input_file)
        if isinstance(job_result, list):
            for chunk in job_result:
                if chunk.get("result"):
                    all_chunks.append(chunk["result"]["real_results"])
        else:
            all_chunks.append(job_result)
    return all_chunks

def compute_lin_quad_weights(event_weights, reweight_map, operator="cqlm1"):
    sm_idx = reweight_map["sm"]["idx"]
    op_idx = reweight_map[operator]["idx"]
    op_m1_idx = reweight_map[f"{operator}_m1"]["idx"]

    w_SM = event_weights[:, sm_idx]
    w_op = event_weights[:, op_idx]
    w_op_m1 = event_weights[:, op_m1_idx]

    w_lin = 0.5 * (w_op - w_op_m1)
    w_quad = 0.5 * (w_op + w_op_m1 - 2 * w_SM)
    return w_lin, w_quad

# -----------------------------
# Process a single file
# -----------------------------
def process_file(file_path, regions, samples_to_process, variables, reweight_map):
    # Initialize histograms
    local_histos = {}
    print("HELLO")
    for region in regions:
        local_histos[region] = {}
        for sample in samples_to_process:
            local_histos[region][sample] = {}
            for operator in reweight_map:
                if operator == "sm":
                    continue
                local_histos[region][sample][operator] = {"lin": {}, "quad": {}}
                for var in variables:
                    local_histos[region][sample][operator]["lin"][var] = hist.Hist(
                        hist.axis.Variable(np.linspace(*variables[var][sample]["binning"]),
                                           name=var, label=variables[var][sample]["xaxis"]),
                        storage=hist.storage.Weight()
                    )
                    local_histos[region][sample][operator]["quad"][var] = hist.Hist(
                        hist.axis.Variable(np.linspace(*variables[var][sample]["binning"]),
                                           name=var, label=variables[var][sample]["xaxis"]),
                        storage=hist.storage.Weight()
                    )

    # Read file chunks
    print(file_path)
    job_results = read_inputs([file_path])

    for chunk in job_results:
        for dataset in chunk:
            if dataset not in samples_to_process:
                continue
            for region in regions:
                if region not in chunk[dataset]['events']:
                    continue
                values_per_var = {var: chunk[dataset]['events'][region][var] for var in variables}
                n_events = len(values_per_var["mll"])
                weights = np.ones(n_events)

                
                
                for operator in reweight_map:
                    if operator == "sm":
                        continue

                    # Determine the "_m1" variant
                    if operator.endswith("_m1"):
                        continue  # skip the "_m1" keys, we handle them together with the base key

                    # Base and minus-one operator names
                    op_name    = operator           # e.g., "cqlm1"
                    op_m1_name = f"{operator}_m1"  # e.g., "cqlm1_m1"

                    # Access arrays from events
                    w_SM    = chunk[dataset]['events'][region]["sm"]
                    w_op    = chunk[dataset]['events'][region][op_name]
                    w_op_m1 = chunk[dataset]['events'][region][op_m1_name]

                    lin  = 0.5 * (w_op - w_op_m1)
                    quad = 0.5 * (w_op + w_op_m1 - 2*w_SM)

                    for var in variables:
                        local_histos[region][dataset][operator]["lin"][var].fill(values_per_var[var], weight=weights*lin)
                        local_histos[region][dataset][operator]["quad"][var].fill(values_per_var[var], weight=weights*quad)

    return local_histos

# -----------------------------
# Merge histograms
# -----------------------------
def merge_histos(h1, h2, regions, samples_to_process, reweight_map, variables):
    for region in regions:
        for sample in samples_to_process:
            for operator in reweight_map:
                if operator == "sm":
                    continue
                for comp in ["lin", "quad"]:
                    for var in variables:
                        h1[region][sample][operator][comp][var] += h2[region][sample][operator][comp][var]
    return h1

# -----------------------------
# Plot histograms
# -----------------------------
def plot_hist_worker(args):
    h, region, sample, operator, comp, var, variables, output_folder = args
    
    print(region, sample, operator, comp, var)

    sample_dir = os.path.join(output_folder, sample)
    os.makedirs(sample_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,6))
    bin_edges = h.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    values = h.values()
    errors = np.sqrt(h.variances()) if h.variances() is not None else np.zeros_like(values)

    ax.step(bin_edges[:-1], values, where='post', color='black', label=f"{operator} {comp}")
    ax.fill_between(bin_edges[:-1], values-errors, values+errors, step='post', alpha=0.3)
    
    # ax.errorbar(bin_centers, values, yerr=errors, fmt="o", color="black", label=f"{operator} {comp}")
    # h.plot1d(ax=ax, histtype='step', color='black', label=f"{operator} {comp}")
    ax.set_xlabel(variables[var][sample]["xaxis"])
    ax.set_ylabel("Events")
    #ax.set_yscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.legend()
    hep.cms.label(ax=ax, year=2018, com=13, data=False)

    # outname_pdf = os.path.join(sample_dir, f"{region}_{var}_{operator}_{comp}.pdf")
    outname_png = os.path.join(sample_dir, f"{region}_{var}_{operator}_{comp}.png")
    # plt.savefig(outname_pdf, bbox_inches="tight")
    plt.savefig(outname_png, bbox_inches="tight")
    plt.close(fig)
    
def plot_histos(global_histos, output_folder, regions, samples_to_process, variables, reweight_map):
    mkdir(output_folder)
    for region in regions:
        for sample in samples_to_process:
            sample_dir = os.path.join(output_folder, sample)
            mkdir(sample_dir)
            for operator in reweight_map:
                if operator == "sm":
                    continue
                for comp in ["lin", "quad"]:
                    for var in variables:
                        h = global_histos[region][sample][operator][comp][var]

                        fig, ax = plt.subplots(figsize=(8,6))
                        bin_edges = h.axes[0].edges
                        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                        values = h.values()
                        errors = np.sqrt(h.variances()) if h.variances() is not None else np.zeros_like(values)

                        ax.errorbar(bin_centers, values, yerr=errors, fmt="o", color="black", label=f"{operator} {comp}")
                        ax.set_xlabel(variables[var]["xaxis"])
                        ax.set_ylabel("Events")
                        #ax.set_yscale("log")
                        ax.set_xlim(bin_edges[0], bin_edges[-1])
                        ax.legend()
                        hep.cms.label(ax=ax, year=2018, com=13, data=False)

                        outname_pdf = os.path.join(sample_dir, f"{region}_{var}_{operator}_{comp}.pdf")
                        outname_png = os.path.join(sample_dir, f"{region}_{var}_{operator}_{comp}.png")
                        plt.savefig(outname_pdf, bbox_inches="tight")
                        plt.savefig(outname_png, bbox_inches="tight")
                        plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    mkdir(args.output)

    

    # DY samples and regions
    # regions__ = list(regions.keys())
    regions__ = ["inc_mm", "inc_ee"]
    
    samples_to_process = [
        "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos",
        "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos"
    ]
    
    m_ll_binning = {
        "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos": (50, 100, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos": (100, 200, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos": (200, 400, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos": (400, 600, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos": (600, 800, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos": (800, 1000, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos": (1000, 1500, 50),
        "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos": (1500, 3000, 50)
    }

    variables = {
        "mll": {i: {"binning": m_ll_binning[i], "xaxis": r"$m_{\ell\ell}$ [GeV]"} for i in samples_to_process},
        "costhetastar_bins": {i: {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]"} for i in samples_to_process},
        "yZ_bins": {i: {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]"} for i in samples_to_process}
    }

    # Load LHE reweight JSON
    with open(args.lhe_json) as f:
        reweight_map = json.load(f)
        
    reweight_map = [i.split("_m1")[0] for i in reweight_map.keys() if "_m1" in i]

    # List input files
    input_files = glob(args.input_dir + "/*/*.pkl")[:args.max_files]
    
    

    # Multiprocessing
    with mp.Pool(processes=args.nworkers) as pool:
        func = partial(process_file, regions=regions__, samples_to_process=samples_to_process,
                       variables=variables, reweight_map=reweight_map)
        print(func, input_files)
        partial_histos = pool.map(func, input_files)

    # Merge histograms
    global_histos = partial_histos[0]
    for h in partial_histos[1:]:
        merge_histos(global_histos, h, regions__, samples_to_process, reweight_map, variables)

    # Save merged histograms
    output_file = os.path.join(args.output, "histos_merged.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(global_histos, f)
    print(f"Histograms saved to {output_file}")
    
    
    plot_tasks = []

    for region in regions:
        for sample in samples_to_process:
            for base in reweight_map:  # from JSON mapping
                for comp in ["lin", "quad"]:
                    for var in variables:
                        h = global_histos[region][sample][base][comp][var]
                        plot_tasks.append((h, region, sample, base, comp, var, variables, args.output))
    
    with mp.Pool(processes=args.nworkers) as pool:
        pool.map(plot_hist_worker, plot_tasks)
    
    # # Plot all histograms
    # plot_histos(global_histos, output_folder=args.output,
    #             regions=regions__,
    #             samples_to_process=samples_to_process,
    #             variables=variables,
    #             reweight_map=reweight_map)
    print(f"Plots saved in {args.output}/")

if __name__ == "__main__":
    main()
