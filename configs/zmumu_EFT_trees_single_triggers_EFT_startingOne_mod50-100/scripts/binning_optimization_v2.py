import os, sys
import zlib
import pickle
import copy
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import hist
import argparse
from glob import glob
import matplotlib
matplotlib.use("Agg")  # fast, no GUI
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
import mplhep as hep
plt.style.use(hep.style.CMS)
from config import datasets, regions, lumi, samples
from spritz.framework.framework import read_chunks
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy.optimize import curve_fit
import random
from spritz.framework.framework import get_fw_path
from config import datasets, regions, lumi, samples, year

def renorm(xs, sumw, lumi):
    scale = xs * 1000 * lumi / sumw
    # print(scale)
    return scale

path_fw = get_fw_path()

with open(f"{path_fw}/data/{year}/samples/samples.json") as file:
        cross_sections = json.load(file)["samples"]

# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    #parser.add_argument("-o", "--output", default="plots", help="Output folder")
    parser.add_argument("-j", "--nworkers", type=int, default=4, help="Number of workers")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with input pickle files")
    parser.add_argument("--max-files", dest="max_files", type=int, required=False, help="maximum files to process, by default all", default=-1)
    return parser.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def initial_mll_binning(min_bin_width):
    """
    Build initial fine mll bin edges according to minimum width.
    """
    binning = []
    for (start, stop), width in min_bin_width.items():
        edges = np.arange(start, stop, width)
        if len(binning) > 0 and edges[0] == binning[-1]:
            edges = edges[1:]  # avoid duplicates
        binning += edges.tolist()
    # Ensure last edge reaches stop
    last_stop = list(min_bin_width.keys())[-1][1]
    if binning[-1] < last_stop:
        binning.append(last_stop)
    return np.array(binning)

def optimize_mll_binning(var, costheta_bins, rapll_abs_bins, min_bin_width, 
                         min_weighted_events=10, region="inc_mm"):
    """
    Optimized mll binning:
    - Each bin respects min_bin_width
    - Each bin has at least min_weighted_events in the SM template
    - Each operator component has relative stat. uncertainty <= max_rel_unc
    """

    # --- Step 0: initial fine binning ---
    mll_bins = initial_mll_binning(min_bin_width)
    n_mll_bins = len(mll_bins) - 1
    n_costheta = len(costheta_bins) - 1
    n_rapll = len(rapll_abs_bins) - 1

    operators = ["cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]
    labels = []
    for op in operators:
        labels.append(f"{op}_lin")
        labels.append(f"{op}_quad")
    n_ops = 2 * len(operators)

    # --- Step 1: Precompute sumw and sumw2 for the finest binning ---
    sumw = np.zeros((n_mll_bins, n_costheta, n_rapll, n_ops))
    sumw2 = np.zeros_like(sumw)
    sumw_sm = np.zeros((n_mll_bins, n_costheta, n_rapll))  # for SM min-event check

    for sample, sample_data in var.items():
        data = sample_data[region]
        mll_vals = np.array(data["mll"])
        costheta_vals = np.array(data["costhetastar_bins"])
        rapll_abs_vals = np.array(data["rapll_abs"])
        weights = np.array(data["weight"])
        sumw_sample = data["sumw"]

        xs = float(cross_sections[sample]["xsec"])
        scale = renorm(xs, sumw_sample, lumi)

        # Digitize once
        costheta_idx = np.digitize(costheta_vals, costheta_bins) - 1
        rapll_idx = np.digitize(rapll_abs_vals, rapll_abs_bins) - 1

        for i in range(n_costheta):
            for j in range(n_rapll):
                mask = (costheta_idx == i) & (rapll_idx == j)
                if not np.any(mask):
                    continue

                # --- SM template for min_weighted_events ---
                w_sm = weights[mask] * np.array(data["sm"])[mask]
                counts_sm, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=w_sm)
                sumw_sm[:, i, j] += counts_sm * scale

                # --- All EFT operators for rel. uncertainty ---
                for op_idx, op in enumerate(operators):
                    # LIN component
                    w_lin = weights[mask] * np.array(data[f"{op}_lin"])[mask]
                    counts_lin, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=w_lin)
                    counts2_lin, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=w_lin**2)
                    sumw[:, i, j, 2*op_idx] += counts_lin * scale
                    sumw2[:, i, j, 2*op_idx] += counts2_lin * scale**2

                    # QUAD component
                    w_quad = weights[mask] * np.array(data[f"{op}_quad"])[mask]
                    counts_quad, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=w_quad)
                    counts2_quad, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=w_quad**2)
                    sumw[:, i, j, 2*op_idx+1] += counts_quad * scale
                    sumw2[:, i, j, 2*op_idx+1] += counts2_quad * scale**2
    
    print(sumw_sm)
    
    # --- Helper function to merge a single bad bin ---
    def merge_single_bin(bins, sumw_arr, sumw2_arr=None, bad_bins=None):
        """
        Merge ONLY the first bad bin with a neighbor.
        Always deletes exactly one edge in bins.
        Handles first, middle, and last bins safely.
        """
        if bad_bins is None or len(bad_bins) == 0:
            return bins, sumw_arr if sumw2_arr is None else (bins, sumw_arr, sumw2_arr)

        b = bad_bins[0]  # first bad bin
        n_bins = sumw_arr.shape[0]
        n_edges = len(bins)

        # Determine which neighbor to merge with
        if b == 0:
            neighbor = 1
            edge_to_delete = 1
        elif b == n_bins - 1:
            neighbor = b - 1
            edge_to_delete = b
        else:
            neighbor = b + 1
            edge_to_delete = neighbor

        # Safety: never delete the last edge
        if edge_to_delete >= n_edges - 1:
            edge_to_delete = n_edges - 2
            neighbor = edge_to_delete

        print(f"Merging bin {b} into neighbor {neighbor}, deleting edge {edge_to_delete}")

        # Merge counts
        sw = sumw_arr.copy()
        sw[min(b, neighbor)] += sw[max(b, neighbor)]
        sw = np.delete(sw, max(b, neighbor), axis=0)

        if sumw2_arr is None:
            bins = np.delete(bins, edge_to_delete)
            return bins, sw
        else:
            sw2 = sumw2_arr.copy()
            sw2[min(b, neighbor)] += sw2[max(b, neighbor)]
            sw2 = np.delete(sw2, max(b, neighbor), axis=0)
            bins = np.delete(bins, edge_to_delete)
            return bins, sw, sw2
            
    
    
    #sys.exit(0)
    # --- Step 2: Enforce min_weighted_events on SM ---
    merging_needed = True    
    print("mll_bins initial:", mll_bins, " with", len(mll_bins), "bins")
    while merging_needed:
        print(f"---> Enforcing min weighted events per bin: {min_weighted_events}")
        min_counts = sumw_sm.min(axis=(1,2))
        bad_bins = np.where(min_counts < min_weighted_events)[0]

        if len(bad_bins) == 0 or len(mll_bins) <= 2:
            merging_needed = False
            print("All bins pass min weighted events check.")
            break

        print(f"Bad bins (min weighted events < {min_weighted_events}): {bad_bins} len {len(bad_bins)}")
        print("sumw_sm shape: ", sumw_sm.shape)

        mll_bins_before_merging = mll_bins.copy()
        # merge first bad bin in SM
        mll_bins, sumw_sm = merge_single_bin(mll_bins, sumw_sm, bad_bins=bad_bins)

        # merge first bad bin consistently in EFT arrays
        _, sumw, sumw2 = merge_single_bin(mll_bins_before_merging, sumw, sumw2, bad_bins=bad_bins)

        print("Current mll bins after min_weighted_events:", mll_bins, " with", len(mll_bins), "bins")
    
    """
    # --- Step 3: Enforce max relative uncertainty for all operators ---
    max_rel_unc = 0.5
    merging_needed = True
    while merging_needed:
        rel_unc = np.full_like(sumw, np.inf, dtype=float)
        nonzero = sumw != 0
        rel_unc[nonzero] = np.sqrt(sumw2[nonzero]) / sumw[nonzero]

        max_rel_unc_per_mll = rel_unc.max(axis=(1,2,3))
        bad_bins = np.where(max_rel_unc_per_mll > max_rel_unc)[0]

        print("----> Checking relative uncertainties, max rel unc per bin:", max_rel_unc_per_mll)
        coords = np.argwhere(rel_unc>max_rel_unc)
        for c in coords:
            print(f"rel unc > {max_rel_unc} at: {c}, component: {labels[c[3]]}, sumw: {sumw[tuple(c)]}, sumw2: {sumw2[tuple(c)]}, rel_unc: {abs(np.sqrt(sumw2[tuple(c)]) / sumw[tuple(c)]) if sumw[tuple(c)] !=0 else 'inf'}")

        if len(bad_bins) == 0 or len(mll_bins) <= 2:
            merging_needed = False
            print("No more bins to merge for rel_unc.")
            break

        mll_bins, sumw, sumw2 = merge_single_bin(mll_bins, sumw, sumw2, bad_bins=bad_bins)

        print("Current mll bins after rel_unc:", mll_bins, " with", len(mll_bins), "bins")
    """
    # --- Step 3: Enforce max relative uncertainty for all operators ---
    max_rel_unc = 0.5
    hard_rel_unc = 1.0     # HARD veto threshold
    min_fraction = 0.7     # fraction of cells that must pass
    merging_needed = True

    while merging_needed:
        # compute relative uncertainty
        rel_unc = np.full_like(sumw, np.inf, dtype=float)
        nonzero = sumw != 0
        rel_unc[nonzero] = np.sqrt(sumw2[nonzero]) / sumw[nonzero]

        # print max rel unc per bin
        max_rel_unc_per_mll = rel_unc.max(axis=(1,2,3))
        print("----> Checking relative uncertainties, max rel unc per bin:", max_rel_unc_per_mll)

        # debug prints for bad cells
        coords = np.argwhere(rel_unc > max_rel_unc)
        for c in coords:
            print(
                f"rel unc > {max_rel_unc} at: {c}, "
                f"component: {labels[c[3]]}, "
                f"sumw: {sumw[tuple(c)]}, sumw2: {sumw2[tuple(c)]}, "
                f"rel_unc: {abs(np.sqrt(sumw2[tuple(c)]) / sumw[tuple(c)]) if sumw[tuple(c)] != 0 else 'inf'}"
            )

        # -------- FRACTION CONDITION --------
        pass_mask = rel_unc < max_rel_unc
        frac_pass = pass_mask.sum(axis=(1,2,3)) / np.prod(pass_mask.shape[1:])
        bad_bins_frac = np.where(frac_pass < min_fraction)[0]

        # -------- HARD EFT VETO CONDITION --------
        # max over costheta, rap, operators per mll bin
        max_rel_unc_bin = rel_unc.max(axis=(1,2,3))
        bad_bins_hard = np.where(max_rel_unc_bin >= hard_rel_unc)[0]

        # -------- COMBINE CONDITIONS --------
        bad_bins = np.unique(np.concatenate([bad_bins_frac, bad_bins_hard]))

        print(f"Bins failing fraction condition (<{min_fraction*100:.0f}% pass): {bad_bins_frac}")
        print(f"Bins failing HARD veto (rel_unc >= {hard_rel_unc}): {bad_bins_hard}")
        print(f"Final bad bins: {bad_bins}")

        if len(bad_bins) == 0 or len(mll_bins) <= 2:
            merging_needed = False
            print("No more bins to merge for rel_unc.")
            break

        # merge only first bad bin
        mll_bins, sumw, sumw2 = merge_single_bin(mll_bins, sumw, sumw2, bad_bins=bad_bins)

        print("Current mll bins after rel_unc:", mll_bins, " with", len(mll_bins), "bins")
        
    # --- Always return final bins ---
    print(f"Final optimized mll bins: {mll_bins}")
    return mll_bins
    

def process_single_file(input_file, regions, samples):
    # Nested dict: sample -> region -> events
    var = {
        sample: {
            region: {
                "mll": [],
                "costhetastar_bins": [],
                "rapll_abs": [],
                "weight": [],
                "sumw": 0
            } for region in regions
        } for sample in samples
    }
    
    operators = ["sm", "cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]
    # now add reweighting weights for lin and quad components
    
    # --- single-operator weights ---
    for op in operators:
        for sample in samples:
            for region in regions:
                if op == "sm":
                    var[sample][region]["sm"] = []
                else:
                    var[sample][region][f"{op}_lin"] = []
                    var[sample][region][f"{op}_quad"] = []
                    

    print(f"Reading file: {input_file}")
    job_result = read_chunks(input_file)

    for chunk in job_result:
        #print(chunk.keys(), chunk["result"].keys(), input_file)
        if "real_results" not in chunk["result"]:
            continue
        chunk = chunk["result"]["real_results"]
        for dataset, dset_data in chunk.items():
            if dataset not in samples:
                continue
            sumw = dset_data['sumw']
            if sumw == 0:
                print(f"File path: {input_file} Dataset: {dataset}, sumw: {sumw}")
            
            for region in regions:
                region_data = dset_data['events'].get(region)
                if region_data is None:
                    continue
                base_weights = region_data["weight"]
                if "sm" not in region_data:
                    continue
                w_sm = region_data["sm"]
                
                var[dataset][region]["mll"] += region_data["mll"].tolist()
                var[dataset][region]["costhetastar_bins"] += region_data["costhetastar_bins"].tolist()
                var[dataset][region]["rapll_abs"] += region_data["rapll_abs"].tolist()
                var[dataset][region]["weight"] += (base_weights).tolist()
                var[dataset][region]["sumw"] += sumw  # sumw per dataset, or optionally divide per region
                
                for op in operators:
                    if op == "sm":
                        var[dataset][region]["sm"] += w_sm.tolist()
                    else:
                        w_op = region_data[op]
                        w_op_m1 = region_data[f"{op}_m1"]
                        w_lin = 0.5 * (w_op - w_op_m1)
                        w_quad = 0.5 * (w_op + w_op_m1 - 2 * w_sm)
                        var[dataset][region][f"{op}_lin"] += w_lin.tolist()
                        var[dataset][region][f"{op}_quad"] += w_quad.tolist()
    
    return var


def merge_vars(var_list, samples, regions):
    """
    Merge a list of var dictionaries (nested sample -> region -> data) into one.
    """
    # Initialize empty merged structure
    merged = {
        sample: {
            region: {
                "mll": [],
                "costhetastar_bins": [],
                "rapll_abs": [],
                "weight": [],
                "sumw": 0
            } for region in regions
        } for sample in samples
    }
    
    operators = ["sm", "cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]
    # now add reweighting weights for lin and quad components
    
    # --- single-operator weights ---
    for op in operators:
        for sample in samples:
            for region in regions:
                if op == "sm":
                    merged[sample][region]["sm"] = []
                else:
                    merged[sample][region][f"{op}_lin"] = []
                    merged[sample][region][f"{op}_quad"] = []

    # Loop over all var dictionaries (from different files)
    for v in var_list:
        for sample in samples:
            for region in regions:
                merged[sample][region]["mll"] += v[sample][region]["mll"]
                merged[sample][region]["costhetastar_bins"] += v[sample][region]["costhetastar_bins"]
                merged[sample][region]["rapll_abs"] += v[sample][region]["rapll_abs"]
                merged[sample][region]["weight"] += v[sample][region]["weight"]
                merged[sample][region]["sumw"] += v[sample][region]["sumw"]
                
                for op in operators:
                    if op == "sm":
                        merged[sample][region]["sm"] += v[sample][region]["sm"]
                    else:
                        merged[sample][region][f"{op}_lin"] += v[sample][region][f"{op}_lin"]
                        merged[sample][region][f"{op}_quad"] += v[sample][region][f"{op}_quad"]
                            

    return merged


def read_inputs_skimmed_parallel(inputs, regions, samples, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()
    
    args = [(f, regions, samples) for f in inputs]
    
    with Pool(n_workers) as pool:
        results = pool.starmap(process_single_file, args)
    
    var = merge_vars(results, samples, regions)
    return var


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    # mkdir(args.output)


    # DY samples and regions

    regions__ = ["inc_mm"]
    
    samples_to_process = [
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
    ]
    
    all_files = glob(args.input_dir + "/job_*/chunks_job.pkl")
    if args.max_files == -1:
        file_path = all_files
    else:
        file_path = [args.input_dir + f"/job_{i}/chunks_job.pkl" for i in [random.randint(0, len(all_files)-1) for _ in range(0, len(all_files) if args.max_files == -1 else args.max_files)]]
    print(file_path)
    # var__ = read_inputs_skimmed(file_path, regions__, samples_to_process)
    var__ = read_inputs_skimmed_parallel(file_path, regions__, samples_to_process, n_workers=args.nworkers if len(file_path) > args.nworkers else len(file_path))
    
    min_bin_width = {
        (50,100): 2,
        (100,200): 3,
        (200,400): 5,
        (400,600): 10,
        (600,800): 18,
        (800,1000): 27,
        (1000,1500): 40,
        (1500,3000): 65,
    }
    # mll, costhetastar, rapll_abs, cpwb_lin
    #[0 4 3 6]
    
    costheta_bins = [-1, -0.5, 0.0, 0.5, 1]
    rapll_abs_bins = [0.0, 0.5, 1.0, 2.5]
    
    # costheta_bins = [-1, 1]
    # rapll_abs_bins = [0.0, 2.5]

    # var contains your combined data
    # var = {"mll": [...], "costhetastar_bins": [...], "rapll_abs": [...], "weight": [...]}

    mll_bins = optimize_mll_binning(var__, costheta_bins, rapll_abs_bins, min_bin_width, region=regions__[0])
    print(f"Optimized mll bins: {list(mll_bins)}")


if __name__ == "__main__":
    main()
