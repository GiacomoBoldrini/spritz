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

def optimize_mll_binning(var, costheta_bins, rapll_abs_bins, min_bin_width, min_weighted_events=2, region="inc_mm"):
    """
    Compute global mll binning:
    - Each bin respects min_bin_width
    - Each bin has at least min_weighted_events in every (costheta, rapll_abs_bins) bin
    - Each EFT operator component (lin/quad) has relative stat. uncertainty <= max_rel_unc
    """
    import numpy as np

    # Step 0: initial fine binning
    mll_bins = initial_mll_binning(min_bin_width)

    n_costheta = len(costheta_bins) - 1
    n_rapll = len(rapll_abs_bins) - 1

    # --- First, enforce minimum weighted events ---
    merging_needed = True
    while merging_needed:
        merging_needed = False
        counts_per_bin = np.zeros((len(mll_bins)-1, n_costheta, n_rapll))

        for i in range(n_costheta):
            for j in range(n_rapll):
                for sample in var.keys():
                    data = var[sample][region]
                    mll_vals = np.array(data["mll"])
                    costheta_vals = np.array(data["costhetastar_bins"])
                    rapll_abs_vals = np.array(data["rapll_abs"])
                    weights = np.array(data["weight"])
                    sm = np.array(data["sm"])
                    sumw_sample = data["sumw"]

                    mask = (np.digitize(costheta_vals, costheta_bins)-1 == i) & \
                           (np.digitize(rapll_abs_vals, rapll_abs_bins)-1 == j)
                    if np.any(mask):
                        counts, _ = np.histogram(mll_vals[mask], bins=mll_bins, weights=weights[mask]*sm[mask])
                        xs = float(cross_sections[sample]["xsec"])
                        scale = renorm(xs, sumw_sample, lumi)
                        counts_per_bin[:, i, j] += counts*scale

        # Identify bins with too few events
        min_counts = counts_per_bin.min(axis=(1,2))
        bad_bins = np.where(min_counts < min_weighted_events)[0]
        
        to_delete = []
        if len(bad_bins) > 0:
            merging_needed = True
            i = 0
            while i < len(bad_bins):
                b = bad_bins[i]

                # Merge the bad bin with its right neighbor if it exists
                if b < len(mll_bins) - 2:  # not last bin
                    to_delete.append(b + 1)
                else:  # last bin, merge with previous
                    to_delete.append(b)

                # Skip next bin if consecutive, but never merge more than 2 bins
                if i + 1 < len(bad_bins) and bad_bins[i + 1] == b + 1:
                    i += 2
                else:
                    i += 1

            # Remove duplicates in case multiple bins target same neighbor
            to_delete = sorted(set(to_delete))
            print(f"Bins to delete: {to_delete}")
            mll_bins = np.delete(mll_bins, to_delete)

        print(f"Current mll bins after min_weighted_events: {mll_bins}")
        """
        if len(bad_bins) > 0:
            merging_needed = True
            to_delete = []
            i = 0
            while i < len(bad_bins):
                b = bad_bins[i]
                if b < len(mll_bins)-2:
                    to_delete.append(b+1)
                elif b > 0:
                    to_delete.append(b)
                if i+1 < len(bad_bins) and bad_bins[i+1] == b+1:
                    i += 1
                i += 1
            mll_bins = np.delete(mll_bins, to_delete)

        print(f"Current mll bins after min_weighted_events: {mll_bins}")
        """
    sys.exit(0)
    # --- Second, enforce max relative uncertainty per operator ---
    max_rel_unc = 0.3  # 50%
    operators = ["cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]  # replace with full list if needed
    operators = ["cpqmi"]
    labels = []
    for op in operators:
        labels.append(f"{op}_lin")
        labels.append(f"{op}_quad")
        
    merging_needed = True
    while merging_needed:
        merging_needed = False
        n_ops = 2*len(operators)
        n_mll_bins = len(mll_bins)-1

        # sumw and sumw2 for stat uncertainty
        sumw = np.zeros((n_mll_bins, n_costheta, n_rapll, n_ops))
        sumw2 = np.zeros_like(sumw)

        for i in range(n_costheta):
            for j in range(n_rapll):
                for sample in var.keys():
                    data = var[sample][region]
                    mll_vals = np.array(data["mll"])
                    costheta_vals = np.array(data["costhetastar_bins"])
                    rapll_abs_vals = np.array(data["rapll_abs"])
                    #print(sample, rapll_abs_vals)
                    weights = np.array(data["weight"])
                    sumw_sample = data["sumw"]

                    mask_xy = (np.digitize(costheta_vals, costheta_bins)-1 == i) & \
                              (np.digitize(rapll_abs_vals, rapll_abs_bins)-1 == j)
                    if not np.any(mask_xy):
                        continue

                    xs = float(cross_sections[sample]["xsec"])
                    scale = renorm(xs, sumw_sample, lumi)

                    for op_idx, op in enumerate(operators):
                        # LIN component
                        w_lin = weights * np.array(data[f"{op}_lin"])
                        counts_lin, _ = np.histogram(mll_vals[mask_xy], bins=mll_bins, weights=w_lin[mask_xy])
                        counts2_lin, _ = np.histogram(mll_vals[mask_xy], bins=mll_bins, weights=(w_lin[mask_xy])**2)
                        sumw[:, i, j, 2*op_idx] += counts_lin*scale
                        sumw2[:, i, j, 2*op_idx] += counts2_lin*scale**2

                        # QUAD component
                        w_quad = weights * np.array(data[f"{op}_quad"])
                        counts_quad, _ = np.histogram(mll_vals[mask_xy], bins=mll_bins, weights=w_quad[mask_xy])
                        counts2_quad, _ = np.histogram(mll_vals[mask_xy], bins=mll_bins, weights=(w_quad[mask_xy])**2)
                        sumw[:, i, j, 2*op_idx+1] += counts_quad*scale
                        sumw2[:, i, j, 2*op_idx+1] += counts2_quad*scale**2

        # --- Step 1: min events check (again, all operators) ---
        #min_counts = sumw.min(axis=(1,2))
        #bad_bins_events = np.where((min_counts < min_weighted_events).any(axis=1))[0]

        # --- Step 2: relative uncertainty check ---
        rel_unc = np.full_like(sumw, np.inf, dtype=float)
        nonzero = sumw != 0.0
        
        print("rel_unc shape:", rel_unc.shape)
        
        rel_unc[nonzero] = abs(np.sqrt(sumw2[nonzero]) / sumw[nonzero])
        
        coords = np.argwhere(np.isinf(rel_unc))
        for c in coords:
            print("inf rel unc at:", c, " sumw:", sumw[tuple(c)], " sumw2:", sumw2[tuple(c)], " sqrt(sumw2):", np.sqrt(sumw2[tuple(c)]), " rel_unc:", abs(np.sqrt(sumw2[tuple(c)]) / sumw[tuple(c)]))
            

        coords = np.argwhere(rel_unc>max_rel_unc)
        for c in coords:
            print(f"rel unc > {max_rel_unc} at:", c, "  component: ", labels[c[3]], " sumw:", sumw[tuple(c)], " sumw2:", sumw2[tuple(c)], " sqrt(sumw2):", np.sqrt(sumw2[tuple(c)]))
            
        max_rel_unc_per_mll = rel_unc.max(axis=(1,2,3))
        
        print("max_rel_unc_per_mll")
        
        print(max_rel_unc_per_mll)
        
        bad_bins_unc = np.where(max_rel_unc_per_mll > max_rel_unc)[0]
        
        

        # Combine all failing bins
        #bad_bins = sorted(set(bad_bins_events.tolist() + bad_bins_unc.tolist()))
        bad_bins_unc = sorted(bad_bins_unc.tolist())

        # Print info
        if len(bad_bins_unc) > 0:
            print("\nBins failing relative uncertainty < {:.0f}%:".format(max_rel_unc*100))
            for b in bad_bins_unc:
                print(f"  mll bin {b}: edges [{mll_bins[b]:.1f}, {mll_bins[b+1]:.1f}]")
        else:
            print("All bins have relative uncertainty <= {:.0f}%.".format(max_rel_unc*100))

        """
        # Merge failing bins
        if len(bad_bins) > 0:
            merging_needed = True
            to_delete = []
            i = 0
            while i < len(bad_bins):
                b = bad_bins[i]
                if b < len(mll_bins) - 2:
                    to_delete.append(b+1)
                elif b > 0:
                    to_delete.append(b)
                if i+1 < len(bad_bins) and bad_bins[i+1] == b+1:
                    i += 1
                i += 1
            mll_bins = np.delete(mll_bins, to_delete)

        print(f"Current mll bins after rel_unc check: {mll_bins}")
        """
        
        if len(mll_bins) <= 2:
            return mll_bins  # cannot merge further
        
        
        to_delete = []
        if len(bad_bins_unc) > 0:
            merging_needed = True
            i = 0
            while i < len(bad_bins_unc):
                b = bad_bins_unc[i]

                # Merge the bad bin with its right neighbor if it exists
                if b < len(mll_bins) - 2:  # not last bin
                    to_delete.append(b + 1)
                else:  # last bin, merge with previous
                    to_delete.append(b)

                # Skip next bin if consecutive, but never merge more than 2 bins
                if i + 1 < len(bad_bins_unc) and bad_bins_unc[i + 1] == b + 1:
                    i += 2
                else:
                    i += 1

            # Remove duplicates in case multiple bins target same neighbor
            to_delete = sorted(set(to_delete))
            print(f"Bins to delete: {to_delete}")
            mll_bins = np.delete(mll_bins, to_delete)

            print(f"Current mll bins {len(mll_bins)} after rel_unc check: {mll_bins}")

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
    
    # operators = ["sm","cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]
    operators = ["sm", "cpqmi"]
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
    
    # operators = ["sm","cqlm2","cql32","cqe2","cll1221","cpdc","cpwb","cpl2","c3pl1","c3pl2","cpmu","cpqmi","cpq3i","cpq3","cpqm","cpu","cpd"]
    operators = ["sm", "cpqmi"]
    #operators = ["sm","c3pl1", "c3pl2", "cll1221","cpd","cpl2","cpwb","cpu","cpq3i"]
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
        file_path = [args.input_dir + f"/job_{i}/chunks_job.pkl" for i in [random.randint(0, len(all_files)) for _ in range(0, len(all_files) if args.max_files == -1 else args.max_files)]]
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
    
    costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    rapll_abs_bins = [0.0, 0.5, 1.0, 2.5]

    # var contains your combined data
    # var = {"mll": [...], "costhetastar_bins": [...], "rapll_abs": [...], "weight": [...]}

    mll_bins = optimize_mll_binning(var__, costheta_bins, rapll_abs_bins, min_bin_width, region=regions__[0])
    print(f"Optimized mll bins: {list(mll_bins)}")


if __name__ == "__main__":
    main()
