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
from itertools import combinations
from tqdm import tqdm
import mplhep as hep
plt.style.use(hep.style.CMS)
from config import datasets, regions, lumi, samples, year
import time
from spritz.framework.framework import get_fw_path
import tempfile

# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    parser.add_argument("-o", "--output", default="plots", help="Output folder")
    parser.add_argument("-j", "--nworkers", type=int, default=4, help="Number of workers")
    parser.add_argument("-rew", "--reweighting", type=str, default="sm", help="Name of reweighting branches that you'd like to plot separated by a comma")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with input pickle files")
    parser.add_argument("--lhe-json", type=str, required=True, help="JSON with LHE reweighting weights")
    parser.add_argument("--max-files", dest="max_files", type=int, required=False, help="maximum files to process, by default all", default=-1)
    parser.add_argument("--only-merge", dest="only_merge", required=False, help="Only merge the results in output directory", default=False, action="store_true")
    parser.add_argument("--save-matrix", dest="save_matrix", required=False, help="Save matrix for correlated stat MC", default=False, action="store_true")
    parser.add_argument("--normalize", dest="normalize", required=False, help="Normalize to SM all shapes", default=False, action="store_true")
    parser.add_argument("--lumi", dest="luminosity", type=float, required=False, help="Luminosity to normalize, default None", default=None)
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
#  -----------------------------
# Cross sections
#  -----------------------------

def renorm(xs, sumw, lumi, squareit=False):
    scale = xs * 1000 * lumi / sumw
    if squareit:
        # if the histo is a variance, need to take square of scale
        # to scale nominal bin content
        scale = scale ** 2
    # print(scale)
    return scale

path_fw = get_fw_path()

with open(f"{path_fw}/data/{year}/samples/samples.json") as file:
        cross_sections = json.load(file)["samples"]


def hist_unroll(h):
    """
    Unrolls n-dimensional histogram

    Parameters
    ----------
    h : hist
        Histogram to unroll

    Returns
    -------
    hist
        Unrolled 1-dimensional histogram
    """
    dimension = len(h.axes)

    # if dimension != 2:
    #     raise Exception(
    #         "Error in hist_unroll: can only unroll 2D histograms, while got ",
    #         dimension,
    #         "dimensions",
    #     )

    if dimension == 1:
        return h

    if dimension == 2:
        numpy_view = h.view()  # no under/overflow!
        nx = numpy_view.shape[0]
        ny = numpy_view.shape[1]
        h_unroll = hist.Hist(hist.axis.Regular(nx * ny, 0, nx * ny), hist.storage.Weight())

        numpy_view_unroll = h_unroll.view()
        numpy_view_unroll.value = numpy_view.value.T.flatten()
        numpy_view_unroll.variance = numpy_view.variance.T.flatten()

        return h_unroll

    if dimension == 3:
        numpy_view = h.view()  # no under/overflow!
        nx = numpy_view.shape[0]
        ny = numpy_view.shape[1]
        nz = numpy_view.shape[2]

        h_unroll = hist.Hist(hist.axis.Regular(nx * ny * nz, 0, nx * ny * nz), hist.storage.Weight())

        numpy_view_unroll = h_unroll.view()
        numpy_view_unroll.value = numpy_view.value.T.flatten()
        numpy_view_unroll.variance = numpy_view.variance.T.flatten()

        return h_unroll



# -----------------------------
# Process a single file
# -----------------------------
from itertools import combinations
import numpy as np
import hist

def build_hist(var_cfg):
    if "binning" in var_cfg:
        return hist.Hist(
            hist.axis.Variable(np.linspace(*var_cfg["binning"]),
                                name=var_cfg.get("name", ""),
                                label=var_cfg.get("xaxis", "")),
            storage=hist.storage.Weight()
        )
    elif "axis" in var_cfg:
        if isinstance(var_cfg["axis"], list):
            return hist.Hist(*var_cfg["axis"], storage=hist.storage.Weight())
        else:
            return hist.Hist(var_cfg["axis"], storage=hist.storage.Weight())
    else:
        raise ValueError(f"Variable config missing 'binning' or 'axis': {var_cfg}")

def process_file(file_path, regions, samples_to_process, variables, reweight_map, save_matrix, out_dir=None):
    print(file_path)
    job_results = read_inputs([file_path])

    # --- Precompute things used multiple times ---
    non_sm_ops = [op for op in reweight_map if op != "sm"]
    op_pairs = list(combinations(non_sm_ops, 2))


    # --- Initialize local histograms ---
    local_histos = {
        region: {
            shape: {var: {sample: {"histo": build_hist(cfg), "sumw": 0.0}
                          for sample in samples_to_process}
                    for var, cfg in variables.items()}
            for shape in (
                ["sm"] +
                [s for op in reweight_map if op != "sm" for s in [op, op+"_m1", op+"_lin", op+"_quad"]] +
                [f"{a}_{b}" for a, b in op_pairs] +
                [f"{a}_{b}_mix" for a, b in op_pairs]
            )
        }
        for region in regions
    }
    
    # local_histos = {region: {} for region in regions}
    
    if save_matrix:
        for region in regions:
            for shape in (
                ["sm_variance"] +
                [f"{op}_plus_variance" for op in non_sm_ops] +
                [f"{op}_minus_variance" for op in non_sm_ops] +
                [f"sm_{op}_mixed_variance" for op in non_sm_ops] +
                [f"sm_{op}_m1_mixed_variance" for op in non_sm_ops] +
                [f"{op}_{op}_m1_mixed_variance" for op in non_sm_ops] 
            ):
                local_histos[region][shape] = {}
                for var, cfg in variables.items():
                    local_histos[region][shape][var] = {}
                    for sample in samples_to_process:
                        local_histos[region][shape][var][sample] = {"histo": build_hist(cfg), "sumw": 0.0}
            
            shapes__ =  []
            for op1, op2 in op_pairs:
                shapes__ += [f"{op1}_{op2}_variance", f"sm_{op1}_{op2}_mixed_variance", f"{op1}_{op2}_mixed_variance", f"{op1}_{op2}_m1_mixed_variance", f"{op1}_{op1}_{op2}_mixed_variance", f"{op2}_{op1}_m1_mixed_variance", f"{op2}_{op1}_{op2}_mixed_variance", f"{op1}_m1_{op2}_m1_mixed_variance", f"{op1}_m1_{op1}_{op2}_mixed_variance", f"{op2}_m1_{op1}_{op2}_mixed_variance"]
                
                # single mixed 
                
                for op3 in non_sm_ops:
                    if op3 != op1 and op3 != op2:
                        shapes__ += [f"{op3}_{op1}_{op2}_mixed_variance", f"{op3}_m1_{op1}_{op2}_mixed_variance"]
                                
                # mixed mixed 
                for op3, op4 in op_pairs:
                    if (op3, op4) == (op1, op2):
                        continue
                    
                    sn = f"{op1}_{op2}_{op3}_{op4}_mixed_variance"
                    sn_alt = f"{op3}_{op4}_{op1}_{op2}_mixed_variance"

                    if sn in shapes__ or sn_alt in shapes__:
                        continue
                        
                    shapes__.append(f"{op1}_{op2}_{op3}_{op4}_mixed_variance")
                    
            for shape in shapes__:
                local_histos[region][shape] = {}
                for var, cfg in variables.items():
                    local_histos[region][shape][var] = {}
                    for sample in samples_to_process:
                        local_histos[region][shape][var][sample] = {"histo": build_hist(cfg), "sumw": 0.0}
                            
            # print(local_histos[region].keys())
            # print(f"---> Final length : {len(local_histos[region].keys())}")
                    

    # --- Process events ---
    for idx, chunk in enumerate(tqdm(job_results, desc=f"{file_path}", leave=False)):
        for dataset, dset_data in chunk.items():
            #print(dataset, dataset not in samples_to_process)
            if dataset not in samples_to_process:
                continue

            sumw = dset_data['sumw']
            if sumw == 0:
                print(f"File path: {file_path} Dataset: {dataset}, sumw: {sumw}")

            for region in regions:
                region_data = dset_data['events'].get(region)
                if region_data is None:
                    print(f"Region {region} not found in dataset {dataset}")
                    continue
                
                
                if "sm" not in region_data:
                    print(f"Warning: 'sm' weight not found in events for dataset {dataset} region {region} file {file_path}")
                    continue
                

                # cache variable values once
                var_cache = {
                    var: region_data[var] for var in variables if var in region_data
                }
                
                
                base_weights = region_data["weight"]
                w_sm = region_data["sm"]

                # --- single-operator weights ---
                for op in reweight_map:
                    
                    if op == "sm":
                        weights_and_labels = [(w_sm, "sm")]
                        
                    else:
                        w_op = region_data[op]
                        w_op_m1 = region_data[f"{op}_m1"]
                        w_lin = 0.5 * (w_op - w_op_m1)
                        w_quad = 0.5 * (w_op + w_op_m1 - 2 * w_sm)
                        weights_and_labels = [
                            (w_op, op),
                            (w_op_m1, f"{op}_m1"),
                            (w_lin, f"{op}_lin"),
                            (w_quad, f"{op}_quad"),
                        ]

                    for weight, label in weights_and_labels:
                        total_weight = base_weights * weight
                        for var, cfg in variables.items():
                            vals = var_cache.get(var)
                            histo = local_histos[region][label][var][dataset]["histo"]

                            if vals is not None:
                                histo.fill(vals, weight=total_weight)
                            else:
                                # multi-dim variables
                                if "axis" in cfg:
                                    if isinstance(cfg["axis"], list):
                                        vals_dict = {ax.name: region_data[ax.name] for ax in cfg["axis"]}
                                        histo.fill(**vals_dict, weight=total_weight)
                                    else:
                                        name = cfg["axis"].name
                                        histo.fill(region_data[name], weight=total_weight)
                            local_histos[region][label][var][dataset]["sumw"] += sumw
                            
                # Fill MC stat unc matrix for single ops
                if save_matrix:
                    # save sm only once 
                    w_sm = base_weights * w_sm
                    
                    variance_weight_sm = (w_sm) ** 2
                    weights_and_labels = [
                            (variance_weight_sm, "sm_variance"),
                        ]
                    # save variances and cross products 
                    for op in reweight_map:
                        if op != "sm":
                            
                            w_op = base_weights * region_data[op]
                            w_op_m1 = base_weights * region_data[op + "_m1" ]
                            
                            variance_p1_weight = (w_op) ** 2
                            variance_m1_weight = (w_op_m1) ** 2
                            
                            mixed_01  = 2*(w_sm * w_op)
                            mixed_0m1 = 2*(w_sm * w_op_m1)
                            mixed_1m1 = 2*(w_op * w_op_m1)
                            
                            weights_and_labels.append((variance_p1_weight, f"{op}_plus_variance"))
                            weights_and_labels.append((variance_m1_weight, f"{op}_minus_variance"))
                            weights_and_labels.append((mixed_01, f"sm_{op}_mixed_variance"))
                            weights_and_labels.append((mixed_0m1, f"sm_{op}_m1_mixed_variance"))
                            weights_and_labels.append((mixed_1m1, f"{op}_{op}_m1_mixed_variance"))

                    
                    for weight, label in weights_and_labels:
                        total_weight = weight
                        for var, cfg in variables.items():
                            vals = var_cache.get(var)
                            histo = local_histos[region][label][var][dataset]["histo"]

                            if vals is not None:
                                histo.fill(vals, weight=total_weight)
                            else:
                                # multi-dim variables
                                if "axis" in cfg:
                                    if isinstance(cfg["axis"], list):
                                        vals_dict = {ax.name: region_data[ax.name] for ax in cfg["axis"]}
                                        histo.fill(**vals_dict, weight=total_weight)
                                    else:
                                        name = cfg["axis"].name
                                        histo.fill(region_data[name], weight=total_weight)
                            local_histos[region][label][var][dataset]["sumw"] += sumw
                    

                # --- two-operator weights ---
                for op1, op2 in op_pairs:
                    w_name = f"{op1}_{op2}"
                    region_keys = region_data.keys()

                    if w_name not in region_keys:
                        w_name_alt = f"{op2}_{op1}"
                        if w_name_alt not in region_keys:
                            print(f"Warning: {w_name} not found in events")
                            continue
                        w_name = w_name_alt

                    w_mix = region_data[w_name] * base_weights
                    w_op1 = region_data[op1] * base_weights
                    w_op2 = region_data[op2] * base_weights
                    w_mix_only = w_mix + w_sm - w_op1 - w_op2
                    
                    weights__ = [w_mix, w_mix_only]
                    labels__ = [f"{op1}_{op2}", f"{op1}_{op2}_mix"]
                    
                    if save_matrix:
                        
                        # need to compute all cross products 
                        # op_pairs does not have sm in it 
                        
                        variance_p11_weight = (w_mix) ** 2 # F**2 sum(w(11)**2)
                        variance_p11_sm_weight = 2*(w_sm * w_mix) # 2AF sum(w(0)w(11))
                        
                        w_op1_m1 = region_data[op1 + "_m1"] * base_weights
                        w_op2_m1 = region_data[op2 + "_m1"] * base_weights
                        
                        mixed_op1_p1_op2_p1 = 2*(w_op1 * w_op2)
                        mixed_op1_p1_op2_m1 = 2*(w_op1 * w_op2_m1)
                        mixed_op1_p1_11 = 2*(w_op1 * w_mix)
                        mixed_op2_p1_op1_m1 = 2*(w_op2 * w_op1_m1)
                        mixed_op2_p1_11 = 2*(w_op2 * w_mix)
                        mixed_op1_m1_op2_m1 = 2*(w_op1_m1 * w_op2_m1) 
                        mixed_op1_m1_11 = 2*(w_op1_m1 * w_mix)
                        mixed_op2_m1_11 = 2*(w_op2_m1 * w_mix)
                        
                        weights__ += [variance_p11_weight, variance_p11_sm_weight, mixed_op1_p1_op2_p1, mixed_op1_p1_op2_m1, mixed_op1_p1_11, mixed_op2_p1_op1_m1, mixed_op2_p1_11, mixed_op1_m1_op2_m1, mixed_op1_m1_11, mixed_op2_m1_11]
                        labels__ += [f"{op1}_{op2}_variance", f"sm_{op1}_{op2}_mixed_variance",  f"{op1}_{op2}_mixed_variance", f"{op1}_{op2}_m1_mixed_variance", f"{op1}_{op1}_{op2}_mixed_variance", f"{op2}_{op1}_m1_mixed_variance", f"{op2}_{op1}_{op2}_mixed_variance", f"{op1}_m1_{op2}_m1_mixed_variance", f"{op1}_m1_{op1}_{op2}_mixed_variance", f"{op2}_m1_{op1}_{op2}_mixed_variance"]

                        for op3 in non_sm_ops:
                            if op3 != op1 and op3 != op2:
                                w_op3 = region_data[op3] * base_weights
                                w_op3_m1 = region_data[op3 + "_m1"] * base_weights
                                
                                mixed_op3_op1_op2_11 = 2*(w_op3 * w_mix)
                                mixed_op3_m1_op1_op2_11 = 2*(w_op3_m1 * w_mix)
                                weights__ += [mixed_op3_op1_op2_11, mixed_op3_m1_op1_op2_11]
                                labels__ += [f"{op3}_{op1}_{op2}_mixed_variance", f"{op3}_m1_{op1}_{op2}_mixed_variance"]
                                
                        # mixed mixed 
                        for op3, op4 in op_pairs:
                            if (op3, op4) == (op1, op2) or (f"{op3}_{op4}_{op1}_{op2}_mixed_variance" in shapes__):
                                continue 
                            
                            sn = f"{op1}_{op2}_{op3}_{op4}_mixed_variance"
                            sn_alt = f"{op3}_{op4}_{op1}_{op2}_mixed_variance"

                            if sn in labels__ or sn_alt in labels__:
                                continue
                            
                            w_name_2 = f"{op3}_{op4}"
                            w_mix_2 = region_data[w_name_2] * base_weights
                            mixed_op1_op2_op3_op4 = 2*(w_mix * w_mix_2)
                            
                            weights__.append(mixed_op1_op2_op3_op4)
                            labels__.append(f"{op1}_{op2}_{op3}_{op4}_mixed_variance")
                            

                    for weight, label in zip(weights__, labels__):
                        total_weight = weight
                        for var, cfg in variables.items():
                            vals = var_cache.get(var)
                            histo = local_histos[region][label][var][dataset]["histo"]

                            if vals is not None:
                                histo.fill(vals, weight=total_weight)
                            else:
                                if "axis" in cfg:
                                    if isinstance(cfg["axis"], list):
                                        vals_dict = {ax.name: region_data[ax.name] for ax in cfg["axis"]}
                                        histo.fill(**vals_dict, weight=total_weight)
                                    else:
                                        name = cfg["axis"].name
                                        histo.fill(region_data[name], weight=total_weight)
                            local_histos[region][label][var][dataset]["sumw"] += sumw

     # --- Save to disk ---
    if out_dir is None:
        out_dir = tempfile.gettempdir()  # fallback
    os.makedirs(out_dir, exist_ok=True)

    # base_name = os.path.basename(file_path).replace(".pkl", "_histos.pkl")
    fd, out_file = tempfile.mkstemp(suffix="_histos.pkl", dir=out_dir)
    os.close(fd)
    # out_file = os.path.join(out_dir, base_name)

    with open(out_file, "wb") as f:
        pickle.dump(local_histos, f, protocol=4)

    # Free memory
    del local_histos

    return out_file


# -----------------------------
# Merge histograms
# -----------------------------


def merge_histos(h1, h2):
    for region in h2:
        if region not in h1:
            h1[region] = {}
        for operator in h2[region]:
            #print(operator)
            if operator not in h1[region]:
                h1[region][operator] = {}
            for var in h2[region][operator]:
                if var not in h1[region][operator]:
                    h1[region][operator][var] = {}
                for sample in h2[region][operator][var]:
                    if sample not in h1[region][operator][var]:
                        h1[region][operator][var][sample] = {
                            "histo": h2[region][operator][var][sample]["histo"].copy(),
                            "sumw": 0.0
                        }

                    # Unroll histograms to ensure compatible shapes
                    h1_hist = hist_unroll(h1[region][operator][var][sample]["histo"])
                    h2_hist = hist_unroll(h2[region][operator][var][sample]["histo"])
                    h1[region][operator][var][sample]["histo"] = h1_hist + h2_hist
                    h1[region][operator][var][sample]["sumw"] += h2[region][operator][var][sample]["sumw"]

    return h1

"""
def merge_histos_files_pair(file_pair, out_dir):
    f1, f2 = file_pair
    print(f1, f2)
    with open(f1, "rb") as f:
        h1 = pickle.load(f)
    with open(f2, "rb") as f:
        h2 = pickle.load(f)
    
    print("Loaded files")
    merge_histos(h1, h2)  # in-place
    
    # Save merged histograms to a new temporary file
    fd, merged_file = tempfile.mkstemp(prefix="tmp_merge_", suffix=".pkl", dir=out_dir)
    os.close(fd)
    with open(merged_file, "wb") as f:
        pickle.dump(h1, f, protocol=4)
    
    # Optionally delete the original files to free disk
    os.remove(f1)
    os.remove(f2)
    
    return merged_file
    
"""

def merge_histos_files_pair(file_pair, out_dir):
    f1, f2 = file_pair
    with open(f1, "rb") as ff1:
        h1 = pickle.load(ff1)
    with open(f2, "rb") as ff2:
        h2 = pickle.load(ff2)

    merge_histos(h1, h2)  # in-place merge

    # Save merged histograms to a temporary file
    fd, merged_file = tempfile.mkstemp(prefix="tmp_merge_", suffix=".pkl", dir=out_dir)
    os.close(fd)
    with open(merged_file, "wb") as f:
        pickle.dump(h1, f, protocol=4)

    # Delete input files immediately
    #os.remove(f1)
    #os.remove(f2)

    return merged_file

"""
def merge_histos_from_files(file_list, out_dir, nproc=4):
    files = file_list[:]
    
    step = 0
    while len(files) > 1:
        step += 1
        pairs = [(files[i], files[i+1]) for i in range(0, len(files)-1, 2)]
        leftover = files[-1] if len(files) % 2 == 1 else None
        
        print(f"[Step {step}] Merging {len(pairs)} pairs...")
        with mp.Pool(nproc) as pool:
            merge_func = partial(merge_histos_files_pair, out_dir=out_dir)
            merged_files = list(tqdm(pool.imap_unordered(merge_func, pairs), total=len(pairs)))
    
        files = merged_files
        if leftover:
            files.append(leftover)
    
    # Load the final merged histogram into memory
    final_file = files[0]
    with open(final_file, "rb") as f:
        final_histos = pickle.load(f)
    
    return final_histos
"""

def merge_histos_from_files(file_list, out_dir, nproc=4):
    files = list(file_list)

    step = 0
    while len(files) > 1:
        step += 1
        print(f"[Step {step}] {len(files)} files remaining...")

        # Pair files for merging
        pairs = [(files[i], files[i+1]) for i in range(0, len(files)-1, 2)]
        leftover = files[-1] if len(files) % 2 == 1 else None

        merge_func = partial(merge_histos_files_pair, out_dir=out_dir)
        with mp.Pool(nproc) as pool:
            merged_files = list(tqdm(pool.imap_unordered(merge_func, pairs), total=len(pairs), desc=f"Step {step}"))

        # Carry leftover forward
        files = merged_files
        if leftover:
            files.append(leftover)

    # Only one file remains now
    final_file = files[0]

    # Rename/move to final output path if you want
    final_output = os.path.join(out_dir, "histos_merged.pkl")
    os.replace(final_file, final_output)

    with open(final_output, "rb") as f:
        final_histos = pickle.load(f)
    
    return final_histos

def scale_samples(histos, lumi):
    for region in histos.keys():
        for operator in histos[region].keys():
            square=False
            if operator.endswith("variance"):
                square=True 
            for var in histos[region][operator].keys():
                for sample in histos[region][operator][var].keys():
                    if sample == "all": continue 
                    xs = float(cross_sections[sample]["xsec"])
                    sumw = histos[region][operator][var][sample]["sumw"]
                    scale = 1.0
                    if sumw != 0:
                        scale = renorm(xs, sumw, lumi, squareit=square)
                    histos[region][operator][var][sample]["histo"] = histos[region][operator][var][sample]["histo"] * scale
                
    return histos

def merge_samples(histos):
    for region in histos.keys():
        for operator in histos[region].keys():
            for var in histos[region][operator].keys():
                samples_to_process = list(histos[region][operator][var].keys())
                default =  deepcopy(histos[region][operator][var][samples_to_process[0]]["histo"])
                for sample in samples_to_process[1:]:
                    default += histos[region][operator][var][sample]["histo"]
                histos[region][operator][var]["all"] = {"histo": default, "sumw": 0}
                    
    return histos

def normalize_to_sm(histos, reweight_map, save_matrix):
    
    # nominal histos can be normalized by dividing for SM 
    # while matrix elements should be divided by sm**2
    non_sm_ops = [op for op in reweight_map if op != "sm"]
    op_pairs = list(combinations(non_sm_ops, 2))
    
    nominal_histos = ["sm"] + [s for op in reweight_map if op != "sm" for s in [op, op+"_m1", op+"_lin", op+"_quad"]] + [f"{a}_{b}" for a, b in op_pairs] + [f"{a}_{b}_mix" for a, b in op_pairs]
    matrix_histos = []
    
    if save_matrix:
        matrix_histos = ["sm_variance"] + [f"{op}_plus_variance" for op in non_sm_ops] + [f"{op}_minus_variance" for op in non_sm_ops] + [f"sm_{op}_mixed_variance" for op in non_sm_ops] + [f"sm_{op}_m1_mixed_variance" for op in non_sm_ops] + [f"{op}_{op}_m1_mixed_variance" for op in non_sm_ops]
                
        for op1, op2 in op_pairs:
                matrix_histos += [f"{op1}_{op2}_variance", f"sm_{op1}_{op2}_mixed_variance", f"{op1}_{op2}_mixed_variance", f"{op1}_{op2}_m1_mixed_variance", f"{op1}_{op1}_{op2}_mixed_variance", f"{op2}_{op1}_m1_mixed_variance", f"{op2}_{op1}_{op2}_mixed_variance", f"{op1}_m1_{op2}_m1_mixed_variance", f"{op1}_m1_{op1}_{op2}_mixed_variance", f"{op2}_m1_{op1}_{op2}_mixed_variance"]
            
                # single mixed 
                
                for op3 in non_sm_ops:
                    if op3 != op1 and op3 != op2:
                        matrix_histos += [f"{op3}_{op1}_{op2}_mixed_variance", f"{op3}_m1_{op1}_{op2}_mixed_variance"]
                                
                # mixed mixed 
                for op3, op4 in op_pairs:
                    if (op3, op4) == (op1, op2):
                        continue
                    
                    sn = f"{op1}_{op2}_{op3}_{op4}_mixed_variance"
                    sn_alt = f"{op3}_{op4}_{op1}_{op2}_mixed_variance"

                    if sn in matrix_histos or sn_alt in matrix_histos:
                        continue
                        
                    matrix_histos.append(f"{op1}_{op2}_{op3}_{op4}_mixed_variance")

    
    new_h = {}
    for region in histos.keys():
        new_h[region] = {}
        for op in histos[region].keys():
            new_h[region][op] = {}
            is_matrix = False 
            if op in matrix_histos:
                is_matrix = True
            elif op in nominal_histos:
                is_matrix = False 
            else:
                print(f"-->ERROR shape {op} not in any histos")
                
            for var in histos[region][op].keys():
                new_h[region][op][var] = {}
                for sample in histos[region][op][var].keys():
                    
                    
                    
                    h = histos[region][op][var][sample]["histo"]
                    h_sm = histos[region]["sm"][var][sample]["histo"]

                    h_val = h.view().value
                    h_var = h.view().variance
                    sm_val = h_sm.view().value

                    mask = sm_val != 0  # avoid division by zero
                    ratio = np.zeros_like(h_val)
                    variance = np.zeros_like(h_var)
                    
                    
                    if not is_matrix:
                        ratio[mask] = h_val[mask] / sm_val[mask]
                        variance[mask] = h_var[mask] / (sm_val[mask] ** 2) # important to have this properly
                    else:
                        ratio[mask] = h_val[mask] / (sm_val[mask]**2) # because it is a variance in the end
                        variance[mask] = np.zeros_like(h_val[mask])

                    h_new = h.copy()
                    h_new.view().value = ratio
                    h_new.view().variance = variance
                    
                    new_h[region][op][var][sample] = {"histo": h_new, "sumw": histos[region][op][var][sample]["sumw"]}
                    
    return new_h

# -----------------------------
# Plot histograms
# -----------------------------
def plot_hist_worker(args):
    h, region, operator, var, variables, output_folder = args
    h = h["histo"]
    print(region, operator, var, output_folder)

    fig, ax = plt.subplots(figsize=(8,8))
    bin_edges = h.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    values = h.values()
    errors = np.sqrt(h.variances()) if h.variances() is not None else np.zeros_like(values)

    ax.step(bin_edges[:-1], values, where='post', color='black', label=f"{operator}")
    ax.fill_between(bin_edges[:-1], values-errors, values+errors, step='post', alpha=0.3)
    
    # ax.errorbar(bin_centers, values, yerr=errors, fmt="o", color="black", label=f"{operator} {comp}")
    # h.plot1d(ax=ax, histtype='step', color='black', label=f"{operator} {comp}")
    ax.set_xlabel(variables[var]["xaxis"])
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.legend()
    hep.cms.label(ax=ax, year=2018, com=13, data=False)

    # outname_pdf = os.path.join(sample_dir, f"{region}_{var}_{operator}_{comp}.pdf")
    outname_png = os.path.join(output_folder, f"{region}_{var}_{operator}.png")
    outname_pdf = os.path.join(output_folder, f"{region}_{var}_{operator}.pdf")
    # plt.savefig(outname_pdf, bbox_inches="tight")
    plt.savefig(outname_png, bbox_inches="tight")
    plt.savefig(outname_pdf, bbox_inches="tight")
    plt.close(fig)
    
# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    mkdir(args.output)

    

    # DY samples and regions
    # regions__ = list(regions.keys())
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
    
    minimal_bin_width = initial_mll_binning(min_bin_width)
    
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
    gen_mll_optimized = [50, 64, 76, 82, 86, 90, 98, 103, 121, 127, 130, 133, 148, 151, 154, 157, 163, 166, 172, 178, 184, 205, 210, 220, 235, 240, 260, 265, 325, 345, 500, 530, 570, 618, 654, 708, 3000]
    

    variables = {
        # "mll": {"binning": (50, 3000, 150), "xaxis": r"$m_{\ell\ell}$ [GeV]"},
        "mll": {"axis": hist.axis.Variable(minimal_bin_width, name="mll"), "xaxis": r"$m_{\ell\ell}$ [GeV]"},
        # "costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]"},
        # "yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]"},
        #"triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(rapll_abs_bins, name="rapll_abs")], "xaxis": r"Triple diff bin"},
        "triple_diff_optimized": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(rapll_abs_bins_opt, name="rapll_abs")], "xaxis": r"Triple diff bin"},
        
        # "triple_diff_rapll_0_0p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_0p5_1p0_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p0_1p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p5_2p5_costheta_m1_m0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-1, -0.5], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # 
        # "triple_diff_rapll_0_0p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_0p5_1p0_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p0_1p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, 0.0], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p5_2p5_costheta_m0p5_m0p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.5, -0.0], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # 
        # "triple_diff_rapll_0_0p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_0p5_1p0_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p0_1p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p5_2p5_costheta_m0p0_0p5": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([-0.0, 0.5], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # 
        # "triple_diff_rapll_0_0p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([0.0, 0.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_0p5_1p0_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([0.5, 1.0], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p0_1p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([1.0, 1.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
        # "triple_diff_rapll_1p5_2p5_costheta_0p5_1p0": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable([0.5, 1.0], name="costhetastar_bins"), hist.axis.Variable([1.5, 2.5], name="rapll_abs")], "xaxis": r"Triple diff bin"},
    }

    # Load LHE reweight JSON
    with open(args.lhe_json) as f:
        reweight_map = json.load(f)
        
    # reweight_map = [i.split("_m1")[0] for i in reweight_map.keys() if "_m1" in i] + ["sm"]

    reweight_map = args.reweighting.split(",")
    # List input files
    input_files = glob(args.input_dir + "/*/chunks_job.pkl")[:args.max_files]
    
    
    if not args.only_merge:

        # Multiprocessing
        with mp.Pool(processes=args.nworkers if len(input_files) > args.nworkers else len(input_files)) as pool:
            func = partial(process_file, regions=regions__, samples_to_process=samples_to_process,
                        variables=variables, reweight_map=reweight_map, save_matrix=args.save_matrix, out_dir=args.output)
            #print(func, input_files)
            partial_histos = pool.map(func, input_files)
        
        print(partial_histos)
        print("Done processing files, now merging histograms...")
        
    else:
        partial_histos = glob(args.output + "/*.pkl")
        print(partial_histos)
        print(partial_histos)
    
    # global_histos = parallel_merge(partial_histos, partial_histos[0], nproc=args.nworkers if len(partial_histos) > args.nworkers else len(partial_histos))
    
    global_histos = merge_histos_from_files(partial_histos, args.output, nproc=args.nworkers if len(partial_histos) > args.nworkers else len(partial_histos))
    
    #sys.exit(0)
    
    # scale samples    
    print("Now scaling histograms...")
    global_histos = scale_samples(global_histos, lumi if args.luminosity == None else args.luminosity)
    # Create overall 
    print("Now merging samples...")
    global_histos = merge_samples(global_histos)
    
    if args.normalize: 
        global_histos = normalize_to_sm(global_histos, reweight_map, args.save_matrix)
    
    # Save merged histograms
    output_file = os.path.join(args.output, "histos_merged.pkl")
    with open(output_file, "wb") as f:
        print(len(global_histos["inc_mm"].keys()))
        pickle.dump(global_histos, f)
    print(f"Histograms saved to {output_file}")
    
    sys.exit(0)
    
    plot_tasks = []
    for region in regions:
        for operator in reweight_map:
            for var in variables:
                h = global_histos[region][operator][var]["all"]
                plot_tasks.append((h, region, operator, var, variables, args.output))
                
    with mp.Pool(processes=args.nworkers) as pool:
        pool.map(plot_hist_worker, plot_tasks)

    
    print(f"Plots saved in {args.output}/")

if __name__ == "__main__":
    main()
