import zlib, pickle
from config import datasets, lumi, samples
from spritz.framework.framework import (  # noqa: F401
    read_chunks,
)
from typing import NewType
import sys 
import multiprocessing as mp
import copy
from functools import partial
import uproot 
import argparse

def mkdir(path):
    import os

    if not os.path.exists(path):
        os.makedirs(path)
    return

def get_args():
    parser = argparse.ArgumentParser(
        description="Run analysis over NanoAOD files with optional reweighting."
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        default="plots",
        help="Output path for plots"
    )

    parser.add_argument(
        "-j", "--nworkers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--ptll-rew",
        type=str,
        default="",
        help="DY ptll reweighting file name"
    )

    return parser.parse_args()

def renorm(h, xs, sumw, lumi):
    scale = xs * 1000 * lumi / sumw
    _h = h.copy()  # copy histogram

    # Get the structured view
    view = _h.view(flow=False)  # flow=False to ignore under/overflow bins

    # Scale values
    if "value" in view.dtype.names:
        view["value"] *= scale
    else:
        # fallback for plain Double storage (no variance tracking)
        _h.view(flow=False)[:] *= scale

    # Scale variances if present
    if "variance" in view.dtype.names and view["variance"] is not None:
        view["variance"] *= scale**2

    return _h

def load_xsecs():
    import json

    # open and read the file
    with open("../../../data/Full2018v9/samples/samples.json", "r") as f:
        samples = json.load(f)

    return samples

def read_file(path):
    with open(path, "rb") as f:
        compressed = f.read()
        decompressed = zlib.decompress(compressed)
        data = pickle.loads(decompressed)
        return data


Result = NewType("Result", dict[str, dict])

def read_inputs(inputs: list[str]) -> list[Result]:
    inputs_obj = []
    for input in inputs:
        print(input)
        job_result = read_chunks(input)
        # print("JOB RESULT")
        # print(job_result, job_result == -99999, inputs)
        new_job_result = []
        if isinstance(job_result, list):
            for job_result_single in job_result:
                if job_result_single["result"] != {}:
                    new_job_result.append(job_result_single["result"]["real_results"])
            # job_result = new_job_result
            # if check_input(job_result):
            #     inputs_obj.append(job_result["real_results"])
            inputs_obj.extend(new_job_result)
        else:
            # job_result = {k: v for k, v in job_result.items() if k in ["result", "error"]}
            # del job_result["result"]["performance"]
            # if check_input(job_result):
            #     inputs_obj.append(job_result["real_results"])
            inputs_obj.append(job_result)
    # print(inputs_obj)
    print("-->Done ", inputs)
    return inputs_obj

import hist
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)

args = get_args()

data_samples = samples["Data"]["samples"]
others = []
for i in samples:
    if i != "Data":
        others += samples[i]["samples"]
        

variables = {
    "GenPtLL": {
        "name": "GenPtLL",
        "title": "Generated pT of the dilepton system",
        "binning": (0, 100, 50),
        "xaxis": r"Gen $p_{T}^{\ell\ell}$ [GeV]",
    },
    "ptll": {
        "name": "ptll",
        "title": "Reconstructed pT of the dilepton system",
        "binning": (0, 600, 50),
        "xaxis": r"$p_{T}^{\ell\ell}$ [GeV]"
    },
    "mll": {
        "name": "mll",
        "title": "Invariant mass of the dilepton system",
        "binning": (50, 200, 60),
        "xaxis": r"$m_{\ell\ell}$ [GeV]"
    }
}

regions = ["inc_ee", "inc_mm", "inc_em"]

histos = {
    region: {
        "GenPtLL": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["GenPtLL"]["binning"]), name=variables['GenPtLL']["name"], label=variables["GenPtLL"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples
        },
        "ptll": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["ptll"]["binning"]), name=variables['ptll']["name"], label=variables["ptll"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples
        },
        "mll": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["mll"]["binning"]), name=variables['mll']["name"], label=variables["mll"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples
        }
    } for region in regions
}

def process_file(path, ptll_rew=False, rew_samples=[]):
    """Worker function: read file, fill histograms, return partial histos."""
    local_histos = copy.deepcopy(histos)  # each process starts with empty copy
    
    if ptll_rew:
        # load reweighting factor
        with uproot.open(ptll_rew) as f:
            h_ratio = f["h_ratio"].to_hist()  # back as hist.Hist
        
        corr_values = h_ratio.values()
        bin_edges = h_ratio.axes[0].edges
            
    job_results = read_inputs([path])
    for chunk in job_results:
        for dataset in chunk:
            sumw = chunk[dataset]["sumw"]
            for region in regions:
                if region not in chunk[dataset]['events']:
                    continue
                for var in variables:
                    values = chunk[dataset]['events'][region][var]
                    if ptll_rew and dataset in rew_samples:
                        # apply ptll reweighting
                        weights = chunk[dataset]['events'][region].get("weight", np.ones_like(values))
                        gen_ptll = chunk[dataset]['events'][region].get("GenPtLL", np.ones_like(values))
                        # digitize GenPtLL into bins
                        indices = np.digitize(gen_ptll, bin_edges) - 1  # bin indices

                        # clip to valid range (avoid overflow/underflow issues)
                        indices = np.clip(indices, 0, len(corr_values) - 1)

                        # reweights = scale factors per event
                        reweights = corr_values[indices]
                        weights = weights * reweights
                    else: 
                        weights = chunk[dataset]['events'][region].get("weight", np.ones_like(values))
                        
                    local_histos[region][var][dataset]["hist"].fill(values, weight=weights)
                    #print(f"Filling {region} {var} {dataset} with {len(values)} entries and sumw {sumw} variances {local_histos[region][var][dataset]["hist"].variances()}")
                    local_histos[region][var][dataset]["sumw"] += sumw

    return local_histos


def merge_histos(h1, h2):
    """Add two nested histo dicts together."""
    for region in regions:
        for var in variables:
            for dataset in h1[region][var]:
                h1[region][var][dataset]["hist"] += h2[region][var][dataset]["hist"]
                h1[region][var][dataset]["sumw"] += h2[region][var][dataset]["sumw"]
    return h1


# detect if we have a reweight map 
ptll_rew=False
rew_samples=[]

if args.ptll_rew:
    ptll_rew = args.ptll_rew
    rew_samples = ["DYtt_M-50", "DYmm_M-50","DYJetsToMuMu_M-100to200","DYJetsToMuMu_M-200to400","DYJetsToMuMu_M-400to500","DYJetsToMuMu_M-500to700","DYJetsToMuMu_M-700to800","DYJetsToMuMu_M-800to1000","DYJetsToMuMu_M-1000to1500","DYJetsToMuMu_M-1500to2000","DYJetsToMuMu_M-2000toInf","DYee_M-50","DYJetsToEE_M-100to20","DYJetsToEE_M-200to400","DYJetsToEE_M-400to500","DYJetsToEE_M-500to700","DYJetsToEE_M-700to800","DYJetsToEE_M-800to1000","DYJetsToEE_M-1000to1500","DYJetsToEE_M-1500to2000","DYJetsToEE_M-2000toInf"]
    print(f"---> Using ptll reweighting from {ptll_rew} for samples {rew_samples}")

output_folder = args.output
mkdir(output_folder)
    
    
xss = load_xsecs()
print("---> Loading data")

nfiles = 100
nworkers = args.nworkers
input_files = [f"../condor/job_{i}/chunks_job.pkl" for i in range(nfiles)]

print("---> Loading data with multiprocessing")
with mp.Pool(processes=nworkers) as pool:
    func = partial(process_file, ptll_rew=ptll_rew, rew_samples=rew_samples)
    partial_histos = pool.map(func, input_files)

print("---> Merging results")
global_histos = partial_histos[0]
for h in partial_histos[1:]:
    merge_histos(global_histos, h)

# Now use global_histos instead of histos in your plotting code
histos = global_histos
print("---> Histos merged, ready for plotting")

for region in regions: 
    for var in variables:
        print(f"Plotting {region} {var}")

        # --- Build MC ---
        first = others[0]
        scale = float(xss["samples"][datasets[first]["files"]]["xsec"])
        sumw = histos[region][var][first]["sumw"]
        h_MC = renorm(histos[region][var][first]["hist"], scale, sumw, lumi)
        
        for i in others[1:]:
            scale = float(xss["samples"][datasets[i]["files"]]["xsec"])
            sumw = histos[region][var][i]["sumw"]
            if sumw == 0:
                continue
            _h = renorm(histos[region][var][i]["hist"], scale, sumw, lumi)
            h_MC += _h
            
        # --- Build Data ---
        first = data_samples[0]
        h_Data = histos[region][var][first]["hist"].copy()
        for i in data_samples[1:]:
            _h = histos[region][var][i]["hist"].copy()
            h_Data += _h
        
        # --- Figure with two panels ---
        fig, (ax, rax) = plt.subplots(
            2, 1,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            figsize=(8, 8),
            sharex=True
        )

        # --- Main panel ---
        h_MC.plot(ax=ax, histtype="step", linewidth=2, label="MC [{:.1f}]".format(sum(h_MC.values())))
        h_Data.plot(ax=ax, histtype="errorbar", color="black", label="Data [{:.1f}]".format(sum(h_Data.values())), markersize=5)

        ax.set_ylabel("Events")
        ax.set_yscale("log")
        #ax.set_xscale("log")
        ax.set_xlim(variables[var]["binning"][0], variables[var]["binning"][1])
        ax.set_xlabel("")
        #ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        # Add CMS label
        hep.cms.label(ax=ax, year=2018, lumi="{:.1f}".format(lumi), com=13, data=True)

        # --- Ratio plot: Data / MC ---
        mc_vals = h_MC.values()
        mc_errs = np.sqrt(h_MC.variances()) if h_MC.variances() is not None else np.zeros_like(mc_vals)
        data_vals = h_Data.values()
        data_errs = np.sqrt(h_Data.variances()) if h_Data.variances() is not None else np.zeros_like(data_vals)

        # Avoid division by zero
        mask = mc_vals > 0
        ratio = np.ones_like(data_vals) * np.nan
        ratio[mask] = data_vals[mask] / mc_vals[mask]

        ratio_err = np.zeros_like(ratio)
        ratio_err[mask] = data_errs[mask] / mc_vals[mask]

        # Bin centers for plotting
        bin_edges = h_Data.axes[0].edges
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        rax.errorbar(
            bin_centers, ratio, yerr=ratio_err,
            fmt="o", color="black", markersize=4
        )
        
        # --- Add MC uncertainty band (shaded grey) ---
        mc_unc_ratio = np.zeros_like(ratio)
        mc_unc_ratio[mask] = mc_errs[mask] / mc_vals[mask]  # fractional MC uncertainty
        rax.fill_between(
            bin_centers,
            1 - mc_unc_ratio,
            1 + mc_unc_ratio,
            color="gray",
            alpha=0.4,
            step="mid",
            label="MC stat. unc."
        )
        rax.axhline(1.0, color="red", linestyle="--")

        

        rax.set_ylabel("Data/MC")
        rax.set_xlabel(variables[var]["xaxis"])
        rax.set_ylim(0.91, 1.09 )  # adjust as needed
        #rax.grid(True, linestyle="--", alpha=0.6)

        # --- Save ---
        plt.savefig(f"{output_folder}/{region}_{var}.pdf", bbox_inches="tight")
        plt.savefig(f"{output_folder}/{region}_{var}.png", bbox_inches="tight")
        plt.close(fig)
