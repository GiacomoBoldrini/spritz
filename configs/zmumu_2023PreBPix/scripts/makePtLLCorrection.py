import zlib, pickle
from config import datasets, lumi, samples
from spritz.framework.framework import (  # noqa: F401
    read_chunks,
)
from typing import NewType
import sys 
import multiprocessing as mp
import copy

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
    return inputs_obj

import hist
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.CMS)

data_samples = samples["Data"]["samples"]
dymumu_samples = []
others = []
for i in samples:
    if i != "Data" and i != "DYll":
        others += samples[i]["samples"]
    elif i == "DYll":
        for sample in samples[i]["samples"]:
            if any(j in sample for j in ["MuMu", "DYmm"]):
                dymumu_samples.append(sample)
            else:
                others.append(sample)
    else:
        continue
        
print(data_samples)
print(dymumu_samples)
print(others)        

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
        "binning": (0, 300, 50),
        "xaxis": r"$p_{T}^{\ell\ell}$ [GeV]"
    },
    "mll": {
        "name": "mll",
        "title": "Invariant mass of the dilepton system",
        "binning": (50, 200, 50),
        "xaxis": r"$m_{\ell\ell}$ [GeV]"
    }
}

# regions = ["inc_ee", "inc_mm", "inc_em"]

# only derive correction in inc_mm region 

regions = ["inc_mm"]

histos = {
    region: {
        "GenPtLL": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["GenPtLL"]["binning"]), name=variables['GenPtLL']["name"], label=variables["GenPtLL"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples + dymumu_samples
        },
        "ptll": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["ptll"]["binning"]), name=variables['ptll']["name"], label=variables["ptll"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples + dymumu_samples
        },
        "mll": {
            i: {"hist": hist.Hist(hist.axis.Variable(np.linspace(*variables["mll"]["binning"]), name=variables['mll']["name"], label=variables["mll"]["xaxis"]), storage=hist.storage.Weight()), "sumw": 0} for i in others + data_samples + dymumu_samples
        }
    } for region in regions
}

def process_file(path):
    """Worker function: read file, fill histograms, return partial histos."""
    local_histos = copy.deepcopy(histos)  # each process starts with empty copy

    job_results = read_inputs([path])
    for chunk in job_results:
        for dataset in chunk:
            sumw = chunk[dataset]["sumw"]
            for region in regions:
                if region not in chunk[dataset]['events']:
                    continue
                for var in variables:
                    values = chunk[dataset]['events'][region][var]
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


xss = load_xsecs()
print("---> Loading data")

nfiles = 100
nworkers = 50
input_files = [f"../condor/job_{i}/chunks_job.pkl" for i in range(nfiles)]

print("---> Loading data with multiprocessing")
with mp.Pool(processes=nworkers) as pool:  # adjust number of processes
    partial_histos = pool.map(process_file, input_files)

print("---> Merging results")
global_histos = partial_histos[0]
for h in partial_histos[1:]:
    merge_histos(global_histos, h)

# Now use global_histos instead of histos in your plotting code
histos = global_histos
print("---> Histos merged, ready for plotting")

"""
for region in regions: 
    for var in variables:
        print(f"Plotting {region} {var}")

        # --- Build other MC ---
        first = others[0]
        scale = float(xss["samples"][datasets[first]["files"]]["xsec"])
        sumw = histos[region][var][first]["sumw"]
        h_MC_others = renorm(histos[region][var][first]["hist"], scale, sumw, lumi)
        
        for i in others[1:]:
            scale = float(xss["samples"][datasets[i]["files"]]["xsec"])
            sumw = histos[region][var][i]["sumw"]
            if sumw == 0:
                continue
            _h = renorm(histos[region][var][i]["hist"], scale, sumw, lumi)
            h_MC_others += _h
            
        # --- Build DYmumu MC ---
        first = dymumu_samples[0]
        scale = float(xss["samples"][datasets[first]["files"]]["xsec"])
        sumw = histos[region][var][first]["sumw"]
        h_DY = renorm(histos[region][var][first]["hist"], scale, sumw, lumi)
        
        for i in dymumu_samples[1:]:
            scale = float(xss["samples"][datasets[i]["files"]]["xsec"])
            sumw = histos[region][var][i]["sumw"]
            if sumw == 0:
                continue
            _h = renorm(histos[region][var][i]["hist"], scale, sumw, lumi)
            h_DY += _h
            
        # --- Build Data ---
        first = data_samples[0]
        h_Data = histos[region][var][first]["hist"].copy()
        for i in data_samples[1:]:
            _h = histos[region][var][i]["hist"].copy()
            h_Data += _h
            
        # --- Total MC ---
        h_MC = h_MC_others + h_DY
        
        # --- Figure with two panels ---
        fig, (ax, rax) = plt.subplots(
            2, 1,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            figsize=(8, 8),
            sharex=True
        )

        # --- Main panel ---
        h_MC_others.plot(ax=ax, histtype="step", linewidth=2, label="Other MC [{:.1f}]".format(sum(h_MC_others.values())))
        h_DY.plot(ax=ax, histtype="step", linewidth=2, label=r"$Z/\gamma \rightarrow \mu\mu$ [{:.1f}]".format(sum(h_DY.values())))
        h_MC.plot(ax=ax, histtype="step", linewidth=2, label="MC [{:.1f}]".format(sum(h_MC.values())))
        h_Data.plot(ax=ax, histtype="errorbar", color="black", label="Data [{:.1f}]".format(sum(h_Data.values())), markersize=5)

        ax.set_ylabel("Events")
        ax.set_yscale("log")
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
        rax.set_ylim(0.91, 1.09)  # adjust as needed
        #rax.grid(True, linestyle="--", alpha=0.6)

        # --- Save ---
        plt.savefig(f"plots/{region}_{var}.pdf", bbox_inches="tight")
        plt.savefig(f"plots/{region}_{var}.png", bbox_inches="tight")
        plt.close(fig)
    
"""

region = "inc_mm"

# Make Ptll ratio 

def subtract_hists(h1, h2):
    # Copy structure of h1
    h_out = h1.copy()

    # Get views
    v1 = h1.view(flow=False)
    v2 = h2.view(flow=False)
    vout = h_out.view(flow=False)

    # Subtract values
    vout["value"] = v1["value"] - v2["value"]

    # Combine variances (they add in quadrature even for subtraction)
    if "variance" in v1.dtype.names and "variance" in v2.dtype.names:
        vout["variance"] = v1["variance"] + v2["variance"]

    return h_out       
        
# --- Build other MC ---
first = others[0]
scale = float(xss["samples"][datasets[first]["files"]]["xsec"])
sumw = histos[region]["ptll"][first]["sumw"]
h_MC_others = renorm(histos[region]["ptll"][first]["hist"], scale, sumw, lumi)

for i in others[1:]:
    scale = float(xss["samples"][datasets[i]["files"]]["xsec"])
    sumw = histos[region]["ptll"][i]["sumw"]
    if sumw == 0:
        continue
    _h = renorm(histos[region]["ptll"][i]["hist"], scale, sumw, lumi)
    h_MC_others += _h
    
# --- Build DYmumu MC ---
first = dymumu_samples[0]
scale = float(xss["samples"][datasets[first]["files"]]["xsec"])
sumw = histos[region]["ptll"][first]["sumw"]
h_DY = renorm(histos[region]["ptll"][first]["hist"], scale, sumw, lumi)

for i in dymumu_samples[1:]:
    scale = float(xss["samples"][datasets[i]["files"]]["xsec"])
    sumw = histos[region]["ptll"][i]["sumw"]
    if sumw == 0:
        continue
    _h = renorm(histos[region]["ptll"][i]["hist"], scale, sumw, lumi)
    h_DY += _h
    
# --- Build Data ---
first = data_samples[0]
h_Data = histos[region]["ptll"][first]["hist"].copy()
for i in data_samples[1:]:
    _h = histos[region]["ptll"][i]["hist"].copy()
    h_Data += _h
    
# --- Correction MC ---
h_corr = subtract_hists(h_Data, h_MC_others)

# --- Figure with two panels ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# --- Main panel ---

ax.set_ylabel("(Data - Other MC) / DY MC")
ax.set_xlabel(r"Reconstructed $p_{T}^{\ell\ell}$ [GeV]")


# Add CMS label
hep.cms.label(ax=ax, year=2018, lumi="{:.1f}".format(lumi), com=13, data=True)

# --- Ratio plot: Data / MC ---
mc_vals = h_DY.values()
mc_errs = np.sqrt(h_DY.variances()) if h_DY.variances() is not None else np.zeros_like(mc_vals)
N_mc = np.sum(mc_vals)
mc_vals /= N_mc
mc_errs /= N_mc

data_vals = h_corr.values()
data_errs = np.sqrt(h_corr.variances()) if h_corr.variances() is not None else np.zeros_like(data_vals)
N_data = np.sum(data_vals)
data_vals /= N_data
data_errs /= N_data

# Avoid division by zero
mask = mc_vals > 0
ratio = np.ones_like(data_vals) * np.nan
ratio[mask] = data_vals[mask] / mc_vals[mask]

ratio_err = np.zeros_like(ratio)
ratio_err[mask] = data_errs[mask] / mc_vals[mask]

# Bin centers for plotting
bin_edges = h_Data.axes[0].edges
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

ax.errorbar(
    bin_centers, ratio, yerr=ratio_err,
    fmt="o", color="black", markersize=4
)

ax.axhline(1.0, color="red", linestyle="--")

# --- Save ---
plt.savefig(f"plots/{region}_ptll_corr.pdf", bbox_inches="tight")
plt.savefig(f"plots/{region}_ptll_corr.png", bbox_inches="tight")
plt.close(fig)

import uproot

# create histogram with Weighted storage
h_ratio = hist.Hist(
    hist.axis.Variable(bin_edges, name="x"),
    storage=hist.storage.Weight()
)

# fill structured view
view = h_ratio.view(flow=False)
view["value"][:] = ratio
view["variance"][:] = ratio_err**2

# save to ROOT
with uproot.recreate(f"{region}_ptll_corr.root") as fout:
    fout["h_ratio"] = h_ratio
