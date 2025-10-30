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
from config import datasets, regions, lumi, samples
from spritz.framework.framework import read_chunks
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy.optimize import curve_fit
import random

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
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file(path):
    with open(path, "rb") as f:
        compressed = f.read()
        decompressed = zlib.decompress(compressed)
        data = pickle.loads(decompressed)
        return data
    
    
def read_inputs_skimmed(inputs, regions, samples):
    var = { i: {
            "mll": [],
            "gen_mll": [],
            "weight": []
        } for i in samples
    }
    
    for input_file in inputs:
        print(f"Reading file: {input_file}")
        job_result = read_chunks(input_file)
    
        for chunk in job_result:
            chunk = chunk["result"]["real_results"]
            for dataset, dset_data in chunk.items():
                if dataset not in samples:
                    # print(f"Dataset {dataset} not in {samples}")
                    continue
                for region in regions:
                    region_data = dset_data['events'].get(region)
                    #print(region_data.keys())
                    if region_data is None:
                        print(f"Region {region} not found in dataset {dataset}")
                        continue
                    
                    #print("sm" in region_data.keys())
                    base_weights = region_data["weight"]
                    if "sm" not in region_data:
                        print(f"Warning: 'sm' weight not found in events for dataset {dataset} region {region} file {file_path}")
                        continue
                    w_sm = region_data["sm"]
                    
                    var[dataset]["mll"] += region_data["mll"].tolist()
                    var[dataset]["gen_mll"] += region_data["Gen_mll"].tolist()
                    var[dataset]["weight"] += (base_weights * w_sm).tolist()
                        

    return var

#  -----------------------------
# Cross sections
#  -----------------------------

def renorm(xs, sumw, lumi):
    scale = xs * 1000 * lumi / sumw
    # print(scale)
    return scale

cross_sectins = {
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_50_100_Photos/251014_153049",
      "xsec": "1909.9894816",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_100_200_Photos/251014_153054",
      "xsec": "172.69619",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_200_400_Photos/251014_102527",
      "xsec": "2.9751425472",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_400_600_Photos/251014_102534",
      "xsec": "0.19447485764",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_600_800_Photos/251014_153101",
      "xsec": "0.047187595244",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_800_1000_Photos/251014_153107",
      "xsec": "0.010173648348",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1000_1500_Photos/250904_134823",
      "xsec": "0.0071970617",
      "kfact": "1.000",
      "ref": "A1"
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne":{
      "path": "root://eos.grif.fr:1094//eos/grif/cms/llr//store/user/gboldrin/3DY_SMEFTsim_NLO/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/ZDYEFT-nanoaod18_SMEFTatNLO_mll_1500_inf_Photos/251013_080210",
      "xsec": "0.000870364354627",
      "kfact": "1.000",
      "ref": "A1"
    },
}

    
# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    # mkdir(args.output)


    # DY samples and regions
    # regions__ = list(regions.keys())
    regions__ = ["inc_mm"]
    
    samples_to_process = [
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
    ]

    binning_options = {
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne": [(50, 100, 51), (-3,3)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne": [(200,400, 201), (-10,10)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne": [(400, 600, 201), (-25,25)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne": [(600, 800, 201), (-40,40)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne": [(800, 1000, 201), (-60,60)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne": [(1500, 2000, 501), (-100,100)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne": [(100, 200, 101), (-7,7)],
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne": [(1000, 1500, 501), (-100,100)],
    }
    # gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1400, 15000]
    # mll_medium_bins = [50,58,64,72,78,84,90,96,102,108,116,124,132,140,
    #             148,156,164,172,180,190,200,210,220,230,240,255,270,285,300,325,350,375,
    #             400,450,500]
    # costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    # etaZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0] 
    # variables = {
    #     "mll": {"binning": (50, 3000, 150), "xaxis": r"$m_{\ell\ell}$ [GeV]"},
    #     "costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]"},
    #     "yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]"},
    #     "triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
    #     "triple_diff_medium": {"axis": [hist.axis.Variable(mll_medium_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
    # }
    
    file_path = [args.input_dir + f"/job_{i}/chunks_job.pkl" for i in [random.randint(0, 200) for _ in range(0, 50)]]
    print(file_path)
    #job_results = read_inputs(file_path)
    var__ = read_inputs_skimmed(file_path, regions__, samples_to_process)
    
    #sys.exit(0)
            
    for dataset in samples_to_process:
        var = var__[dataset]
        if len(var["mll"]) == 0:
            print(f"No events for dataset {dataset}, skipping.")
            continue
        
        print(dataset, binning_options[dataset])
        # Define 2D bins
        x_bins = np.linspace(*binning_options[dataset][0])  # e.g., mll
        y_bins = np.linspace(*binning_options[dataset][0])  # gen_mll

        # Compute 2D weighted histogram
        H, x_edges, y_edges = np.histogram2d(
            var["mll"], var["gen_mll"], 
            bins=[x_bins, y_bins], 
            weights=var["weight"]
        )

        # Plot as heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        # Use pcolormesh: x_edges and y_edges define bin edges
        pcm = ax.pcolormesh(x_edges, y_edges, H.T, cmap="viridis", norm=mcolors.LogNorm(vmin=H[H>0].min(), vmax=H.max()))  # transpose for correct orientation
        fig.colorbar(pcm, ax=ax, label="Weighted events")
        ax.set_xlabel(r"$m_{\ell\ell}^{Reco}$ [GeV]")
        ax.set_ylabel(r"$m_{\ell\ell}^{Gen}$ [GeV]")
        #ax.set_title(f"{dataset}")
        ax.set_title("")
        fig.savefig(f"2D_weighted_heatmap_{dataset}.png", bbox_inches="tight")
        fig.savefig(f"2D_weighted_heatmap_{dataset}.pdf", bbox_inches="tight")
        plt.close(fig)
        
        
        # difference
        diff = [var["mll"][i] - var["gen_mll"][i] for i in range(len(var["mll"]))]
        
        
        # --- 1. Histogram ---
        bins = 100  # adjust number of bins
        hist_vals, bin_edges = np.histogram(diff, bins=bins, density=True, range=binning_options[dataset][1])
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # --- 2. Define Gaussian function ---
        def gaussian(x, mu, sigma, A):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        # --- 3. Initial guesses ---
        A0 = hist_vals.max()
        mu0 = np.mean(diff)
        sigma0 = np.std(diff)

        p0 = [mu0, sigma0, A0]

        # --- 4. Fit Gaussian ---
        popt, pcov = curve_fit(gaussian, bin_centers, hist_vals, p0=p0)
        mu_fit, sigma_fit, A_fit = popt

        # --- 5. Plot ---
        fig = plt.figure(figsize=(7,5))
        plt.hist(diff, bins=bins, density=True, alpha=0.6, color='g', label='Data', range=binning_options[dataset][1])
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label=f'Gaussian fit\n$\mu$={mu_fit:.2f}, $\sigma$={sigma_fit:.2f}')
        plt.xlabel(r'$m_{\ell\ell}^{Reco} - m_{\ell\ell}^{Gen}$ [GeV]')
        plt.ylabel('Normalized counts')
        plt.legend()
        plt.xlim(*binning_options[dataset][1])
        fig.savefig(f"resolution_{dataset}.png", bbox_inches="tight")
        fig.savefig(f"resolution_{dataset}.pdf", bbox_inches="tight")
        plt.close(fig)

        print(f"Fitted Gaussian width (sigma): {sigma_fit:.3f}")
    



if __name__ == "__main__":
    main()
