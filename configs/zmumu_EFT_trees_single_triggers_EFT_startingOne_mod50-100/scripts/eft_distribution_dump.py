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

def process_file(file_path, regions, samples_to_process, variables, reweight_map):
    print(file_path)
    job_results = read_inputs([file_path])

    # --- Precompute things used multiple times ---
    non_sm_ops = [op for op in reweight_map if op != "sm"]
    op_pairs = list(combinations(non_sm_ops, 2))

    # --- Pre-build histogram templates once ---
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

    # --- Process events ---
    for idx, chunk in enumerate(tqdm(job_results, desc=f"{file_path}", leave=False)):
    #for idx, chunk in enumerate(job_results):
        #print(f"Processing chunk {idx+1}/{len(job_results)}")

        for dataset, dset_data in chunk.items():
            #print(dataset, dataset not in samples_to_process)
            if dataset not in samples_to_process:
                continue

            sumw = dset_data['sumw']
            if sumw == 0:
                print(f"File path: {file_path} Dataset: {dataset}, sumw: {sumw}")

            for region in regions:
                #print(region)
                #print(dset_data['events'].keys())
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

                # cache variable values once
                var_cache = {
                    var: region_data[var] for var in variables if var in region_data
                }

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

                    w_mix = region_data[w_name]
                    w_op1 = region_data[op1]
                    w_op2 = region_data[op2]
                    w_mix_only = w_mix + w_sm - w_op1 - w_op2

                    for weight, label in zip([w_mix, w_mix_only],
                                             [f"{op1}_{op2}", f"{op1}_{op2}_mix"]):
                        total_weight = base_weights * weight
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

    return local_histos



# -----------------------------
# Merge histograms
# -----------------------------
def merge_histos(h1, h2):
    for region in h1.keys():
        for operator in h1[region].keys():
            for var in h1[region][operator].keys():
                for sample in h1[region][operator][var].keys():
                    # unroll if needed... This operation should be done only on h2 but ok...
                    h1[region][operator][var][sample]["histo"] = hist_unroll(h1[region][operator][var][sample]["histo"])
                    h2[region][operator][var][sample]["histo"] = hist_unroll(h2[region][operator][var][sample]["histo"])
                    
                    #now merge them 
                    h1[region][operator][var][sample]["histo"] += h2[region][operator][var][sample]["histo"]
                    h1[region][operator][var][sample]["sumw"] += h2[region][operator][var][sample]["sumw"]
    return h1
    
def scale_samples(histos, lumi):
    for region in histos.keys():
        for operator in histos[region].keys():
            for var in histos[region][operator].keys():
                for sample in histos[region][operator][var].keys():
                    xs = float(cross_sectins[sample]["xsec"])
                    sumw = histos[region][operator][var][sample]["sumw"]
                    scale = renorm(xs, sumw, lumi)
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
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_100_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll100_200_Photos_startingOne",
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
    ]

    gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1400, 15000]
    gen_mll_optimized = [50, 92, 94, 96, 98, 106, 112, 160, 265, 275, 295, 315, 430, 440, 450, 460, 480, 490, 500, 510, 520, 540, 560, 580, 600, 636, 672, 708, 744, 800, 908, 3000]
    mll_medium_bins = [50,58,64,72,78,84,90,96,102,108,116,124,132,140,
                148,156,164,172,180,190,200,210,220,230,240,255,270,285,300,325,350,375,
                400,450,500]
    costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    etaZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]

    variables = {
        "mll": {"binning": (50, 3000, 150), "xaxis": r"$m_{\ell\ell}$ [GeV]"},
        "costhetastar_bins": {"binning": (-1, 1, 50), "xaxis": r"$cos \theta*$ [a.u.]"},
        "yZ_bins": {"binning": (-5, 5, 50), "xaxis": r"$y_{\ell\ell}$ [a.u.]"},
        "triple_diff": {"axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_medium": {"axis": [hist.axis.Variable(mll_medium_bins, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
        "triple_diff_optimized": {"axis": [hist.axis.Variable(gen_mll_optimized, name="mll"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(etaZ_bins, name="yZ_bins")], "xaxis": r"Triple diff bin"},
    }

    # Load LHE reweight JSON
    with open(args.lhe_json) as f:
        reweight_map = json.load(f)
        
    # reweight_map = [i.split("_m1")[0] for i in reweight_map.keys() if "_m1" in i] + ["sm"]

    reweight_map = args.reweighting.split(",")
    # List input files
    input_files = glob(args.input_dir + "/*/chunks_job.pkl")[:args.max_files]
    
    

    # Multiprocessing
    with mp.Pool(processes=args.nworkers) as pool:
        func = partial(process_file, regions=regions__, samples_to_process=samples_to_process,
                       variables=variables, reweight_map=reweight_map)
        print(func, input_files)
        partial_histos = pool.map(func, input_files)
    
    print("Done processing files, now merging histograms...")
    # Merge histograms
    global_histos = partial_histos[0]
    for h in partial_histos[1:]:
        merge_histos(global_histos, h)

    # scale samples    
    print("Now scaling histograms...")
    global_histos = scale_samples(global_histos, lumi if args.luminosity == None else args.luminosity)
    # Create overall 
    print("Now merging samples...")
    global_histos = merge_samples(global_histos)
    
    
    
    # Save merged histograms
    output_file = os.path.join(args.output, "histos_merged.pkl")
    with open(output_file, "wb") as f:
        print(global_histos["inc_mm"].keys())
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
