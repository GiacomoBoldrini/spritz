import os, sys
import pickle
import numpy as np
import hist
import argparse
import uproot
from hist import Hist
from itertools import combinations
from tqdm import tqdm 


# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    parser.add_argument("-o", "--output", default="datacards", help="output directory")
    parser.add_argument("-f", "--file", type=str, required=True, help="Pickled file ")
    parser.add_argument("-v", "--variable", type=str, required=False, help="Variable to process", default="")
    parser.add_argument("-op", "--operator", type=str, required=True, help="Build histos.root for this operators, comma separated")
    parser.add_argument("--keep-unc", required=False, help="Keep MC stat unc on all templates", default=False, action="store_true")
    parser.add_argument("--matrix", required=False, help="Store 3D matrix for correlated MC stat unc", default=False, action="store_true")
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

def get_shape_name(names):
    shape1, shape2 = names.split("x") 
    #print(shape1, shape2 )
    ops = []
    for sh__ in [shape1, shape2]:
        if sh__.startswith("w1_"): ops.append(sh__.split("w1_")[1])
        elif sh__.startswith("wm1_"): ops.append(sh__.split("wm1_")[1] + "_m1")
        elif sh__.startswith("w11_"): ops.append(sh__.split("w11_")[1])
        elif sh__ == "sm": ops.append(sh__)

    #print(ops)
    # diagonal
    if shape1 == shape2:
        if shape1 == "sm": 
            return "sm_variance"
        elif shape1.startswith("w1_"): 
            op__ = shape1.split("w1_")[1]
            return f"{op__}_plus_variance"
        elif shape1.startswith("wm1_"):
            op__ = shape1.split("wm1_")[1]
            return f"{op__}_minus_variance"
        elif shape1.startswith("w11_"): 
            op1_op2 = shape1.split("w11_")[1]
            return f"{op1_op2}_variance"
        
    # off diagonal 
    return ops, "_mixed_variance"

def create_config(samples, variables):
    l = """# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path


fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2018v9"
lumi = lumis[year]["tot"] / 1000  # All of 2018
#lumi = lumis[year]["B"] / 1000
plot_label = "DY"
year_label = "2018"
njobs = 1000

runner = f"{fw_path}/src/spritz/runners/runner_3DY_trees_singleTriggers.py"

special_analysis_cfg = {
    "do_theory_variations": False
}

"""
    l += "\n\ndatasets = {\n"
    for s in samples:
        l += f'       "{s}": {{ "files": "{s}", "task_weight": 8, "is_signal": 0}},\n'
    l += "}\n\n"
    l += "variables = {\n"
    for v in variables:
        l += f'    "{v}":' +  '{"func": lambda events: events.Lepton[:, 0].eta, "axis": [hist.axis.Regular(4, 0, 150, name="ptll")]},\n'
    l += "}\n"
    
    l += """for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"



samples = {}
colors = {}

samples = {
	i : {
        "samples": [i],
        "is_signal": 0
    } for i in datasets.keys()
}

colors = {}

for i in datasets.keys():
    found=False
    for merged in samples.keys():
        if i in samples[merged]["samples"]:
            found=True
            break
    
    if not found:
        samples[i] = {
                "samples": [i]
                }
        colors[i] = cmap_petroff[3]

###

print("Samples:", samples)

# regions
preselections = lambda events: (events.mll > 50)  # noqa E731


regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0
    }
}

nuisances = {}

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances["stat"] = {
    "type": "auto",
    "maxPoiss": "10",
    "includeSignal": "0",
    "samples": {},
}
nuisances["lumi"] = {
    "name": "lumi",
    "type": "lnN",
    "samples": dict((skey, "1.02") for skey in samples),
}

check_weights = {}

"""
    
    
    return l  
# -----------------------------
# Main
# -----------------------------

def main():
    args = get_args()
    
    # read your pickled/file data
    f = read_file(args.file)
    
    requested_ops = [[op.strip()] for op in args.operator.split(",")]
    requested_ops += [a + b for a, b in list(combinations(requested_ops, 2))]
    # add profiled case
    requested_ops += list(combinations(args.operator.split(","), len(args.operator.split(","))))

    
    
    # need to create 1D and 2D histograms for each region/variable/nominal/histo_label
    
    for req_ops in requested_ops:
        
        # create output directory 
        print(req_ops)
        outfolder = os.path.join(args.output, "_".join(req_ops))
        mkdir(outfolder)
        
        
        dout = {}

        for region in f.keys():
            operators = [i.split("_m1")[0] for i in f[region].keys() if i.endswith("_m1")]
            operators = [op for op in operators if op in req_ops]
            
            op_pairs = list(combinations(operators, 2))
            variables = list(f[region][operators[0]].keys())
            variables = [var for var in variables if var in args.variable.split(",") or args.variable == ""]
            print(variables)
            for var in variables: 
                
                print(f"-----> var {var}") 
                histo_labels = []
                
                h_sm = f[region]["sm"][var]["all"]["histo"]
                if not args.keep_unc: h_sm.view().variance = (np.zeros_like(h_sm.variances()))
                histo_labels += [[h_sm, "sm"]]
                for operator in operators:
                    h_p1 = f[region][operator][var]["all"]["histo"]
                    h_m1 = f[region][operator + "_m1"][var]["all"]["histo"]
                    if not args.keep_unc:
                        h_p1.view().variance = (np.zeros_like(h_p1.variances()))
                        h_m1.view().variance = (np.zeros_like(h_m1.variances()))
                        
                    histo_labels += [[h_p1, f"w1_{operator}"], [h_m1, f"wm1_{operator}"]] 

                # do op pairs 
                for op1, op2 in op_pairs:
                    h_11 = f[region][f"{op1}_{op2}"][var]["all"]["histo"]
                    if not args.keep_unc:
                        # remove MC stat unc
                        h_11.view().variance = (np.zeros_like(h_11.variances()))
                        
                    histo_labels += [[h_11, f"w11_{op1}_{op2}"]] 
            
                # Write nominal histos in output dictionary 
                
                for hl__ in histo_labels:
                    histo, histoName = hl__
                    #print(type(histo), type(histoName))
                    key = f"{region}/{var}/nominal/histo_{histoName}"
                    if key not in dout:
                        dout[key] = histo.copy()
                    else:
                        dout[key] += histo.copy()
                
                categories = [i[1] for i in histo_labels] # ["sm", "w1_cw", "wm1_cw", ..., "w11_cw_chl1"]
                
                # Now build 3D for MC stat uncertainties       
                if args.matrix:
                    
                    corr_pairs = []
                    # build the histo matrix     
                    
                    # matrix form 
                    labels = np.array([[f"{a}x{b}" for b in categories] for a in categories])
                    
                    #sys.exit(0)
                    # cycle on variables 
                    
                    # for var in variables:
                        
                    hist_matrix = {}
                    
                    for i in range(0, labels.shape[0]):
                        for j in range(i, labels.shape[1]):
                            sn__ = get_shape_name(labels[i,j])
                            if isinstance(sn__, str):
                                hist_matrix[(i,j)] = f[region][sn__][var]["all"]["histo"]
                            else:
                                perm_name = "_".join(sn__[0]) + sn__[1]
                                if perm_name not in f[region].keys():
                                    perm_name = "_".join(sn__[0][::-1]) + sn__[1]
                                    
                                #print(len(f[region].keys()))
                                hist_matrix[(i,j)] = f[region][perm_name][var]["all"]["histo"]

                    # Determine X (numerical) binning from one of the histograms
                    x_edges = hist_matrix[(0,0)].axes[0].edges
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

                    # Create 3D histogram: numeric X, categorical Y and Z
                    histo_name = f"histo_correlation"
                    h3 = Hist.new.Reg(len(x_edges)-1, x_edges[0], x_edges[-1], name="x") \
                                .StrCategory(categories, name="y") \
                                .StrCategory(categories, name="z") \
                                .Double()

                    # Fill the 3D histogram
                    for (i, j), h in tqdm(hist_matrix.items(), desc=f"Filling 3D histogram {var}", unit="pair"):
                        y_label = categories[i]
                        z_label = categories[j]

                        # Convert the 1D hist.Hist to values and edges
                        x_values = h.values().flatten()
                        x_bin_edges = h.axes[0].edges

                        # Fill using X-bin centers
                        for x_idx, weight in enumerate(x_values):
                            x_center = 0.5 * (x_bin_edges[x_idx] + x_bin_edges[x_idx + 1])
                            h3.fill(x=x_center, y=y_label, z=z_label, weight=weight)

                    # Store in output dict
                    key = f"{region}/{var}/nominal/{histo_name}"
                    if key not in dout:
                        dout[key] = h3.copy()
                    else:
                        print(f"ERROR: CORRELATION MATRIX ALREADY IN OUTPUT DICT! {region}/{var}/nominal/{histo_name}")
                        sys.exit(0)
                
        # Here we create the config 
        config = create_config(categories, variables)
        with open(os.path.join(outfolder, "config.py"), "w") as cfgout:
            cfgout.write(config)
           
        # write to ROOT
        with uproot.recreate(os.path.join(outfolder, "histos.root")) as fout:
            for key, histo in dout.items():
                print(key)
                fout[key] = histo

if __name__ == "__main__":
    main()

