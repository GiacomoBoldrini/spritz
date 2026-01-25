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
    parser.add_argument("-o", "--output", default="histos.root", help="output file histos.root")
    parser.add_argument("-f", "--file", type=str, required=True, help="Pickled file ")
    parser.add_argument("-op", "--operator", type=str, required=False, help="Build histos.root for this operators, comma separated", default="")
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
        
# -----------------------------
# Main
# -----------------------------

def main():
    args = get_args()
    
    # read your pickled/file data
    f = read_file(args.file)
    
    dout = {}

    for region in f.keys():
        operators = [i.split("_m1")[0] for i in f[region].keys() if i.endswith("_m1")]
        if args.operator:
            requested_ops = [op.strip() for op in args.operator.split(",")]
            operators = [op for op in operators if op in requested_ops]
        
        op_pairs = list(combinations(operators, 2))
        variables = list(f[region][operators[0]].keys())
        
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
            
            # Now build 3D for MC stat uncertainties       
            if args.matrix:
                
                corr_pairs = []
                # build the histo matrix 
                categories = [i[1] for i in histo_labels] # ["sm", "w1_cw", "wm1_cw", ..., "w11_cw_chl1"]                
                
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
            
            
    
    # write to ROOT
    with uproot.recreate(args.output) as fout:
        for key, histo in dout.items():
            print(key)
            fout[key] = histo

if __name__ == "__main__":
    main()
