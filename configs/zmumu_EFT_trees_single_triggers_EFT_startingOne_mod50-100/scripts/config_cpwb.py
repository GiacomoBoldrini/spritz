# ruff: noqa: E501

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


datasets = {
    "sm": { "files": "sm", "task_weight": 8, "is_signal": 0},
    "w1_cpwb": { "files": "w1_cpwb", "task_weight": 8 },
    "wm1_cpwb": { "files": "wm1_cpwb", "task_weight": 8 },
}



for dataset in datasets:
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

gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1500, 15000]
costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
yZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)

variables = {
    "triple_diff": {
        "func": lambda events: events.Lepton[:, 0].eta,
        "axis": [hist.axis.Variable(gen_mll_bins, name="mll"), hist.axis.Regular(4, -2.5, 2.5, name="etal1"), hist.axis.Regular(4, 0, 150, name="ptll")]
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
