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
lumi = lumis[year]["B"] / 1000  # ERA C of 2017
# lumi = lumis[year]["tot"] / 1000  # All of 2017
plot_label = "ZmumuEFT"
year_label = "2018B"
njobs = 200

runner = f"{fw_path}/src/spritz/runners/runner_test.py"

special_analysis_cfg = {
    "do_theory_variations": False
}

bins = {
    "ptll": np.linspace(0, 200, 5),
    "mll": np.linspace(60, 200, 50)
}


datasets = {}

datasets["DYmm"] = {
    "files": "DYJetsToMuMu_M-50",
    "task_weight": 8,
    "max_chunks": 100,
    "read_form": "mc"
}

datasets["DYee"] = {
    "files": "DYJetsToEE_M-50",
    "task_weight": 8,
    "max_chunks": 100,
    "read_form": "mc"
}




DataTrig = {
    "DoubleMuon": "events.DoubleMu",
    "SingleMuon": "(~events.DoubleMu) & events.SingleMu",
    "EGamma": "(~events.DoubleMu) & (~events.SingleMu) & (events.SingleEle | events.DoubleEle)",
}

samples = {}
colors = {}

#####
samples["DYee"] = {
    "samples": ["DYee"],
}
colors["DYee"] = cmap_pastel[0]
#####
samples["DYmm"] = {
    "samples": ["DYmm"],
}
colors["DYmm"] = cmap_pastel[1]
#####




# regions
preselections = lambda events: (events.mll > 60) & (events.mll < 180) # noqa E731

regions = {}

regions["inc_ee"] = {
    "func": lambda events: events["ee"],
    "mask": 0,
}

regions["inc_mm"] = {
    "func": lambda events: events["mm"],
    "mask": 0,
}

variables = {}

variables["genWeight"] = {
    "func": lambda events: events.weight,
    "axis": hist.axis.Regular(100, 0, 10000, name="genWeight"),
    "unweighted": True
}



nuisances = {
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": dict((skey, "1.02") for skey in samples),
    },
}

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances["stat"] = {
    "type": "auto",
    "maxPoiss": "10",
    "includeSignal": "0",
    "samples": {},
}
