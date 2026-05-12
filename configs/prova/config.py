# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, interpolate_colors, get_fw_path


fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2018v9"
lumi = lumis[year]["tot"] / 1000  # All of 2018
#lumi = lumis[year]["B"] / 1000
plot_label = "DY"
year_label = "2018"
njobs = 300

runner = f"{fw_path}/src/spritz/runners/runner_3DY_trees_singleTriggers_EFT.py"

special_analysis_cfg = {
    "do_theory_variations": False
}


all_events_mask = "ak.ones_like(events.run) == 1"
rwgt_col = "events.rwgt"

subsamples_eft = {
    "sm": (
        "ak.ones_like(events.run) == 1",
        f"{rwgt_col}[:, 0]",
    ),
}

# create subsamples from DY100_200.json 

import json 
d_100_200 = json.load(open("/gwpool/users/gboldrini/spritz/configs/zmumu_EFT_trees_single_triggers_EFT_startingOne_mod50-100/scripts/DY100_200.json"))
subsamples_eft_100_200 = {}
for key, values in d_100_200.items():
    subsamples_eft_100_200[key] = (
        "ak.ones_like(events.run) == 1",
        f'{rwgt_col}[:, {values["idx"]}]',
    )
    
d_1500_Inf = json.load(open("/gwpool/users/gboldrini/spritz/configs/zmumu_EFT_trees_single_triggers_EFT_startingOne_mod50-100/scripts/DY1500_Inf.json"))
subsamples_eft_1500_Inf = {}
for key, values in d_1500_Inf.items():
    subsamples_eft_1500_Inf[key] = (
        "ak.ones_like(events.run) == 1",
        f'{rwgt_col}[:, {values["idx"]}]',
    )

neft_100_200 = len(subsamples_eft_100_200)
neft_1500_Inf = len(subsamples_eft_1500_Inf)

datasets = {
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne",
        "subsamples": subsamples_eft_100_200,
        "neft": neft_100_200, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne",
        "subsamples": subsamples_eft_100_200,
        "neft": neft_100_200, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
        "subsamples": subsamples_eft_1500_Inf, 
        "neft": neft_1500_Inf, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
        "subsamples": subsamples_eft_1500_Inf,
        "neft": neft_1500_Inf, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
        "subsamples": subsamples_eft_1500_Inf,
        "neft": neft_1500_Inf, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
        "subsamples": subsamples_eft_1500_Inf,
        "neft": neft_1500_Inf, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
        "subsamples": subsamples_eft_100_200,
        "neft": neft_100_200, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
        "subsamples": subsamples_eft_1500_Inf,
        "neft": neft_1500_Inf, # for all events only read the first entry of events.LHEReweightingWeight
        "task_weight": 8,
        "max_chunks": 20
    }
}


for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


samples = {}
colors = {}

# build samples from datasets and reweighting subsamples 
# merge them for convenience

# take only 100_200 because it has the correct operators, 1500_Inf has additional operators not of interest 

for reweighting_weight in subsamples_eft_100_200.keys():
    samples[f"DYMuMu_{reweighting_weight}"] = {
        "samples": [f"{dataset}_{reweighting_weight}" for dataset in datasets.keys()]
    }

"""
samples = {
    "DYMuMu_mll200_400_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne_sm"]
    },
    "DYMuMu_mll400_600_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne_sm"]
    },
    "DYMuMu_mll600_800_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne_sm"]
    },
    "DYMuMu_mll800_1000_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne_sm"]
    },
    "DYMuMu_mll1500_inf_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne_sm"]
    },
    "DYMuMu_mll50_120_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne_sm"]
    },
    "DYMuMu_mll1000_1500_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne_sm"]
    },
    "DYMuMu_mll120_200_sm" : {
        "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne_sm"]
    }
}
"""  

colors = {}
cpalette = interpolate_colors(cmap_petroff, len(samples.keys()))

for idx, i in enumerate(samples.keys()):
    colors[i] = cpalette[idx]

###

print("Samples:", samples)

# regions
preselections = lambda events: (events.mll > 50)  # noqa E731


regions = {
    "inc_ee": {
        "func": lambda events: preselections(events) & events["ee"],
        "mask": 0
    },
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0
    },
    "inc_em": {
        "func": lambda events: preselections(events) & events["em"],
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
    # Dilepton
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(60, 50, 800, name="mll"),
    },
    "costhetastar_bins": {
            "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
            "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins"),
    },
    "yZ_bins": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
            "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
        "axis": hist.axis.Regular(60, 0, 3, name="rapll_abs"),
    },
    "Gen_mll":{
        "func": lambda events: events.Gen_mll,
    },
}

nuisances = {}

# nuisances = {
#     "lumi": {
#         "name": "lumi",
#         "type": "lnN",
#         "samples": dict((skey, "1.02") for skey in samples),
#     },
# }

## Use the following if you want to apply the automatic combine MC stat nuisances.
nuisances["stat"] = {
    "type": "auto",
    "maxPoiss": "10",
    "includeSignal": "0",
    "samples": {},
}

check_weights = {}

