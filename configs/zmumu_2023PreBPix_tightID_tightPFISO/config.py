# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path


fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2023PreBPix_v12"
#lumi = lumis[year]["tot"] / 1000  # All of 2018
lumi = lumis[year]["C"] / 1000
plot_label = "DY"
year_label = "2023 Pre-BPix"
njobs = 1000

runner = f"{fw_path}/src/spritz/runners/runner_3DY_trees_singleTriggers.py"

special_analysis_cfg = {
    "do_theory_variations": False
}

datasets = {
    # "DYto2L-2Jets_MLL-50": {
    #     "files": "DYto2L-2Jets_MLL-50",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 10000000
    # },
    "TbarWplusto2L2Nu": {
        "files": "TbarWplusto2L2Nu",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "TWminusto2L2Nu": {
        "files": "TWminusto2L2Nu",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "TbarBtoLminusNuB-s-channel": {
        "files": "TbarBtoLminusNuB-s-channel",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "TBbartoLplusNuBbar-s-channel": {
        "files": "TBbartoLplusNuBbar-s-channel",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "TQbartoLNu-t-channel": {
        "files": "TQbartoLNu-t-channel",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "TbarQtoLNu-t-channel": {
        "files": "TbarQtoLNu-t-channel",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "WW": {
        "files": "WW_TuneCP5_13p6TeV-pythia8",
        "task_weight": 8,
        "max_weight": 1e9,
    },
    "WZ": {
        "files": "WZ_TuneCP5_13p6TeV-pythia8",
        "task_weight": 8,
        "max_weight": 1e9,
    },
    "ZZ": {
        "files": "ZZ_TuneCP5_13p6TeV_pythia8",
        "task_weight": 8,
        "max_weight": 1e9,
    },
    # "DYto2L-2Jets_MLL-50_0J_amcatnloFXFX": {
    #     "files": "DYto2L-2Jets_MLL-50_0J_amcatnloFXFX",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    # },
	# "DYto2L-2Jets_MLL-50_1J_amcatnloFXFX": {
    #     "files": "DYto2L-2Jets_MLL-50_1J_amcatnloFXFX",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    # },
	# "DYto2L-2Jets_MLL-50_2J_amcatnloFXFX": {
    #     "files": "DYto2L-2Jets_MLL-50_2J_amcatnloFXFX",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    # },
    "DYto2Mu-MiNNLO-Photos": {
        "files": "DYto2Mu-MiNNLO-Photos",
        "task_weight": 8,
        "max_weight": 1e9,
    },
    "DYto2E_MLL-50to120": {
        "files": "DYto2E_MLL-50to120",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-120to200": {
        "files": "DYto2E_MLL-120to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-200to400": {
        "files": "DYto2E_MLL-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-400to800": {
        "files": "DYto2E_MLL-400to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-800to1500": {
        "files": "DYto2E_MLL-800to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-1500to2500": {
        "files": "DYto2E_MLL-1500to2500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-2500to4000": {
        "files": "DYto2E_MLL-2500to4000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-4000to6000": {
        "files": "DYto2E_MLL-4000to6000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2E_MLL-6000": {
        "files": "DYto2E_MLL-6000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    # "DYto2Mu_MLL-50to120": {
    #     "files": "DYto2Mu_MLL-50to120",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-120to200": {
    #     "files": "DYto2Mu_MLL-120to200",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-200to400": {
    #     "files": "DYto2Mu_MLL-200to400",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-400to800": {
    #     "files": "DYto2Mu_MLL-400to800",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-800to1500": {
    #     "files": "DYto2Mu_MLL-800to1500",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-1500to2500": {
    #     "files": "DYto2Mu_MLL-1500to2500",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-2500to4000": {
    #     "files": "DYto2Mu_MLL-2500to4000",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-4000to6000": {
    #     "files": "DYto2Mu_MLL-4000to6000",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    # "DYto2Mu_MLL-6000": {
    #     "files": "DYto2Mu_MLL-6000",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 1000
    # },
    "DYto2Tau_MLL-50to120": {
        "files": "DYto2Tau_MLL-50to120",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-120to200": {
        "files": "DYto2Tau_MLL-120to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-200to400": {
        "files": "DYto2Tau_MLL-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-400to800": {
        "files": "DYto2Tau_MLL-400to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-800to1500": {
        "files": "DYto2Tau_MLL-800to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-1500to2500": {
        "files": "DYto2Tau_MLL-1500to2500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-2500to4000": {
        "files": "DYto2Tau_MLL-2500to4000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-4000to6000": {
        "files": "DYto2Tau_MLL-4000to6000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },
    "DYto2Tau_MLL-6000": {
        "files": "DYto2Tau_MLL-6000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 1000
    },  
    # "TTJets": {
    #     "files": "TTJets",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 10000000
    # },  
    "TTTo2L2Nu": {
        "files": "TTTo2L2Nu",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 10000000
    },  
    # "TTToSemiLeptonic": {
    #     "files": "TTToSemiLeptonic",
    #     "task_weight": 8,
    #     "max_weight": 1e9,
    #     "max_chunks": 10000000
    # }

}

# datasets = {
# 	"DYto2Mu_MLL-50to120": {
#         "files": "DYto2Mu_MLL-50to120",
#         "task_weight": 8,
#         "max_weight": 1e9,
#         "max_chunks": 1000
#     }
# }

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# DataSets = ["MuonEG", "Muon0", "Muon1", "EGamma0", "EGamma1"]
DataSets = [ "Muon0", "Muon1" ]

DataTrig = {
    "Muon0": "events.SingleMu",
    "Muon1": "events.SingleMu",
}

DataRun = [
    ["C", "Run2023C", ["1", "2", "3", "4"]],
]

samples_data = []
for era, sd, keys in DataRun:
    for pd in DataSets:
        for k in keys:
            tag = pd + "_" + sd + "_" + k

            datasets[f"{pd}_{era}_{k}"] = {
                "files": tag,
                "trigger_sel": DataTrig[pd],
                "read_form": "data",
                "is_data": True,
                "era": f"2023{era}",
            }
            samples_data.append(f"{pd}_{era}_{k}")

samples = {}
colors = {}


samples = {
    "Data": {
        "samples": samples_data,
        "is_data": True,
    },
    "ST": {
        "samples": [
            "TbarWplusto2L2Nu",
            "TWminusto2L2Nu",
            "TbarBtoLminusNuB-s-channel",
            "TBbartoLplusNuBbar-s-channel",
            "TQbartoLNu-t-channel",
            "TbarQtoLNu-t-channel"
        ]
    },
    "TT": {
        "samples": [
            "TTTo2L2Nu",
            # "TTToSemiLeptonic",
            # "TTJets"
        ]
    },
    "VV": {
       "samples": [
            "WW",
            "WZ",
            "ZZ"
       ]
    },
    "DYtt": {
       "samples": [
           "DYto2Tau_MLL-50to120",
           "DYto2Tau_MLL-120to200",
           "DYto2Tau_MLL-200to400",
           "DYto2Tau_MLL-400to800",
           "DYto2Tau_MLL-800to1500",
           "DYto2Tau_MLL-1500to2500",
           "DYto2Tau_MLL-2500to4000",
           "DYto2Tau_MLL-4000to6000",
           "DYto2Tau_MLL-6000"
       ]
    },
    "DY2Mu-MiNNLO": {
        "samples": ["DYto2Mu-MiNNLO-Photos"]
    },
    # "DYmumu-120-200": {
    #     "samples": ["DYto2Mu_MLL-120to200"]
    # },
    # "DYll_M50":{
    #     "samples": ["DYto2L-2Jets_MLL-50"]
    # },
    # "DYM50-MGNLO": {
    #     "samples": ["DYto2Mu-2Jets_Bin-MLL-50_amcatnloFXFX"],
    #     "is_signal": True
    # },
    # "DYM50-MGLO": {
    #     "samples": ["DYto2Mu-4Jets_Bin-MLL-50_madgraphMLM"],
    #     "is_signal": True
    # },
    # "DYll": {
    #    "samples": [
    #        "DYto2E_MLL-50to120",
    #        "DYto2E_MLL-120to200",
    #        "DYto2E_MLL-200to400",
    #        "DYto2E_MLL-400to800",
    #        "DYto2E_MLL-800to1500",
    #        "DYto2E_MLL-1500to2500",
    #        "DYto2E_MLL-2500to4000",
    #        "DYto2E_MLL-4000to6000",
    #        "DYto2E_MLL-6000",
    #        "DYto2Mu_MLL-200to400",
    #        "DYto2Mu_MLL-400to800",
    #        "DYto2Mu_MLL-800to1500",
    #        "DYto2Mu_MLL-1500to2500",
    #        "DYto2Mu_MLL-2500to4000",
    #        "DYto2Mu_MLL-4000to6000",
    #        "DYto2Mu_MLL-6000"
    #    ]
    # }
    "DYee": {
        "samples": [
           "DYto2E_MLL-50to120",
           "DYto2E_MLL-120to200",
           "DYto2E_MLL-200to400",
           "DYto2E_MLL-400to800",
           "DYto2E_MLL-800to1500",
           "DYto2E_MLL-1500to2500",
           "DYto2E_MLL-2500to4000",
           "DYto2E_MLL-4000to6000",
           "DYto2E_MLL-6000"
        ]
    },
    # "DY0J": {
    #     "samples": [
    #             "DYto2L-0Jets_MLL-50_0J_amcatnloFXFX",
    #         ]
    # },
	# "DY1J": {
    #     "samples": [
    #             "DYto2L-1Jets_MLL-50_0J_amcatnloFXFX",
    #         ]
    # },
	# "DY2J": {
    #     "samples": [
    #             "DYto2L-2Jets_MLL-50_0J_amcatnloFXFX",
    #         ]
    # },
}

# samples = {
# 	"DYmumu-50-120": {
#         "samples": ["DYto2Mu_MLL-50to120"]
#     }
# }

colors = {}
colors["ST"] = cmap_pastel[2]
colors["TT"] = cmap_pastel[3]
colors["VV"] = cmap_pastel[4]
colors["DYtt"] = cmap_pastel[5]
colors["DY2Mu-MiNNLO"] = cmap_pastel[1]
colors["DYee"] = cmap_pastel[6]

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
	"mll_restricted": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(150, 80, 105, name="mll_restricted"),
        "save_events": False
    },
	"mll_medium": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable([50,58,64,72,78,84,90,96,102,108,116,124,132,140,
            148,156,164,172,180,190,200,210,220,230,240,255,270,285,300,325,350,375,
            400,450,500], name="mll_medium")
    },
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(150, 50, 200, name="mll"),
        "save_events": False
    },
    "mll_uncorr": {
        "func": lambda events: (events.Lepton_uncorr[:, 0] + events.Lepton_uncorr[:, 1]).mass,
        "axis": hist.axis.Regular(150, 50, 200, name="mll_uncorr"),
        "save_events": False
    },
    "costhetastar_bins": {
            "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
            "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins"),
            "save_events": False
    },
    "yZ_bins": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
            "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
            "save_events": False
    },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Regular(60, 0, 600, name="ptll"),
        "save_events": False
    },
    "etall": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
        "axis": hist.axis.Regular(80, -8, 8, name="etall"),
        "save_events": False
    },
    "rapll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="rapll"),
        "save_events": False
    },
    "detall": {
        "func": lambda events: abs(events.Lepton[:, 0].deltaeta(events.Lepton[:, 1])),
        "axis": hist.axis.Regular(50, 0, 5, name="detall"),
        "save_events": False
    },
    "dphill": {
        "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
        "axis": hist.axis.Regular(63, 0, 3.15, name="dphill"),
        "save_events": False
    },
    "dRll": {
        "func": lambda events: events.Lepton[:, 0].deltaR(events.Lepton[:, 1]),
        "axis": hist.axis.Regular(60, 0, 2, name="dRll"),
        "save_events": False
    },
    "ptl1": {
        "func": lambda events: events.Lepton[:, 0].pt,
        "axis": hist.axis.Regular(60, 20, 320, name="ptl1"),
        "save_events": False
    },
    "ptl2": {
        "func": lambda events: events.Lepton[:, 1].pt,
        "axis": hist.axis.Regular(60, 10, 160, name="ptl2"),
        "save_events": False
    },
    "etal1": {
        "func": lambda events: events.Lepton[:, 0].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal1"),
        "save_events": False
    },
    "etal2": {
        "func": lambda events: events.Lepton[:, 1].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2"),
        "save_events": False
    },
    "etal2": {
        "func": lambda events: events.Lepton[:, 1].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2"),
        "save_events": False
    },
    "GenPtLL": {
        "func": lambda events: events.Gen_ptll, 
        "axis": hist.axis.Regular(50, 0, 500, name="GenPtLL"),
        "save_events": False
    },
	"mll_etall_bins": {
        "func": lambda events: events.Lepton[:, 0].eta,
        "axis": [hist.axis.Regular(20, 50, 120, name="mll"), hist.axis.Regular(10, -8, 8, name="etall")]
    },
}

nuisances = {}

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

check_weights = {}
