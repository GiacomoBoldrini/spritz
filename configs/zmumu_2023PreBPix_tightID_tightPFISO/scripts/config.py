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
njobs = 100

runner = f"{fw_path}/src/spritz/runners/runner_3DY_trees_singleTriggers.py"

special_analysis_cfg = {
    "do_theory_variations": False
}


datasets = {
    # "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos",
    #     "task_weight": 8,
    # },
    # "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos": {
    #     "files": "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos",
    #     "task_weight": 8,
    # },
    # 
    "DYtt_M-50": {
        "files": "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYee_M-50": {
        "files": "DYJetsToEE_M-50",
        "task_weight": 8,
        "max_weight": 1e9,
    },
    "DYJetsToEE_M-100to20": {
        "files": "DYJetsToEE_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-200to400": {
        "files": "DYJetsToEE_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-400to500": {
        "files": "DYJetsToEE_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-500to700": {
        "files": "DYJetsToEE_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-700to800": {
        "files": "DYJetsToEE_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-800to1000": {
        "files": "DYJetsToEE_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-1000to1500": {
        "files": "DYJetsToEE_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-1500to2000": {
        "files": "DYJetsToEE_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToEE_M-2000toInf": {
        "files": "DYJetsToEE_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYmm_M-50": {
        "files": "DYJetsToMuMu_M-50",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "DYJetsToMuMu_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        
    },
    "ST_s-channel": {
        "files": "ST_s-channel",
        "task_weight": 8,
        
    },
    "ST_t-channel_top_5f": {
        "files": "ST_t-channel_top_5f",
        "task_weight": 8,
        
    },
    "ST_t-channel_antitop_5f": {
        "files": "ST_t-channel_antitop_5f",
        "task_weight": 8,
        
    },
    "ST_tW_top_noHad": {
        "files": "ST_tW_top_noHad",
        "task_weight": 8,
        
    },
    "ST_tW_antitop_noHad": {
        "files": "ST_tW_antitop_noHad",
        "task_weight": 8,
        
    },
    "TTTo2L2Nu": {
        "files": "TTTo2L2Nu",
        "task_weight": 8,
        "top_pt_rwgt": True,
        
    },
    "WWTo2L2Nu": {
        "files": "WWTo2L2Nu",
        "task_weight": 8,
        
    },
    "WZ": {
        "files": "WZ_TuneCP5_13TeV-pythia8",
        "task_weight": 8,
    },
    "ZZ": {
        "files": "ZZ_TuneCP5_13TeV-pythia8",
        "task_weight": 8,
    },
    "GGToEE_M-50to200_El-El": {
        "files": "GGToEE_M-50to200_El-El",
        "task_weight": 8,
        
    },
    "GGToEE_M-50to200_Inel-El_El-Inel": {
        "files": "GGToEE_M-50to200_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToEE_M-50to200_Inel-Inel": {
        "files": "GGToEE_M-50to200_Inel-Inel",
        "task_weight": 8,
        
    },
    "GGToEE_M-200to1500_El-El": {
        "files": "GGToEE_M-200to1500_El-El",
        "task_weight": 8,
        
    },
    "GGToEE_M-200to1500_Inel-El_El-Inel": {
        "files": "GGToEE_M-200to1500_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToEE_M-200to1500_Inel-Inel": {
        "files": "GGToEE_M-200to1500_Inel-Inel",
        "task_weight": 8,
        
    },
    "GGToEE_M-1500toInf_El-El": {
        "files": "GGToEE_M-1500toInf_El-El",
        "task_weight": 8,
        
    },
    "GGToEE_M-1500toInf_Inel-El_El-Inel": {
        "files": "GGToEE_M-1500toInf_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToEE_M-1500toInf_Inel-Inel": {
        "files": "GGToEE_M-1500toInf_Inel-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-50to200_El-El": {
        "files": "GGToMuMu_M-50to200_El-El",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-50to200_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-50to200_Inel-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-200to1500_El-El": {
        "files": "GGToMuMu_M-200to1500_El-El",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-200to1500_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-200to1500_Inel-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-1500toInf_El-El": {
        "files": "GGToMuMu_M-1500toInf_El-El",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-1500toInf_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-El_El-Inel",
        "task_weight": 8,
        
    },
    "GGToMuMu_M-1500toInf_Inel-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-Inel",
        "task_weight": 8,
        
    },
    "WJetsToLNu_0J": {
        "files": "WJetsToLNu_0J",
        "task_weight": 8,
        
    },
    "WJetsToLNu_1J": {
        "files": "WJetsToLNu_1J",
        "task_weight": 8,
        
    },
    "WJetsToLNu_2J": {
        "files": "WJetsToLNu_2J",
        "task_weight": 8,
        
    }
}



for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


DataRun = [
    ["A", "Run2018A-UL2018-v1"],
    ["B", "Run2018B-UL2018-v1"],
    ["C", "Run2018C-UL2018-v1"],
    ["D", "Run2018D-UL2018-v1"],
]

DataSets = ["SingleMuon", "EGamma"]

DataTrig = {
    "SingleMuon": "events.SingleMu",
    "EGamma": "(~events.SingleMu) & (events.SingleEle)"
}


samples_data = []
for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        if "Run2018" in sd and "Muon" in pd:
            tag = tag.replace("v1","GT36")

        datasets[f"{pd}_{era}"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"UL2018{era}",
        }
        samples_data.append(f"{pd}_{era}")


samples = {}
colors = {}

samples = {
    "Data": {
        "samples": samples_data,
        "is_data": True,
    },
    "W+Jets": {
        "samples": [
            "WJetsToLNu_0J",
            "WJetsToLNu_1J",
            "WJetsToLNu_2J",
       ]
    },
    "GGToLL": { 
        "samples": [
            "GGToEE_M-50to200_El-El",
            "GGToEE_M-50to200_Inel-El_El-Inel",
            "GGToEE_M-50to200_Inel-Inel",
            "GGToEE_M-200to1500_El-El",
            "GGToEE_M-200to1500_Inel-El_El-Inel",
            "GGToEE_M-200to1500_Inel-Inel",
            "GGToEE_M-1500toInf_El-El",
            "GGToEE_M-1500toInf_Inel-El_El-Inel",
            "GGToEE_M-1500toInf_Inel-Inel",
            "GGToMuMu_M-50to200_El-El",
            "GGToMuMu_M-50to200_Inel-El_El-Inel",
            "GGToMuMu_M-50to200_Inel-Inel",
            "GGToMuMu_M-200to1500_El-El",
            "GGToMuMu_M-200to1500_Inel-El_El-Inel",
            "GGToMuMu_M-200to1500_Inel-Inel",
            "GGToMuMu_M-1500toInf_El-El",
            "GGToMuMu_M-1500toInf_Inel-El_El-Inel",
            "GGToMuMu_M-1500toInf_Inel-Inel",
        ] 
    },
    "ST": {
        "samples": [
            "ST_s-channel",
            "ST_t-channel_top_5f",
            "ST_t-channel_antitop_5f",
            "ST_tW_top_noHad",
            "ST_tW_antitop_noHad"
        ]
    },
    "TT": {
        "samples": [
            "TTTo2L2Nu"
        ]
    },
    "VV": {
       "samples": [
            "WWTo2L2Nu",
            "WZ",
            "ZZ"
       ]
    },
    "DYtt": {
       "samples": [
           "DYtt_M-50"
       ]
    },
    "DYll": {
       "samples": [
           "DYmm_M-50",
           "DYJetsToMuMu_M-100to200",
           "DYJetsToMuMu_M-200to400",
           "DYJetsToMuMu_M-400to500",
           "DYJetsToMuMu_M-500to700",
           "DYJetsToMuMu_M-700to800",
           "DYJetsToMuMu_M-800to1000",
           "DYJetsToMuMu_M-1000to1500",
           "DYJetsToMuMu_M-1500to2000",
           "DYJetsToMuMu_M-2000toInf",
           "DYee_M-50",
           "DYJetsToEE_M-100to20",
           "DYJetsToEE_M-200to400",
           "DYJetsToEE_M-400to500",
           "DYJetsToEE_M-500to700",
           "DYJetsToEE_M-700to800",
           "DYJetsToEE_M-800to1000",
           "DYJetsToEE_M-1000to1500",
           "DYJetsToEE_M-1500to2000",
           "DYJetsToEE_M-2000toInf",
       ]
    },
    # "DYEFT":{
    #    "samples": [
    #        "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos",
    #           "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos",
    #    ],
    #    "is_signal": True
    # },
}

colors = {}
colors["W+Jets"] = cmap_pastel[0]
colors["GGToLL"] = cmap_pastel[1]
colors["ST"] = cmap_pastel[2]
colors["TT"] = cmap_pastel[3]
colors["VV"] = cmap_pastel[4]
colors["DYtt"] = cmap_pastel[5]
colors["DYll"] = cmap_pastel[6]
colors["DYEFT"] = cmap_pastel[7]

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
    },
    #"inc_ee_ss": {
    #    "func": lambda events: preselections(events) & (events.mll < 500) & events["ee_ss"],
    #    "mask": 0
    #},
    #"inc_mm_ss": {
    #    "func": lambda events: preselections(events) & (events.mll < 500) & events["mm_ss"],
    #    "mask": 0
    #},
    #"inc_em_ss": {
    #    "func": lambda events: preselections(events) & events["em_ss"],
    #    "mask": 0
    #},
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
        "axis": hist.axis.Regular(60, 50, 200, name="mll")
    },
    # "costhetastar_bins": {
    #         "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
    #         "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins")
    # },
    # "yZ_bins": {
    #         "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
    #         "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
    # },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Regular(60, 0, 600, name="ptll"),
    },
    # "etall": {
    #     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
    #     "axis": hist.axis.Regular(80, -8, 8, name="etall"),
    # },
    # "rapll": {
    #     "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity,
    #     "axis": hist.axis.Regular(50, -2.5, 2.5, name="rapll"),
    # },
    # "detall": {
    #     "func": lambda events: abs(events.Lepton[:, 0].deltaeta(events.Lepton[:, 1])),
    #     "axis": hist.axis.Regular(50, 0, 5, name="detall")
    # },
    # "dphill": {
    #     "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
    #     "axis": hist.axis.Regular(63, 0, 3.15, name="dphill")
    # },
    # "dRll": {
    #     "func": lambda events: events.Lepton[:, 0].deltaR(events.Lepton[:, 1]),
    #     "axis": hist.axis.Regular(60, 0, 6, name="dRll")
    # },
    # "ptl1": {
    #     "func": lambda events: events.Lepton[:, 0].pt,
    #     "axis": hist.axis.Regular(60, 20, 320, name="ptl1")
    # },
    # "ptl2": {
    #     "func": lambda events: events.Lepton[:, 1].pt,
    #     "axis": hist.axis.Regular(60, 10, 160, name="ptl2")
    # },
    # "etal1": {
    #     "func": lambda events: events.Lepton[:, 0].eta,
    #     "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal1")
    # },
    # "etal2": {
    #     "func": lambda events: events.Lepton[:, 1].eta,
    #     "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2")
    # },
    # "etal2": {
    #     "func": lambda events: events.Lepton[:, 1].eta,
    #     "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2")
    # },
    "GenPtLL": {
        "func": lambda events: events.Gen_ptll, 
        "axis": hist.axis.Regular(50, 0, 500, name="GenPtLL")
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
