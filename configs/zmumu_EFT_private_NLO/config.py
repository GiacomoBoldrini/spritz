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
plot_label = "DY"
year_label = "2018"
njobs = 1000

runner = f"{fw_path}/src/spritz/runners/runner_3DY.py"

special_analysis_cfg = {
    "do_theory_variations": False
}

bins = {
    "ptll": np.linspace(0, 200, 5),
    "mll": np.linspace(60, 100, 50)
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
    
    "DYtt_M-50": {
        "files": "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYee_M-50": {
        "files": "DYJetsToEE_M-50",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-100to20": {
        "files": "DYJetsToEE_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-200to400": {
        "files": "DYJetsToEE_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-400to500": {
        "files": "DYJetsToEE_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-500to700": {
        "files": "DYJetsToEE_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-700to800": {
        "files": "DYJetsToEE_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-800to1000": {
        "files": "DYJetsToEE_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-1000to1500": {
        "files": "DYJetsToEE_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-1500to2000": {
        "files": "DYJetsToEE_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToEE_M-2000toInf": {
        "files": "DYJetsToEE_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYmm_M-50": {
        "files": "DYJetsToMuMu_M-50",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "DYJetsToMuMu_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        "max_chunks": 100
    },
    "ST_s-channel": {
        "files": "ST_s-channel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "ST_t-channel_top_5f": {
        "files": "ST_t-channel_top_5f",
        "task_weight": 8,
        "max_chunks": 100
    },
    "ST_t-channel_antitop_5f": {
        "files": "ST_t-channel_antitop_5f",
        "task_weight": 8,
        "max_chunks": 100
    },
    "ST_tW_top_noHad": {
        "files": "ST_tW_top_noHad",
        "task_weight": 8,
        "max_chunks": 100
    },
    "ST_tW_antitop_noHad": {
        "files": "ST_tW_antitop_noHad",
        "task_weight": 8,
        "max_chunks": 100
    },
    "TTTo2L2Nu": {
        "files": "TTTo2L2Nu",
        "task_weight": 8,
        "top_pt_rwgt": True,
        "max_chunks": 100
    },
    "WWTo2L2Nu": {
        "files": "WWTo2L2Nu",
        "task_weight": 8,
        "max_chunks": 100
    },
    "WZ": {
        "files": "WZ_TuneCP5_13TeV-pythia8",
        "task_weight": 8,
        "max_chunks": 100
    },
    "ZZ": {
        "files": "ZZ_TuneCP5_13TeV-pythia8",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-50to200_El-El": {
        "files": "GGToEE_M-50to200_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-50to200_Inel-El_El-Inel": {
        "files": "GGToEE_M-50to200_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-50to200_Inel-Inel": {
        "files": "GGToEE_M-50to200_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-200to1500_El-El": {
        "files": "GGToEE_M-200to1500_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-200to1500_Inel-El_El-Inel": {
        "files": "GGToEE_M-200to1500_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-200to1500_Inel-Inel": {
        "files": "GGToEE_M-200to1500_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-1500toInf_El-El": {
        "files": "GGToEE_M-1500toInf_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-1500toInf_Inel-El_El-Inel": {
        "files": "GGToEE_M-1500toInf_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToEE_M-1500toInf_Inel-Inel": {
        "files": "GGToEE_M-1500toInf_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-50to200_El-El": {
        "files": "GGToMuMu_M-50to200_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-50to200_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-50to200_Inel-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-200to1500_El-El": {
        "files": "GGToMuMu_M-200to1500_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-200to1500_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-200to1500_Inel-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-1500toInf_El-El": {
        "files": "GGToMuMu_M-1500toInf_El-El",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-1500toInf_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-El_El-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "GGToMuMu_M-1500toInf_Inel-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-Inel",
        "task_weight": 8,
        "max_chunks": 100
    },
    "WJetsToLNu_0J": {
        "files": "WJetsToLNu_0J",
        "task_weight": 8,
        "max_chunks": 100
    },
    "WJetsToLNu_1J": {
        "files": "WJetsToLNu_1J",
        "task_weight": 8,
        "max_chunks": 100
    },
    "WJetsToLNu_2J": {
        "files": "WJetsToLNu_2J",
        "task_weight": 8,
        "max_chunks": 100
    }
    
}



for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


DataRun = [
    #["A", "Run2018A-UL2018-v1"],
    ["B", "Run2018B-UL2018-v1"],
    #["C", "Run2018C-UL2018-v1"],
    #["D", "Run2018D-UL2018-v1"],
]

DataSets = ["SingleMuon", "EGamma", "DoubleMuon"]

DataTrig = {
    "DoubleMuon": "events.DoubleMu",
    "SingleMuon": "(~events.DoubleMu) & events.SingleMu",
    "EGamma": "(~events.DoubleMu) & (~events.SingleMu) & (events.SingleEle | events.DoubleEle)",
}

samples_data = []
for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        # # FIXME limit to only first era
        if era != "B":
            continue

        if (
            ("DoubleMuon" in pd and "Run2018B" in sd)
            or ("DoubleMuon" in pd and "Run2018D" in sd)
            or ("SingleMuon" in pd and "Run2018A" in sd)
            or ("SingleMuon" in pd and "Run2018B" in sd)
            or ("SingleMuon" in pd and "Run2018C" in sd)
        ):
            tag = tag.replace("v1", "v2")

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
            # "ZZ"
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
           # "DYJetsToMuMu_M-100to200",
           # "DYJetsToMuMu_M-200to400",
           # "DYJetsToMuMu_M-400to500",
           # "DYJetsToMuMu_M-500to700",
           # "DYJetsToMuMu_M-700to800",
           # "DYJetsToMuMu_M-800to1000",
           # "DYJetsToMuMu_M-1000to1500",
           # "DYJetsToMuMu_M-1500to2000",
           # "DYJetsToMuMu_M-2000toInf",
           "DYee_M-50",
           # "DYJetsToEE_M-100to20",
           # "DYJetsToEE_M-200to400",
           # "DYJetsToEE_M-400to500",
           # "DYJetsToEE_M-500to700",
           # "DYJetsToEE_M-700to800",
           # "DYJetsToEE_M-800to1000",
           # "DYJetsToEE_M-1000to1500",
           # "DYJetsToEE_M-1500to2000",
           # "DYJetsToEE_M-2000toInf",
       ],
       # "samples": [
       #     "DY_NLO_EFT_SMEFTatNLO_mll50_100_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll100_200_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll200_400_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll400_600_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll600_800_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll800_1000_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos",
       #        "DY_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos",
       # ],
       "is_signal": True
    },
}

colors = {}
colors["W+Jets"] = cmap_pastel[0]
colors["GGToLL"] = cmap_pastel[1]
colors["ST"] = cmap_pastel[2]
colors["TT"] = cmap_pastel[3]
colors["VV"] = cmap_pastel[4]
colors["DYtt"] = cmap_pastel[5]
colors["DYll"] = cmap_pastel[6]


# for i in datasets.keys():
#     found=False
#     for merged in samples.keys():
#         if i in samples[merged]["samples"]:
#             found=True
#             break
#     
#     if not found:
#         samples[i] = {
#                 "samples": [i]
#                 }
#         colors[i] = cmap_petroff[3]

#####

print("Samples:", samples)

# regions
preselections = lambda events: (events.mll > 50)  # noqa E731

regions = {}

regions["inc_mm"] = {
    "func": lambda events: events["mm"]  & (events.mll > 50),
    "mask": 0,
}

regions["inc_ee"] = {
    "func": lambda events: events["ee"] & (events.mll > 50),
    "mask": 0,
}

regions["inc_emu"] = {
    "func": lambda events: events["em"]  & (events.mll > 50),
    "mask": 0,
}

"""
gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1500, 15000]
costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
yZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)


variables = {}

variables["yZ"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
    "axis": hist.axis.Regular(50, -3.0, 3.0, name="yZ"),
}

variables["yZ_bins"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
    "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
}

variables["mll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Regular(50, 50, 2000, name="mll"),
}

variables["mll_peak"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Regular(50, 80, 180, name="mll_peak"),
}

variables["mll_peak2"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Regular(80, 50, 100, name="mll_peak2"),
}

variables["mll_bins"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    "axis": hist.axis.Variable(gen_mll_bins, name="mll_bins"), 
}

variables["ptll"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
    "axis": hist.axis.Regular(19, 2, 1000, name="ptll"),
}

variables["ptll_less"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
    "axis": hist.axis.Regular(30, 0, 100, name="ptll_less"),
}

variables["ptl1"] = {
    "func": lambda events: (events.Lepton[:, 0]).pt,
    "axis": hist.axis.Regular(30, 0, 100, name="ptl1"),
}

variables["ptl2"] = {
    "func": lambda events: (events.Lepton[:, 1]).pt,
    "axis": hist.axis.Regular(30, 0, 100, name="ptl2"),
}

variables["etal1"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal1"),
}

variables["etall"] = {
    "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
    "axis": hist.axis.Regular(80, -8, 8, name="etall"),
}

variables["costhetastar"] = {
    "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
    "axis": hist.axis.Regular(50, -1, 1, name="costhetastar")
}

variables["costhetastar_bins"] = {
    "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
    "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins")
}

variables["dRll"] = {
    "func": lambda events: events.Lepton[:, 0].deltaR(events.Lepton[:, 1]),
    "axis": hist.axis.Regular(60, 0, 6, name="dRll")
}

variables["dphill"] = {
    "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
    "axis": hist.axis.Regular(63, 0, 3.15, name="dphill")
}

variables["detall"] = {
    "func": lambda events: abs(events.Lepton[:, 0].deltaeta(events.Lepton[:, 1])),
    "axis": hist.axis.Regular(50, 0, 5, name="detall")
}

variables["etall_abs"] = {
    "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).eta),
    "axis": hist.axis.Regular(45, 0, 9, name="etall_abs"),
}

variables["mll_eta"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": [hist.axis.Regular(20, 50, 100, name="mll"), hist.axis.Regular(5, -2.5, 2.5, name="etal1")] 
}

variables["mll_eta_bins"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": [hist.axis.Variable(gen_mll_bins, name="mll_bins"), hist.axis.Regular(5, -2.5, 2.5, name="etal1")]
}

variables["triple_diff"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": [hist.axis.Regular(20, 50, 100, name="mll"), hist.axis.Regular(4, -2.5, 2.5, name="etal1"), hist.axis.Regular(4, 0, 150, name="ptll")] 
}

variables["triple_diff_mll_bins"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": [hist.axis.Variable(gen_mll_bins, name="mll_bins"), hist.axis.Regular(4, -2.5, 2.5, name="etal1"), hist.axis.Regular(4, 0, 150, name="ptll")]
}

variables["triple_diff_theory"] = {
    "func": lambda events: events.Lepton[:, 0].eta,
    "axis": [hist.axis.Variable(gen_mll_bins, name="mll_bins"), hist.axis.Variable(costheta_bins, name="costhetastar_bins"), hist.axis.Variable(yZ_bins, name="yZ_bins")]
}
"""

regions = {
    "inc_ee": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["ee"],
        "mask": 0
    },
    "inc_mm": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["mm"],
        "mask": 0
    },
    "inc_em": {
        "func": lambda events: preselections(events) & events["em"],
        "mask": 0
    },
    "inc_ee_ss": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["ee_ss"],
        "mask": 0
    },
    "inc_mm_ss": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["mm_ss"],
        "mask": 0
    },
    "inc_em_ss": {
        "func": lambda events: preselections(events) & events["em_ss"],
        "mask": 0
    },
}

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)

variables = {
    # Dilepton
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(60, 50, 200, name="mll")
    },
    "mll_medium": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable([50,58,64,72,78,84,90,96,102,108,116,124,132,140,
            148,156,164,172,180,190,200,210,220,230,240,255,270,285,300,325,350,375,
            400,450,500], name="mll_medium")
    },
    "mll_high": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable([50,80,110,130,150,175,200,225,250,275,300,325,350,
            400,450,500,600,800,1000,2000], name="mll_high")
    },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Regular(60, 0, 600, name="ptll"),
    },
    "etall": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
        "axis": hist.axis.Regular(80, -8, 8, name="etall"),
    },
    "etall_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).eta),
        "axis": hist.axis.Regular(45, 0, 9, name="etall_abs"),
    },
    "rapll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="rapll"),
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
        "axis": hist.axis.Regular(60, 0, 3, name="rapll_abs"),
    },
    "detall": {
        "func": lambda events: abs(events.Lepton[:, 0].deltaeta(events.Lepton[:, 1])),
        "axis": hist.axis.Regular(50, 0, 5, name="detall")
    },
    "dphill": {
        "func": lambda events: abs(events.Lepton[:, 0].deltaphi(events.Lepton[:, 1])),
        "axis": hist.axis.Regular(63, 0, 3.15, name="dphill")
    },
    "dRll": {
        "func": lambda events: events.Lepton[:, 0].deltaR(events.Lepton[:, 1]),
        "axis": hist.axis.Regular(60, 0, 6, name="dRll")
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
        "axis": hist.axis.Regular(50, -1, 1, name="costhetastar")
    },
    "ptl1": {
        "func": lambda events: events.Lepton[:, 0].pt,
        "axis": hist.axis.Regular(60, 20, 320, name="ptl1")
    },
    "ptl1_high": {
        "func": lambda events: events.Lepton[:, 0].pt,
        "axis": hist.axis.Variable([20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,
            190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,
            400,415,430,445,460,480,500,525,550,575,600,650,700,800], name="ptl1_high")
    },
    "ptl2": {
        "func": lambda events: events.Lepton[:, 1].pt,
        "axis": hist.axis.Regular(60, 10, 160, name="ptl2")
    },
    "ptl2_high": {
        "func": lambda events: events.Lepton[:, 1].pt,
        "axis": hist.axis.Variable([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,
            180,190,200,215,230,250,290,350,450], name="ptl2_high")
    },
    "etal1": {
        "func": lambda events: events.Lepton[:, 0].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal1")
    },
    "etal2": {
        "func": lambda events: events.Lepton[:, 1].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2")
    },
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

check_weights = {}
