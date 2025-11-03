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
njobs = 50

runner = f"{fw_path}/src/spritz/runners/runner_3DY_trees_singleTriggers.py"

special_analysis_cfg = {
    "do_theory_variations": False
}

USE_FULL = False  

if USE_FULL:
    datasets = {
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
            "task_weight": 8,
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
            "task_weight": 8,
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
            "task_weight": 8,
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
            "task_weight": 8,
        },
         "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
            "task_weight": 8,
        }
    }
    
else:
    datasets = {
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne",
            "task_weight": 8,
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne",
            "task_weight": 8,
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne": {
            "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
            "task_weight": 8,
        }
    }



for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"



samples = {}
colors = {}

if USE_FULL:
    
    samples = {
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne"]
        }
    }
        
else:
    samples = {
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne"]
        },
        "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne" : {
            "samples": ["DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne"]
        }
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
    # "inc_ee": {
    #     "func": lambda events: preselections(events) & events["ee"],
    #     "mask": 0
    # },
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0
    },
    # "inc_em": {
    #     "func": lambda events: preselections(events) & events["em"],
    #     "mask": 0
    # }
}

gen_mll_bins = [50, 100, 200, 400, 600, 800, 1000, 1500, 15000]
costheta_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
yZ_bins = [-3.0, -1.5, 0.0, 1.5, 3.0]

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)

if USE_FULL:
    variables = {
        # Dilepton
        "mll": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
            "axis": hist.axis.Regular(60, 50, 200, name="mll"),
            "save_events": True
        },
        "costhetastar_bins": {
                "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
                "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins"),
                "save_events": True
        },
        "yZ_bins": {
                "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
                "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
                "save_events": True
        },
        "Gen_mll":{
            "func": lambda events: events.Gen_mll,
            "save_events": True
        },
        'sm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 0], 'save_events': True}, 
        'cqlm2_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 3], 'save_events': True}, 
        'cqlm2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 4], 'save_events': True}, 
        'cql32_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 7], 'save_events': True}, 
        'cql32': 
            {'func': lambda events: events.LHEReweightingWeight[:, 8], 'save_events': True}, 
        'cqe2_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 11], 'save_events': True}, 
        'cqe2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 12], 'save_events': True}, 
        'cll1221_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 19], 'save_events': True}, 
        'cll1221': 
            {'func': lambda events: events.LHEReweightingWeight[:, 20], 'save_events': True}, 
        'cpdc_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 21], 'save_events': True}, 
        'cpdc': 
            {'func': lambda events: events.LHEReweightingWeight[:, 22], 'save_events': True}, 
        'cpwb_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 23], 'save_events': True}, 
        'cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 24], 'save_events': True}, 
        'cpl2_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 27], 'save_events': True}, 
        'cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 28], 'save_events': True}, 
        'c3pl1_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 31], 'save_events': True}, 
        'c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 32], 'save_events': True}, 
        'c3pl2_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 33], 'save_events': True}, 
        'c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 34], 'save_events': True}, 
        'cpmu_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 39], 'save_events': True}, 
        'cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 40], 'save_events': True}, 
        'cpqmi_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 43], 'save_events': True}, 
        'cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 44], 'save_events': True}, 
        'cpq3i_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 45], 'save_events': True}, 
        'cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 46], 'save_events': True}, 
        'cpq3_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 47], 'save_events': True}, 
        'cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 48], 'save_events': True}, 
        'cpqm_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 49], 'save_events': True}, 
        'cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 50], 'save_events': True}, 
        'cpu_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 51], 'save_events': True}, 
        'cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 52], 'save_events': True}, 
        'cpd_m1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 53], 'save_events': True}, 
        'cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 54], 'save_events': True}, 
        'cqlm2_cql32': 
            {'func': lambda events: events.LHEReweightingWeight[:, 82], 'save_events': True}, 
        'cqlm2_cqe2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 84], 'save_events': True}, 
        'cqlm2_cll1221': 
            {'func': lambda events: events.LHEReweightingWeight[:, 88], 'save_events': True}, 
        'cqlm2_cpdc': 
            {'func': lambda events: events.LHEReweightingWeight[:, 89], 'save_events': True}, 
        'cqlm2_cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 90], 'save_events': True},
        'cqlm2_cpl2':
            {'func': lambda events: events.LHEReweightingWeight[:, 92], 'save_events': True}, 
        'cqlm2_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 94], 'save_events': True}, 
        'cqlm2_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 95], 'save_events': True}, 
        'cqlm2_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 98], 'save_events': True}, 
        'cqlm2_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 100], 'save_events': True}, 
        'cqlm2_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 101], 'save_events': True}, 
        'cqlm2_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 102], 'save_events': True}, 
        'cqlm2_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 103], 'save_events': True}, 
        'cqlm2_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 104], 'save_events': True}, 
        'cqlm2_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 105], 'save_events': True}, 
        'cql32_cqe2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 131], 'save_events': True}, 
        'cql32_cll1221': 
            {'func': lambda events: events.LHEReweightingWeight[:, 135], 'save_events': True}, 
        'cql32_cpdc': 
            {'func': lambda events: events.LHEReweightingWeight[:, 136], 'save_events': True}, 
        'cql32_cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 137], 'save_events': True}, 
        'cql32_cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 139], 'save_events': True}, 
        'cql32_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 141], 'save_events': True}, 
        'cql32_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 142], 'save_events': True}, 
        'cql32_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 145], 'save_events': True}, 
        'cql32_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 147], 'save_events': True}, 
        'cql32_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 148], 'save_events': True}, 
        'cql32_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 149], 'save_events': True}, 
        'cql32_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 150], 'save_events': True}, 
        'cql32_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 151], 'save_events': True}, 
        'cql32_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 152], 'save_events': True}, 
        'cqe2_cll1221': 
            {'func': lambda events: events.LHEReweightingWeight[:, 178], 'save_events': True}, 
        'cqe2_cpdc': 
            {'func': lambda events: events.LHEReweightingWeight[:, 179], 'save_events': True}, 
        'cqe2_cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 180], 'save_events': True}, 
        'cqe2_cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 182], 'save_events': True}, 
        'cqe2_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 184], 'save_events': True}, 
        'cqe2_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 185], 'save_events': True}, 
        'cqe2_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 188], 'save_events': True}, 
        'cqe2_cpqmi':
            {'func': lambda events: events.LHEReweightingWeight[:, 190], 'save_events': True}, 
        'cqe2_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 191], 'save_events': True}, 
        'cqe2_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 192], 'save_events': True}, 
        'cqe2_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 193], 'save_events': True}, 
        'cqe2_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 194], 'save_events': True},
        'cqe2_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 195], 'save_events': True}, 
        'cll1221_cpdc': 
            {'func': lambda events: events.LHEReweightingWeight[:, 253], 'save_events': True}, 
        'cll1221_cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 254], 'save_events': True}, 
        'cll1221_cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 256], 'save_events': True}, 
        'cll1221_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 258], 'save_events': True}, 
        'cll1221_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 259], 'save_events': True}, 
        'cll1221_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 262], 'save_events': True}, 
        'cll1221_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 264], 'save_events': True}, 
        'cll1221_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 265], 'save_events': True}, 
        'cll1221_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 266], 'save_events': True}, 
        'cll1221_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 267], 'save_events': True}, 
        'cll1221_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 268], 'save_events': True}, 
        'cll1221_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 269], 'save_events': True}, 
        'cpdc_cpwb': 
            {'func': lambda events: events.LHEReweightingWeight[:, 270], 'save_events': True}, 
        'cpdc_cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 272], 'save_events': True},
        'cpdc_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 274], 'save_events': True}, 
        'cpdc_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 275], 'save_events': True}, 
        'cpdc_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 278], 'save_events': True}, 
        'cpdc_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 280], 'save_events': True}, 
        'cpdc_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 281], 'save_events': True}, 
        'cpdc_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 282], 'save_events': True}, 
        'cpdc_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 283], 'save_events': True}, 
        'cpdc_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 284], 'save_events': True}, 
        'cpdc_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 285], 'save_events': True}, 
        'cpwb_cpl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 287], 'save_events': True}, 
        'cpwb_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 289], 'save_events': True}, 
        'cpwb_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 290], 'save_events': True}, 
        'cpwb_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 293], 'save_events': True}, 
        'cpwb_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 295], 'save_events': True}, 
        'cpwb_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 296], 'save_events': True}, 
        'cpwb_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 297], 'save_events': True},
        'cpwb_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 298], 'save_events': True}, 
        'cpwb_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 299], 'save_events': True}, 
        'cpwb_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 300], 'save_events': True}, 
        'cpl2_c3pl1': 
            {'func': lambda events: events.LHEReweightingWeight[:, 316], 'save_events': True},
        'cpl2_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 317], 'save_events': True}, 
        'cpl2_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 320], 'save_events': True}, 
        'cpl2_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 322], 'save_events': True}, 
        'cpl2_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 323], 'save_events': True}, 
        'cpl2_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 324], 'save_events': True}, 
        'cpl2_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 325], 'save_events': True}, 
        'cpl2_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 326], 'save_events': True}, 
        'cpl2_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 327], 'save_events': True}, 
        'c3pl1_c3pl2': 
            {'func': lambda events: events.LHEReweightingWeight[:, 340], 'save_events': True}, 
        'c3pl1_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 343], 'save_events': True}, 
        'c3pl1_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 345], 'save_events': True}, 
        'c3pl1_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 346], 'save_events': True}, 
        'c3pl1_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 347], 'save_events': True}, 
        'c3pl1_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 348], 'save_events': True}, 
        'c3pl1_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 349], 'save_events': True}, 
        'c3pl1_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 350], 'save_events': True}, 
        'c3pl2_cpmu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 353], 'save_events': True}, 
        'c3pl2_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 355], 'save_events': True}, 
        'c3pl2_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 356], 'save_events': True}, 
        'c3pl2_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 357], 'save_events': True}, 
        'c3pl2_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 358], 'save_events': True}, 
        'c3pl2_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 359], 'save_events': True}, 
        'c3pl2_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 360], 'save_events': True}, 
        'cpmu_cpqmi': 
            {'func': lambda events: events.LHEReweightingWeight[:, 379], 'save_events': True}, 
        'cpmu_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 380], 'save_events': True}, 
        'cpmu_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 381], 'save_events': True}, 
        'cpmu_cpqm':
            {'func': lambda events: events.LHEReweightingWeight[:, 382], 'save_events': True}, 
        'cpmu_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 383], 'save_events': True}, 
        'cpmu_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 384], 'save_events': True}, 
        'cpqmi_cpq3i': 
            {'func': lambda events: events.LHEReweightingWeight[:, 391], 'save_events': True}, 
        'cpqmi_cpq3':
            {'func': lambda events: events.LHEReweightingWeight[:, 392], 'save_events': True}, 
        'cpqmi_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 393], 'save_events': True}, 
        'cpqmi_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 394], 'save_events': True}, 
        'cpqmi_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 395], 'save_events': True}, 
        'cpq3i_cpq3': 
            {'func': lambda events: events.LHEReweightingWeight[:, 396], 'save_events': True}, 
        'cpq3i_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 397], 'save_events': True}, 
        'cpq3i_cpu':
            {'func': lambda events: events.LHEReweightingWeight[:, 398], 'save_events': True}, 
        'cpq3i_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 399], 'save_events': True}, 
        'cpq3_cpqm': 
            {'func': lambda events: events.LHEReweightingWeight[:, 400], 'save_events': True}, 
        'cpq3_cpu': 
            {'func': lambda events: events.LHEReweightingWeight[:, 401], 'save_events': True}, 
        'cpq3_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 402], 'save_events': True}, 
        'cpqm_cpu':
            {'func': lambda events: events.LHEReweightingWeight[:, 403], 'save_events': True}, 
        'cpqm_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 404], 'save_events': True}, 
        'cpu_cpd': 
            {'func': lambda events: events.LHEReweightingWeight[:, 405], 'save_events': True}
    }
else:
    variables = {
        # Dilepton
        "mll": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
            "axis": hist.axis.Regular(60, 50, 200, name="mll"),
            "save_events": True
        },
        "costhetastar_bins": {
                "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
                "axis": hist.axis.Variable(costheta_bins, name="costhetastar_bins"),
                "save_events": True
        },
        "yZ_bins": {
                "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).eta,
                "axis": hist.axis.Variable(yZ_bins, name="yZ_bins"),
                "save_events": True
        },
        "Gen_mll":{
            "func": lambda events: events.Gen_mll,
            "save_events": True
        },
        "sm": {
            "func": lambda events: events.LHEReweightingWeight[:, 0],
            "save_events": True,
        },

        "cqlm2_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 1],
            "save_events": True,
        },

        "cqlm2": {
            "func": lambda events: events.LHEReweightingWeight[:, 2],
            "save_events": True,
        },

        "cql32_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 3],
            "save_events": True,
        },

        "cql32": {
            "func": lambda events: events.LHEReweightingWeight[:, 4],
            "save_events": True,
        },

        "cqe2_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 5],
            "save_events": True,
        },

        "cqe2": {
            "func": lambda events: events.LHEReweightingWeight[:, 6],
            "save_events": True,
        },

        "cll1221_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 7],
            "save_events": True,
        },

        "cll1221": {
            "func": lambda events: events.LHEReweightingWeight[:, 8],
            "save_events": True,
        },

        "cpdc_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 9],
            "save_events": True,
        },

        "cpdc": {
            "func": lambda events: events.LHEReweightingWeight[:, 10],
            "save_events": True,
        },

        "cpwb_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 11],
            "save_events": True,
        },

        "cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 12],
            "save_events": True,
        },

        "cpl2_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 13],
            "save_events": True,
        },

        "cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 14],
            "save_events": True,
        },

        "c3pl1_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 15],
            "save_events": True,
        },

        "c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 16],
            "save_events": True,
        },

        "c3pl2_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 17],
            "save_events": True,
        },

        "c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 18],
            "save_events": True,
        },

        "cpmu_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 19],
            "save_events": True,
        },

        "cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 20],
            "save_events": True,
        },

        "cpqmi_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 21],
            "save_events": True,
        },

        "cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 22],
            "save_events": True,
        },

        "cpq3i_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 23],
            "save_events": True,
        },

        "cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 24],
            "save_events": True,
        },

        "cpq3_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 25],
            "save_events": True,
        },

        "cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 26],
            "save_events": True,
        },

        "cpqm_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 27],
            "save_events": True,
        },

        "cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 28],
            "save_events": True,
        },

        "cpu_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 29],
            "save_events": True,
        },

        "cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 30],
            "save_events": True,
        },

        "cpd_m1": {
            "func": lambda events: events.LHEReweightingWeight[:, 31],
            "save_events": True,
        },

        "cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 32],
            "save_events": True,
        },

        "cqlm2_cql32": {
            "func": lambda events: events.LHEReweightingWeight[:, 33],
            "save_events": True,
        },

        "cqlm2_cqe2": {
            "func": lambda events: events.LHEReweightingWeight[:, 34],
            "save_events": True,
        },

        "cqlm2_cll1221": {
            "func": lambda events: events.LHEReweightingWeight[:, 35],
            "save_events": True,
        },

        "cqlm2_cpdc": {
            "func": lambda events: events.LHEReweightingWeight[:, 36],
            "save_events": True,
        },

        "cqlm2_cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 37],
            "save_events": True,
        },

        "cqlm2_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 38],
            "save_events": True,
        },

        "cqlm2_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 39],
            "save_events": True,
        },

        "cqlm2_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 40],
            "save_events": True,
        },

        "cqlm2_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 41],
            "save_events": True,
        },

        "cqlm2_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 42],
            "save_events": True,
        },

        "cqlm2_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 43],
            "save_events": True,
        },

        "cqlm2_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 44],
            "save_events": True,
        },

        "cqlm2_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 45],
            "save_events": True,
        },

        "cqlm2_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 46],
            "save_events": True,
        },

        "cqlm2_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 47],
            "save_events": True,
        },

        "cql32_cqe2": {
            "func": lambda events: events.LHEReweightingWeight[:, 48],
            "save_events": True,
        },

        "cql32_cll1221": {
            "func": lambda events: events.LHEReweightingWeight[:, 49],
            "save_events": True,
        },

        "cql32_cpdc": {
            "func": lambda events: events.LHEReweightingWeight[:, 50],
            "save_events": True,
        },

        "cql32_cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 51],
            "save_events": True,
        },

        "cql32_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 52],
            "save_events": True,
        },

        "cql32_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 53],
            "save_events": True,
        },

        "cql32_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 54],
            "save_events": True,
        },

        "cql32_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 55],
            "save_events": True,
        },

        "cql32_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 56],
            "save_events": True,
        },

        "cql32_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 57],
            "save_events": True,
        },

        "cql32_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 58],
            "save_events": True,
        },

        "cql32_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 59],
            "save_events": True,
        },

        "cql32_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 60],
            "save_events": True,
        },

        "cql32_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 61],
            "save_events": True,
        },

        "cqe2_cll1221": {
            "func": lambda events: events.LHEReweightingWeight[:, 62],
            "save_events": True,
        },

        "cqe2_cpdc": {
            "func": lambda events: events.LHEReweightingWeight[:, 63],
            "save_events": True,
        },

        "cqe2_cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 64],
            "save_events": True,
        },

        "cqe2_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 65],
            "save_events": True,
        },

        "cqe2_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 66],
            "save_events": True,
        },

        "cqe2_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 67],
            "save_events": True,
        },

        "cqe2_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 68],
            "save_events": True,
        },

        "cqe2_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 69],
            "save_events": True,
        },

        "cqe2_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 70],
            "save_events": True,
        },

        "cqe2_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 71],
            "save_events": True,
        },

        "cqe2_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 72],
            "save_events": True,
        },

        "cqe2_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 73],
            "save_events": True,
        },

        "cqe2_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 74],
            "save_events": True,
        },

        "cll1221_cpdc": {
            "func": lambda events: events.LHEReweightingWeight[:, 75],
            "save_events": True,
        },

        "cll1221_cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 76],
            "save_events": True,
        },

        "cll1221_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 77],
            "save_events": True,
        },

        "cll1221_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 78],
            "save_events": True,
        },

        "cll1221_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 79],
            "save_events": True,
        },

        "cll1221_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 80],
            "save_events": True,
        },

        "cll1221_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 81],
            "save_events": True,
        },


        "cll1221_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 82],
            "save_events": True,
        },

        "cll1221_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 83],
            "save_events": True,
        },

        "cll1221_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 84],
            "save_events": True,
        },

        "cll1221_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 85],
            "save_events": True,
        },

        "cll1221_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 86],
            "save_events": True,
        },

        "cpdc_cpwb": {
            "func": lambda events: events.LHEReweightingWeight[:, 87],
            "save_events": True,
        },

        "cpdc_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 88],
            "save_events": True,
        },

        "cpdc_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 89],
            "save_events": True,
        },

        "cpdc_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 90],
            "save_events": True,
        },

        "cpdc_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 91],
            "save_events": True,
        },

        "cpdc_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 92],
            "save_events": True,
        },

        "cpdc_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 93],
            "save_events": True,
        },

        "cpdc_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 94],
            "save_events": True,
        },

        "cpdc_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 95],
            "save_events": True,
        },

        "cpdc_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 96],
            "save_events": True,
        },

        "cpdc_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 97],
            "save_events": True,
        },

        "cpwb_cpl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 98],
            "save_events": True,
        },

        "cpwb_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 99],
            "save_events": True,
        },

        "cpwb_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 100],
            "save_events": True,
        },

        "cpwb_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 101],
            "save_events": True,
        },

        "cpwb_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 102],
            "save_events": True,
        },

        "cpwb_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 103],
            "save_events": True,
        },

        "cpwb_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 104],
            "save_events": True,
        },

        "cpwb_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 105],
            "save_events": True,
        },

        "cpwb_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 106],
            "save_events": True,
        },

        "cpwb_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 107],
            "save_events": True,
        },

        "cpl2_c3pl1": {
            "func": lambda events: events.LHEReweightingWeight[:, 108],
            "save_events": True,
        },

        "cpl2_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 109],
            "save_events": True,
        },

        "cpl2_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 110],
            "save_events": True,
        },

        "cpl2_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 111],
            "save_events": True,
        },

        "cpl2_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 112],
            "save_events": True,
        },

        "cpl2_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 113],
            "save_events": True,
        },

        "cpl2_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 114],
            "save_events": True,
        },

        "cpl2_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 115],
            "save_events": True,
        },

        "cpl2_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 116],
            "save_events": True,
        },

        "c3pl1_c3pl2": {
            "func": lambda events: events.LHEReweightingWeight[:, 117],
            "save_events": True,
        },

        "c3pl1_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 118],
            "save_events": True,
        },

        "c3pl1_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 119],
            "save_events": True,
        },

        "c3pl1_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 120],
            "save_events": True,
        },

        "c3pl1_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 121],
            "save_events": True,
        },

        "c3pl1_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 122],
            "save_events": True,
        },

        "c3pl1_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 123],
            "save_events": True,
        },

        "c3pl1_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 124],
            "save_events": True,
        },

        "c3pl2_cpmu": {
            "func": lambda events: events.LHEReweightingWeight[:, 125],
            "save_events": True,
        },

        "c3pl2_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 126],
            "save_events": True,
        },

        "c3pl2_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 127],
            "save_events": True,
        },

        "c3pl2_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 128],
            "save_events": True,
        },

        "c3pl2_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 129],
            "save_events": True,
        },

        "c3pl2_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 130],
            "save_events": True,
        },

        "c3pl2_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 131],
            "save_events": True,
        },

        "cpmu_cpqmi": {
            "func": lambda events: events.LHEReweightingWeight[:, 132],
            "save_events": True,
        },

        "cpmu_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 133],
            "save_events": True,
        },

        "cpmu_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 134],
            "save_events": True,
        },

        "cpmu_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 135],
            "save_events": True,
        },

        "cpmu_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 136],
            "save_events": True,
        },

        "cpmu_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 137],
            "save_events": True,
        },

        "cpqmi_cpq3i": {
            "func": lambda events: events.LHEReweightingWeight[:, 138],
            "save_events": True,
        },

        "cpqmi_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 139],
            "save_events": True,
        },

        "cpqmi_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 140],
            "save_events": True,
        },

        "cpqmi_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 141],
            "save_events": True,
        },

        "cpqmi_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 142],
            "save_events": True,
        },

        "cpq3i_cpq3": {
            "func": lambda events: events.LHEReweightingWeight[:, 143],
            "save_events": True,
        },

        "cpq3i_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 144],
            "save_events": True,
        },

        "cpq3i_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 145],
            "save_events": True,
        },

        "cpq3i_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 146],
            "save_events": True,
        },

        "cpq3_cpqm": {
            "func": lambda events: events.LHEReweightingWeight[:, 147],
            "save_events": True,
        },

        "cpq3_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 148],
            "save_events": True,
        },

        "cpq3_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 149],
            "save_events": True,
        },

        "cpqm_cpu": {
            "func": lambda events: events.LHEReweightingWeight[:, 150],
            "save_events": True,
        },

        "cpqm_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 151],
            "save_events": True,
        },

        "cpu_cpd": {
            "func": lambda events: events.LHEReweightingWeight[:, 152],
            "save_events": True,
        }
        
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

