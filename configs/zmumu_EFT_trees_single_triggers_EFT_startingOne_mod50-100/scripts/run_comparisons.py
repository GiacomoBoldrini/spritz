import os
from glob import glob
import argparse
from multiprocessing import Pool, cpu_count
import sys 

ranges = {
    "cpqmi":{
        "mll":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.3,0.3"
        },
        "triple_diff":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.3,0.3"
        }
    },
    "cpq3i":{
        "mll":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.05,0.05"
        },
        "triple_diff":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.05,0.05"
        }
    },
    "cpqm":{
        "mll":{
            "MCstat": "-2,2",
            "noMCstat": "-1,1"
        },
        "triple_diff":{
            "MCstat": "-2,2",
            "noMCstat": "-1,1"
        }
    },
    "c3pl1":{
        "mll":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.5,0.5"
        },
        "triple_diff":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.5,0.5"
        }
    },
    "cpd":{
        "mll":{
            "MCstat": "-0.8,0.8",
            "noMCstat": "-0.5,0.5"
        },
        "triple_diff":{
            "MCstat": "-0.8,0.8",
            "noMCstat": "-0.5,0.5"
        }
    },
    "cqlm2":{
        "mll":{
            "MCstat": "-0.6,0.4",
            "noMCstat": "-0.4,0.3"
        },
        "triple_diff":{
            "MCstat": "-0.6,0.4",
            "noMCstat": "-0.4,0.3"
        }
    },
    "cql32":{
        "mll":{
            "MCstat": "-0.3,0.2",
            "noMCstat": "-0.3,0.2"
        },
        "triple_diff":{
            "MCstat": "-0.3,0.2",
            "noMCstat": "-0.3,0.2"
        }
    },
    "c3pl2":{
        "mll":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.1,0.1"
        },
        "triple_diff":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.1,0.1"
        }
    },
    "cll1221":{
        "mll":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.5,0.5"
        },
        "triple_diff":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.5,0.5"
        }
    },
    "cpq3":{
        "mll":{
            "MCstat": "-1,1",
            "noMCstat": "-0.5,0.5"
        },
        "triple_diff":{
            "MCstat": "-1,1",
            "noMCstat": "-0.5,0.5"
        }
    },
    "cpdc":{
        "mll":{
            "MCstat": "-0.1,0.1",
            "noMCstat": "-0.05,0.05"
        },
        "triple_diff":{
            "MCstat": "-0.1,0.1",
            "noMCstat": "-0.05,0.05"
        }
    },
    "cqe2":{
        "mll":{
            "MCstat": "-0.6,0.6",
            "noMCstat": "-0.6,0.6"
        },
        "triple_diff":{
            "MCstat": "-0.6,0.6",
            "noMCstat": "-0.6,0.6"
        }
    },
    "cpwb":{
        "mll":{
            "MCstat": "-0.06,0.06",
            "noMCstat": "-0.03,0.03"
        },
        "triple_diff":{
            "MCstat": "-0.06,0.06",
            "noMCstat": "-0.03,0.03"
        }
    },
    "cpu":{
        "mll":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.3,0.3"
        },
        "triple_diff":{
            "MCstat": "-0.5,0.5",
            "noMCstat": "-0.3,0.3"
        }
    },
    "cpmu":{
        "mll":{
            "MCstat": "-0.1,0.1",
            "noMCstat": "-0.05,0.05"
        },
        "triple_diff":{
            "MCstat": "-0.1,0.1",
            "noMCstat": "-0.05,0.05"
        }
    },
    "cpl2":{
        "mll":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.08,0.08"
        },
        "triple_diff":{
            "MCstat": "-0.2,0.2",
            "noMCstat": "-0.08,0.08"
        }
    },
}
# -----------------------------
# Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Process DY EFT histograms")
    parser.add_argument("-d", "--directory", default="datacards", type=str)
    parser.add_argument("-xr", "--xrdcp", default="", type=str)
    parser.add_argument("-j", "--jobs", default=cpu_count(), type=int,
                        help="number of parallel folders")
    return parser.parse_args()


# -----------------------------
# Worker: one folder
# -----------------------------
def process_folder(args):
    folder, xrdcp = args
    pwd = os.getcwd()

    print(f"[PID {os.getpid()}] Processing folder {folder}")

    try:
        ops = os.path.basename(folder).split("_")

        os.chdir(folder)
        os.system("spritz-cards")
        os.chdir("datacards")

        regions = [r.rstrip("/") for r in glob("*/")]

        for region in regions:
            os.chdir(region)
            variables = [v.rstrip("/") for v in glob("*/")]
            var_fol = os.getcwd()

            for variable in variables:
                print(f"  -> {region}/{variable}")
                os.chdir(variable)

                # with MC stat unc
                key="MCstat"
                print("----> createJson.py --binname wm1_ --ranges {}".format(":".join(["{}={}".format(op, ranges[op][variable][key]) for op in ops])))
                os.system("createJson.py --binname wm1_ --ranges {}".format(":".join(["{}={}".format(op, ranges[op][variable][key]) for op in ops])))
                os.system("createCombineJson.py --datacard datacard.txt --binname wm1_")
                os.system(f"createWS.py {len(ops)}")
                os.system(f"runScans.py {len(ops)} initial")
                os.system(f"runScans.py {len(ops)} scan --npoints=2000")

                # no MC stat unc
                os.system("makeBundle.py --datacard datacard.txt --output noMC")
                os.chdir("noMC")
                key="noMCstat"
                print("----> createJson.py --binname wm1_ --ranges {}".format(":".join(["{}={}".format(op, ranges[op][variable][key]) for op in ops])))
                os.system("createJson.py --binname wm1_ --ranges {}".format(":".join(["{}={}".format(op, ranges[op][variable][key]) for op in ops])))
                os.system("createCombineJson.py --datacard datacard.txt --binname wm1_")
                os.system(f"createWS.py {len(ops)}")
                os.system(f"runScans.py {len(ops)} initial")
                os.system(f"runScans.py {len(ops)} scan --npoints=2000")

                os.chdir(var_fol)

            # comparison plot (serial inside folder)
            fn = []
            for variable in variables:
                fn.append(os.path.join(variable,
                    f"higgsCombine.{ '_'.join(ops) }.individual.MultiDimFit.mH125.root"))
                fn.append(os.path.join(variable, "noMC",
                    f"higgsCombine.{ '_'.join(ops) }.individual.MultiDimFit.mH125.root"))

            cmd = (
                f'mkEFTScan.py {fn[0]} -ml "{fn[0].split("/")[0]}" '
                + "-p " + " ".join(["k_" + i for i in ops]) + " -others "
            )

            for idx, f in enumerate(fn[1:]):
                label = f.split("/")[0]
                if "noMC" in f:
                    label += " noMC"
                cmd += f'{f}:{idx+2}:2:"{label}" '

            cmd += f'-o scan_{"_".join(ops)}'
            print("    Running:", cmd)
            os.system(cmd)

            if xrdcp:
                os.system(f"xrdcp -f scan_{'_'.join(ops)}.* {xrdcp}/.")

    finally:
        os.chdir(pwd)


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    datacard_folders = glob(os.path.join(args.directory, "*"))

    tasks = [(folder, args.xrdcp) for folder in datacard_folders]

    with Pool(processes=args.jobs) as pool:
        pool.map(process_folder, tasks)


if __name__ == "__main__":
    main()
