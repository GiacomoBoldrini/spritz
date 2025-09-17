import glob
import json
import os
import sys
import gfal2
import uproot
import multiprocessing as mp

from tqdm import tqdm
from dbs.apis.dbsClient import DbsApi
from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.utils import rucio_utils
from XRootD import client

path_fw = get_fw_path()

def list_directories(path):
    
    
    host = path.split("//eos/")[0]
    path = path.split(host)[1]
    
    print("host ", host)
    print("path ", path)
    
    fs = client.FileSystem(host)

    dirs = []
    status, listing = fs.dirlist(path)
    
    print(status, listing)
    
    if not status.ok:
        raise RuntimeError(f"Error: {status}")

    for entry in listing:
        name = entry.name
        if entry.statinfo is not None:
            # We got statinfo, so we can check flags
            if entry.statinfo.flags & client.flags.StatInfoFlags.IS_DIR:
                dirs.append(name)
        else:
            # No statinfo returned  fall back to stat() call
            fullpath = path.rstrip("/") + "/" + name
            stat_status, statinfo = fs.stat(fullpath)
            if stat_status.ok and statinfo.flags & client.flags.StatInfoFlags.IS_DIR:
                dirs.append(name)

    return dirs

def process_file(args):
    found_file, sample_name = args
    try:
        f = uproot.open(found_file)
        nevents = f["Events"].num_entries
        return {"sample_name": sample_name, "path": [found_file], "nevents": nevents}
    except Exception as e:
        return {"sample_name": sample_name, "path": [found_file], "nevents": 0, "error": str(e)}

def get_files(era, active_samples, pfs={}):
    Samples = {}

    with open(f"{path_fw}/data/{era}/samples/samples.json") as file:
        Samples = json.load(file)
        if active_samples == "ALL":
            Samples = {k: v for k, v in Samples["samples"].items()}
        else:
            Samples = {
                k: v for k, v in Samples["samples"].items() if k in active_samples
            }

    files = {}
    for sampleName in Samples:
        print(sampleName)
        if "nanoAOD" in Samples[sampleName]:
            files[sampleName] = {"query": Samples[sampleName]["nanoAOD"], "files": []}
        elif "path" in Samples[sampleName]:
            if sampleName in pfs.keys(): 
                print(f"Sample {sampleName} already present in the provided fileset, skipping the search for files")
                files[sampleName] = pfs[sampleName]
                continue
            files[sampleName] = {"files": []}
            # handle later the fact that it can have /0000 /0001 etc
            if Samples[sampleName]["path"].startswith("root://"):
                print("searching for directories in ", Samples[sampleName]["path"])
                dirs = list_directories(Samples[sampleName]["path"])
                print(dirs)
                ctx = gfal2.creat_context()
                found_files = []
                for d__ in dirs:
                    fp = os.path.join(Samples[sampleName]["path"], d__)
                    found_files += [os.path.join(fp , p__) for p__ in ctx.listdir(fp)]
                # sanity check 
                if not all([i.endswith(".root") for i in found_files]):
                    print("Found files in the directory that are not .root files!")
                    
                    found_files = [i for i in found_files if i.endswith(".root")]
            else:
                found_files = glob.glob(Samples[sampleName]["path"])
            print(sampleName)
            # Make this parallel, we open each file and retrieve the number of events
            # can be really slow on single core due to I/O operations
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(process_file, [(f, sampleName) for f in found_files]), total=len(found_files)))

            for result in results:
                if "error" in result:
                    print(f"Error processing {result['path'][0]}: {result['error']}")
                else:
                    files[result["sample_name"]]["files"].append(
                        {"path": result["path"], "nevents": result["nevents"]}
                    )
        elif "files" in Samples[sampleName]:
            files[sampleName] = {"files": []}
            for found_file in Samples[sampleName]["files"]:
                if isinstance(found_file, str):
                    f = uproot.open(found_file)
                    nevents = f["Events"].num_entries
                    files[sampleName]["files"].append(
                        {"path": [found_file], "nevents": nevents}
                    )
                elif isinstance(found_file, dict):
                    if "path" not in found_file:
                        raise Exception(
                            "Found file is a dict but does not contain 'path' key!"
                        )
                    if "nevents" in found_file.keys():
                        nevents = found_file["nevents"]
                        files[sampleName]["files"].append(
                        {"path": [found_file["path"]], "nevents": nevents}
                        )

    return files


def main():
    
    recycle = False
    if len(sys.argv) > 1:
        recycle = sys.argv[1] == "-r"
        
    an_dict = get_analysis_dict()
    era = an_dict["year"]
    datasets = [k["files"] for k in an_dict["datasets"].values()]
    
    # avoid to search for files already present 
    present_fileset = {}
    if recycle and os.path.isfile("data/fileset.json"):
        with open("data/fileset.json", "r") as f:
            present_fileset = json.load(f)
            
    files = get_files(era, datasets, pfs=present_fileset)
    rucio_client = rucio_utils.get_rucio_client()
    # DE|FR|IT|BE|CH|ES|UK
    good_sites = ["IT", "FR", "BE", "CH", "UK", "ES", "DE", "US"]
    for dname in files:
        if "query" not in files[dname]:
            continue
        dataset = files[dname]["query"]
        print("Checking", dname, "files with query", dataset)
        try:
            (
                outfiles,
                outsites,
                sites_counts,
            ) = rucio_utils.get_dataset_files_replicas(
                dataset,
                allowlist_sites=[],
                blocklist_sites=[
                    # "T2_FR_IPHC",
                    # "T2_ES_IFCA",
                    # "T2_CH_CERN",
                    "T3_IT_Trieste",
                ],
                # regex_sites=[],
                regex_sites=r"T[123]_(" + "|".join(good_sites) + ")_\w+",
                # regex_sites = r"T[123]_(DE|IT|BE|CH|ES|UK|US)_\w+",
                mode="full",  # full or first. "full"==all the available replicas
                client=rucio_client,
            )
        except Exception as e:
            print(f"\n[red bold] Exception: {e}[/]")
            sys.exit(1)

        url = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
        api = DbsApi(url=url)
        filelist = api.listFiles(dataset=dataset, detail=1)

        for replicas, _ in zip(outfiles, outsites):
            prefix = "/store/data"
            if prefix not in replicas[0]:
                prefix = "/store/mc"
            logical_name = prefix + replicas[0].split(prefix)[-1]

            right_file = list(
                filter(lambda k: k["logical_file_name"] == logical_name, filelist)
            )
            if len(right_file) == 0:
                raise Exception("File present in rucio but not dbs!", logical_name)
            if len(right_file) > 1:
                raise Exception(
                    "More files have the same logical_file_name, not support"
                )
            nevents = right_file[0]["event_count"]
            files[dname]["files"].append({"path": replicas, "nevents": nevents})

    os.makedirs("data", exist_ok=True)
    with open("data/fileset.json", "w") as file:
        json.dump(files, file, indent=2)


if __name__ == "__main__":
    main()
