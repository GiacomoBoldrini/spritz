import json
import os
import subprocess
import sys  # noqa: F401

from spritz.framework.framework import (
    add_dict,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)


def preprocess_chunks(year):
    with open(f"{get_fw_path()}/data/common/forms.json", "r") as file:
        forms_common = json.load(file)
    with open(f"{get_fw_path()}/data/{year}/forms.json", "r") as file:
        forms_era = json.load(file)
    forms = add_dict(forms_common, forms_era)
    new_chunks = read_chunks("data/chunks.pkl")

    for i, chunk in enumerate(new_chunks):
        print(chunk)
        new_chunks[i]["data"]["read_form"] = forms[chunk["data"]["read_form"]]
    return new_chunks


def split_chunks(chunks, n):
    """
    Splits list l of chunks into n jobs with approximately equals sum of values
    see  http://stackoverflow.com/questions/6855394/splitting-list-in-chunks-of-balanced-weight
    """
    jobs = [[] for i in range(n)]
    sums = {i: 0 for i in range(n)}
    c = 0
    for chunk in chunks:
        for i in sums:
            if c == sums[i]:
                jobs[i].append(chunk)
                break
        sums[i] += chunk["weight"]
        c = min(sums.values())
    return jobs


def submit(
    new_chunks,
    path_an,
    an_dict,
    njobs=500,
    clean_up=True,
    start=0,
    dryRun=False,
    script_name="script_worker.py",
):
    machines = [
        # # "clipper.hcms.it",
        # "pccms01.hcms.it",
        # "pccms02.hcms.it",
        "pccms04.hcms.it",
        # # "pccms08.hcms.it",
        # "pccms11.hcms.it",
        # "clipper.hcms.it",
        # "empire.hcms.it",
        "pccms12.hcms.it",
        "pccms13.hcms.it",
        "pccms14.hcms.it",
        # "hercules02.hcms.it",
    ]

    # if running on lxplus we probably want to avoid specifying machines
    # on which to run jobs

    if "lxplus" in os.uname()[1]: machines = []

    print("N chunks", len(new_chunks))
    print(sorted(list(set(list(map(lambda k: k["data"]["dataset"], new_chunks))))))

    jobs = split_chunks(new_chunks, njobs)
    print(len(jobs))

    folders = []

    if clean_up:
        proc = subprocess.Popen(
            "rm -r condor_backup; mv condor condor_backup",
            shell=True,
        )
        proc.wait()

    for i, job in enumerate(jobs):
        folder = f"condor/job_{start+i}"

        os.makedirs(folder, exist_ok=False)
        write_chunks(job, f"{folder}/chunks_job.pkl")
        write_chunks(job, f"{folder}/chunks_job_original.pkl")

        folders.append(folder.split("/")[-1])
    proc = subprocess.Popen(f"cp {script_name} condor/", shell=True)
    proc.wait()

    # need to transfer the user certificate to access
    # xrootd on condor nodes
    if "X509_USER_PROXY" not in os.environ.keys():
        print("--> setup grid certificate at X509_USER_PROXY")
        sys.exit(0)

    x509name = os.environ["X509_USER_PROXY"].split("/")[-1]
    condor_script_name = script_name.split("/")[-1]

    txtsh = "#!/bin/bash\n"
    txtsh += "git clone https://github.com/GiacomoBoldrini/spritz.git\n"
    txtsh += f"export X509_USER_PROXY=$PWD/{x509name}\n"
    txtsh += "cd spritz ; source start.sh; cd ..\n"
    #txtsh += "export PATH=\"$HOME/.local/bin:$PATH\"\n"
    txtsh += f"time python {condor_script_name} .\n"

    with open("condor/run.sh", "w") as file:
        file.write(txtsh)

    txtjdl = "universe = vanilla \n"
    txtjdl += "executable = run.sh\n"
    txtjdl += "arguments = $(Folder)\n"
    txtjdl += "MY.SingularityImage = \"/eos/user/g/gboldrin/spritz_v2.sif\"\n" 
    txtjdl += "should_transfer_files = YES\n"
    txtjdl += f"transfer_input_files = {os.environ["X509_USER_PROXY"]}, $(Folder)/chunks_job.pkl, "
    txtjdl += f" {script_name}, {get_fw_path()}/data/{an_dict['year']}/cfg.json, {path_an}/config.py\n"
    txtjdl += 'transfer_output_remaps = "results.pkl = $(Folder)/chunks_job.pkl"\n'
    txtjdl += "output = $(Folder)/out.txt\n"
    txtjdl += "error  = $(Folder)/err.txt\n"
    txtjdl += "log    = $(Folder)/log.txt\n"
    txtjdl += "request_cpus=1\n"
    txtjdl += "request_memory=2000\n"
    if len(machines) > 0:
        txtjdl += (
            "Requirements = "
            + " || ".join([f'(machine == "{machine}")' for machine in machines])
            + "\n"
        )
    queue = "workday"
    txtjdl += f'+JobFlavour = "{queue}"\n'

    txtjdl += f'queue 1 Folder in {", ".join(folders)}\n'
    with open("condor/submit.jdl", "w") as file:
        file.write(txtjdl)

    if dryRun:
        command = "cd condor/; chmod +x run.sh; cd -"
    else:
        command = "cd condor/; chmod +x run.sh; condor_submit submit.jdl; cd -"
    proc = subprocess.Popen(command, shell=True)
    proc.wait()


def main():
    start = 0
    path_an = os.path.abspath(".")
    an_dict = get_analysis_dict()
    chunks = preprocess_chunks(an_dict["year"])
    dryRun = False
    runner_default = f"{get_fw_path()}/src/spritz/runners/runner_default.py"

    runner_name = an_dict["runner"] if "runner" in an_dict else runner_default 

    if len(sys.argv) > 1:
        dryRun = sys.argv[1] == "-dr"

    submit(
        chunks,
        path_an,
        an_dict,
        njobs=an_dict["njobs"],
        clean_up=True,
        start=start,
        dryRun=dryRun,
        script_name=runner_name,
    )


if __name__ == "__main__":
    main()
