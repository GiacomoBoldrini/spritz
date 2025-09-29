import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

def run_script(i):
    script = "./run_local.sh"
    print(f"[INFO] Starting job {i}...")

    # Open subprocess to stream output
    process = subprocess.Popen(
        ["bash", script, str(i)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered,
        env=os.environ
    )

    # Stream the output line by line
    for line in process.stdout:
        print(f"[Job {i}] {line}", end="")  # Prefix each line with job ID

    returncode = process.wait()
    print(f"[INFO] Job {i} finished with code {returncode}")
    return i, returncode

def main():
    total_jobs = len(glob("condor/job_*"))
    max_workers = 50  # Adjust to number of cores

    print(f"[INFO] Submitting {total_jobs} jobs with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_script, i): i for i in range(0, total_jobs)}

        for future in as_completed(futures):
            i, returncode = future.result()
            if returncode == 0:
                print(f"[OK] Job {i} completed successfully.")
            else:
                print(f"[ERROR] Job {i} failed with return code {returncode}.\n")

    print("\n[INFO] All jobs have finished.")

if __name__ == "__main__":
    main()

