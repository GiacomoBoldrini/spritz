import concurrent.futures
import glob
import hashlib
import os
from math import ceil
from typing import NewType
import sys
import argparse

from spritz.framework.framework import (  # noqa: F401
    add_dict_iterable,
    read_chunks,
    write_chunks,
)

Result = NewType("Result", dict[str, dict])

def read_inputs(inputs: list[str]) -> list[Result]:
    inputs_obj = []
    for input in inputs:
        job_result = read_chunks(input)
        # print("JOB RESULT")
        # print(job_result, job_result == -99999, inputs)
        new_job_result = []
        if isinstance(job_result, list):
            for job_result_single in job_result:
                if job_result_single["result"] != {}:
                    new_job_result.append(job_result_single["result"]["real_results"])
            # job_result = new_job_result
            # if check_input(job_result):
            #     inputs_obj.append(job_result["real_results"])
            inputs_obj.extend(new_job_result)
        else:
            # job_result = {k: v for k, v in job_result.items() if k in ["result", "error"]}
            # del job_result["result"]["performance"]
            # if check_input(job_result):
            #     inputs_obj.append(job_result["real_results"])
            inputs_obj.append(job_result)
    # print(inputs_obj)
    return inputs_obj

basepath = os.path.abspath("condor")
inputs = glob.glob(f"{basepath}/job_*/chunks_job.pkl")[:]

inputs_obj = read_inputs(inputs[:30])
result = add_dict_iterable(inputs_obj)

for key in result.keys():
    print(key, result[key].keys())
