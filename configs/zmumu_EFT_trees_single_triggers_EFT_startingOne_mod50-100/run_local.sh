    #!/bin/bash

    export X509_USER_PROXY=/gwpool/users/gboldrini/spritz/configs/zmumu_EFT_private_NLO/x509up_u11203
    ERA=Full2018v9
    job_id=$1
    
    CWD=$PWD
    cd condor/job_${job_id}

    # configs/.../condor/job_0/tmp

    mkdir tmp
    cd tmp
    cp ../chunks_job.pkl .
    cp ../../run.sh .
    cp $X509_USER_PROXY .
    cp /gwpool/users/gboldrini/spritz/src/spritz/runners/runner_3DY_trees_singleTriggers.py runner.py
    cp /gwpool/users/gboldrini/spritz/data/${ERA}/cfg.json .
    cp $CWD/config.py .

    # ./run.sh 2> err 1> out
    ./run.sh
    cp results.pkl ../chunks_job.pkl
    # mv err ../err.txt
    # mv out ../out.txt
    # echo "Run locally" >> ../err.txt
    echo "Done ${job_id}"
    cd $CWD
