    #!/bin/bash

    export X509_USER_PROXY=/afs/cern.ch/user/g/gboldrin/proxy/x509up_u132569
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
    cp /afs/cern.ch/user/g/gboldrin/spritz/src/spritz/runners/runner_3DY.py runner.py
    cp /afs/cern.ch/user/g/gboldrin/spritz/data/${ERA}/cfg.json .
    cp $CWD/config.py .

    # ./run.sh 2> err 1> out
    ./run.sh
    cp results.pkl ../chunks_job.pkl
    # mv err ../err.txt
    # mv out ../out.txt
    # echo "Run locally" >> ../err.txt
    echo "Done ${job_id}"
    cd $CWD
