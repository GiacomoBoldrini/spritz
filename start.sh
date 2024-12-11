#!/bin/bash

XRD_RUNFORKHANDLER=1

export SPRITZ_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SPRITZ_PATH

SPECIAL_SOURCE=${SPRITZ_PATH}/special_start.sh
if [ -f "$SPECIAL_SOURCE" ]; then
    source $SPECIAL_SOURCE
fi

export PYTHONPATH=${SPRITZ_PATH}:$PYTHONPATH
