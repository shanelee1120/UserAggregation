#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Missing arguments: <party (guest|host)> <dataset> <pack-size> <perturb (proj|iso-proj|marvell|none)> [perturb_param]"
    exit 1
fi

WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${WORKSPACE}/../../env.exp
export PYTHONPATH="$PYTHONPATH:$WORKSPACE"
export DATASET_DIR="$WORKSPACE/data"

party=$1
data=$2
pack_size=$3
perturb=$4

perturb_args="--perturb=${perturb}"
if [ "$perturb" = "proj" ] || [ "$perturb" = "iso-proj" ]; then
    if [ "$#" -lt 5 ]; then
        echo "Missing the sum_kl argument for ${perturb}"
        exit 1
    fi
    sum_kl=$5
    perturb_args="${perturb_args} --sum-kl-bound=${sum_kl}"
elif [ "$perturb" = "marvell" ]; then
    if [ "$#" -lt 5 ]; then
        echo "Missing the init_scale argument for ${perturb}"
        exit 1
    fi
    init_scale=$5
    perturb_args="${perturb_args} --init-scale=${init_scale}"
fi

data_args="--data=${data} --pack-size=${pack_size}"

cmd="python3 -u main.py --party=${party} ${data_args} ${perturb_args}"

echo "Running command: $cmd"
$cmd