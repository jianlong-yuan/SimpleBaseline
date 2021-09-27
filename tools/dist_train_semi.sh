#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $RANDOM \
    $(dirname "$0")/train_semi.py --config-file $CONFIG ${@:3}