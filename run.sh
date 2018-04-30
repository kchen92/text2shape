#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

export CUDA_VISIBLE_DEVICES=$1
export EXPERIMENT=$2
export PURPOSE=$3
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPATH=./outputs/$EXPERIMENT/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p ./outputs
export MASTER_LOG=./outputs/manifest.md
echo "## Experiment: $2" >> $MASTER_LOG
echo "" >> $MASTER_LOG
echo "  Purpose: $3" >> $MASTER_LOG
echo "  Path: $OUTPATH" >> $MASTER_LOG
echo "" >> $MASTER_LOG

mkdir -p $OUTPATH

###################################
# Trap SIGINT
###################################
function post_processing() {
  echo -n "Successful? [y/n]: "
  read
  if [ "$REPLY" = "n" ]; then
    echo -n "Reason for termination?: "
    read
    sed -i "/${TIME}/a \  Termination: $REPLY" $MASTER_LOG
  fi
}

# trap ctrl-c and call ctrl_c()
trap post_processing INT SIGINT SIGTERM
###################################

{
LOG="$OUTPATH/$TIME.txt"
echo Logging output to "$LOG"
time python main.py \
    --log_path $OUTPATH \
    $4 | tee -a "$LOG"
}
