#!/bin/bash

# This script takes in a network model name (e.g. CWGAN1) and a log directory. It automatically
# chooses the latest model in the directory and renders text2shape outputs for the test set.

# Input text descriptions are in the sentences_fake_match.txt files
# Renderings are in the

# NOTE: If the directory contains model checkpoints 500, 1000, 2500, this script will choose model
# 500. If the directory contains 1000, 2500, 5000, this script will choose model 5000.

set -e
export TOOLKIT_PATH=/home/kchen92/Dev/sstk

# Make sure there are two arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi


# #
# # Setup
# #

echo "--- SETUP ---"

# Model type (e.g. CWGAN1)
echo "Model:" $1

# Log path of model
echo "Directory:" $2
log_path=$2


# #
# # Run test script
# #

# Remove .index portion of string
ckpt_path=$(find $log_path -maxdepth 1 -name 'model.ckpt-*.index' | sort | tail -n1)
ckpt_path=${ckpt_path/.index/}

test_command="python main.py --model $1 --test --val_split test --save_outputs --n_minibatch_test 200 --log_path $log_path --ckpt_path $ckpt_path --debug --noise_size 8 --uniform_max 0.5 --dataset shapenet"
echo "Executing:" $test_command
eval $test_command
echo ""


# #
# # Create NRRD files
# #

echo "--- NRRD GENERATION ---"

render_dir=$(basename $ckpt_path)
render_dir=${render_dir/model./}
render_dir=$log_path/$render_dir
gen_nrrd_command="python -m tools.scripts.generate_nrrd $render_dir"

echo "Executing:" $gen_nrrd_command
eval $gen_nrrd_command
echo ""


# #
# # Batch render using javascript script
# #

echo "--- RENDER NRRD ---"

nrrd_file="${render_dir}/nrrd_filenames.txt"

# Render without specifying voxel threshold
out_dir="${render_dir}/no_threshold"
mkdir $out_dir
render_command="node --max-old-space-size=24000 $TOOLKIT_PATH/ssc/render-voxels.js --input $nrrd_file --output_dir $out_dir"
eval $render_command

# Render with 0.8 voxel threshold
# out_dir="${render_dir}/threshold_80"
# mkdir $out_dir
# render_command="node --max-old-space-size=24000 $TOOLKIT_PATH/ssc/render-voxels.js --input $nrrd_file --output_dir $out_dir --voxel_threshold 0.8"
# eval $render_command
