#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

# path to the Conda environment
# ENV_PREFIX="$PROJECT_DIR"/env

# project should have a data directory
DATA_DIR="$PROJECT_DIR"/data

# creates a separate directory for each job
JOB_NAME=example-training-job-1
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
sbatch --job-name "$JOB_NAME" \
    "$PROJECT_DIR"/bin/train.sbatch \
    "$PROJECT_DIR"/src/ronin_lstm_tcn.py \
    	test \
	--type lstm_bi \
	--data_dir "$DATA_DIR"/seen_subjects_test_set \
	--out_dir "$JOB_RESULTS_DIR" \
	--dataset 'ronin' \
	--model_path "$PROJECT_DIR"/results/ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt \
	--test_list "$PROJECT_DIR"/lists/list_test_seen.txt
