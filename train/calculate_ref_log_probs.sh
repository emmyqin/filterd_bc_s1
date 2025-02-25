#!/bin/bash

# Register a signal handler to make sure everything gets killed if this script gets interrupted

# Function to kill background processes
kill_background_processes() {
  jobs -p | xargs kill
  exit 0
}

# Trap the SIGINT signal (Ctrl+C)
trap kill_background_processes SIGINT

echo "Starting background processes to process data."

CUDA_VISIBLE_DEVICES=0 python train/calculate_ref_logits.py --split=0 & pid0=$!
CUDA_VISIBLE_DEVICES=1 python train/calculate_ref_logits.py --split=1 & pid1=$!
CUDA_VISIBLE_DEVICES=2 python train/calculate_ref_logits.py --split=2 & pid2=$!
CUDA_VISIBLE_DEVICES=3 python train/calculate_ref_logits.py --split=3 & pid3=$!
CUDA_VISIBLE_DEVICES=4 python train/calculate_ref_logits.py --split=4 & pid4=$!
CUDA_VISIBLE_DEVICES=5 python train/calculate_ref_logits.py --split=5 & pid5=$!
CUDA_VISIBLE_DEVICES=6 python train/calculate_ref_logits.py --split=6 & pid6=$!
CUDA_VISIBLE_DEVICES=7 python train/calculate_ref_logits.py --split=7 & pid7=$!


# Wait for all background processes to complete
wait $pid0 $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7

echo "All background processes finished processing! Exiting now!"