#!/bin/bash
SBATCH --time=01:00:00  # execution time of the job
SBATCH --job-name=Tsotsallm  # name of the Job
SBATCH --output=tsotsallm_output.txt  # output file
SBATCH --partition=sirocco  # Node "sirocco"
SBATCH --gres=gpu:a100:1  # using A100 GPU

# SBATCH --mem=64G  # memory requested

# Start training
python toy_submission/llama_recipes/train.py --model-name NousResearch/Llama-2-7b-hf --hf_rep yvelos/Tsotsallm --fine-tuned-model-name Tsotsallm  --epochs 1 --split train