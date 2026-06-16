#!/bin/bash
# Clean baseline (GENERator-800M). Submit from the repo root and override the
# account/partition/nodes for your cluster, e.g.:
#   cd pretraining_GENERator
#   sbatch -A <account> -p <partition> -N <nodes> scripts/submit_clean.sh
#SBATCH -J clean_800m
#SBATCH -o logs/clean_%j.out
#SBATCH -e logs/clean_%j.err
#SBATCH -p h100
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH -c 96
#SBATCH -t 48:00:00

# Repo root = the directory you submitted from (override with SCRIPT_DIR).
export SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/experiments/clean_800m.yaml}"
source "${SCRIPT_DIR}/scripts/submit_train.sh"
