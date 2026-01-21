#!/bin/bash
#SBATCH -e data/DCCbatchout/slurm_%a.err
#SBATCH -a 100-140
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --partition=common,scavenger
#SBATCH --output=data/DCCbatchout/test%a.out
#conda init bash
#conda activate ieeg
python batch_preproc.py