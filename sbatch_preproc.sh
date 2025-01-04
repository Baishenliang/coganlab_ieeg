#!/bin/bash
#SBATCH -e data/DCCbatchout/slurm_%a.err
#SBATCH -a 20-85%20
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20
#SBATCH --partition=common,scavenger
#SBATCH --output=data/DCCbatchout/test%a.out
#conda init bash
#conda activate ieeg
python batch_preproc.py