#!/bin/bash
#SBATCH -e data/DCCbatchout/slurm_%a.err
#SBATCH -a 115-120%20
#SBATCH --mem=800G
#SBATCH --cpus-per-task=80
#SBATCH --partition=common,scavenger
#SBATCH --output=data/DCCbatchout/test%a.out
#conda init bash
#conda activate ieeg
python batch_preproc.py