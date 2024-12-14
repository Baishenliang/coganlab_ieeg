#!/bin/bash
#SBATCH -e out/slurm_%a.err
#SBATCH -a 50-110%20
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --partition=common,scavenger,common-old
#SBATCH --output=out/test%a.out
#conda init bash
#conda activate ieeg
python batch_preproc.py