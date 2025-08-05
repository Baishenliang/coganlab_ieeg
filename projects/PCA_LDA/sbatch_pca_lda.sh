#!/bin/bash
#SBATCH -e DCC_out/slurm_%a.err
#SBATCH -a 1-10
#SBATCH --mem=10G
#SBATCH --cpus-per-task=60
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/test%a.out
#conda init bash
#conda activate ieeg
python pca_lda.py