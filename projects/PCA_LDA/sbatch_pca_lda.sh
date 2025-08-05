#!/bin/bash
#SBATCH -e ~/workspace/PCA_LDA_results/slurm_%a.err
#SBATCH -a 1-30%20
#SBATCH --mem=30G
#SBATCH --cpus-per-task=20
#SBATCH --partition=common,scavenger
#SBATCH --output=~/workspace/PCA_LDA_results/test%a.out
#conda init bash
#conda activate ieeg
python pca_lda.py