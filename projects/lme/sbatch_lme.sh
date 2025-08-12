#!/bin/bash
#SBATCH -e DCC_out/slurm_%a.err
#SBATCH -a 1-20
#SBATCH --mem=100G
#SBATCH --cpus-per-task=30
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/test%a.out
module load R
Rscript lme_encode.R
