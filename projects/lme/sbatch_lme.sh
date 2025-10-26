#!/bin/bash
#SBATCH -e DCC_out/%a.err
#SBATCH -a 1-4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=30
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/%a.out
module load R
Rscript lme_encode.R
