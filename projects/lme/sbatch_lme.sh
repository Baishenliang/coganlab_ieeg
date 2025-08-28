#!/bin/bash
#SBATCH -e DCC_out/%a.err
#SBATCH -a 1-10
#SBATCH --mem=60G
#SBATCH --cpus-per-task=30
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/%a.out
module load R
Rscript lme_encode.R
