#!/bin/bash
#SBATCH -e DCC_out/%a.err
#SBATCH -a 3-40
#SBATCH --mem=600G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/%a.out
module load R
Rscript lme_encode.R
