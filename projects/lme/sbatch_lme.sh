#!/bin/bash
#SBATCH -e DCC_out/%a.err
#SBATCH -a 1,2,6
#SBATCH --mem=320G
#SBATCH --cpus-per-task=40
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/%a.out
module load R
Rscript lme_encode.R
