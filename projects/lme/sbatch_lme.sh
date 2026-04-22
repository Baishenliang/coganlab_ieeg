#!/bin/bash
#SBATCH -e DCC_out/%a.err
#SBATCH -a 2,6
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=80
#SBATCH --partition=common,scavenger
#SBATCH --output=DCC_out/%a.out
module load R
Rscript lme_encode.R
