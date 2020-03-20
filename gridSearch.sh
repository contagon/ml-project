#!/bin/env bash

#SBATCH --array=0-50
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=2-10:0:0
#SBATCH --cpus-per-task=16
#SBATCH -C 'rhel7'
#SBATCH -J "rr_rec"   # job name
#SBATCH --nodes=1   # limit to one node


rec_values=( sum fr rf )
data_values=( U Uhat )

trial=${SLURM_ARRAY_TASK_ID}
rec=${rec_values[$(( trial % ${#rec_values[@]} ))]}
trial=$(( trial / ${#rec_values[@]} ))
data=${data_values[$(( trial % ${#data_values[@]} ))]}


## use ${rec}, ${data} below
module load miniconda3
source activate ml

python gridSearch.py --data ${data} --rec ${rec} --n_jobs 4 --rdr kNN