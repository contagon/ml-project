#!/bin/env bash

#SBATCH --array=0-3
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=1-01:0:0
#SBATCH --cpus-per-task=8
#SBATCH -J "knn"   # job name
#SBATCH --nodes=1   # limit to one node


sc_values=( int com )
data_values=( U Uhat )

trial=${SLURM_ARRAY_TASK_ID}
sc=${sc_values[$(( trial % ${#sc_values[@]} ))]}
trial=$(( trial / ${#sc_values[@]} ))
data=${data_values[$(( trial % ${#data_values[@]} ))]}


## use ${sc}, ${data} below
module load miniconda3
source activate ml

#create profile so we can find cluster, also start cluster
profile=job_${SLURM_JOB_ID}
ipython profile create ${profile}
sleep 10
ipcluster start --profile=${profile} -n 8 &
#ipcontroller --ip="*" --profile=${profile} &
#srun ipengine --profile=${profile} --location=$(hostname) &
echo "Cluster Started"
sleep 30


echo "Starting py file..."
python userGridSearch.py --data ${data} --sc ${sc} --n_jobs -1 --rdr kNN --profile ${profile}
