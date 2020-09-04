#!/bin/bash

#PBS -q celltypes
#PBS -l walltime=24:00:00
#PBS -l nodes=16:ppn=16
#PBS -l mem=150g
#PBS -N dend_opt
#PBS -e /dev/null
#PBS -o /dev/null
#PBS -r n
#PBS -m bea

cd $PBS_O_WORKDIR

set -e

source activate ateam_opt

# Relaunch batch job if not finished
qsub -W depend=afternotok:$PBS_JOBID batch_job.sh


# Configure ipython profile
PWD=$(pwd)
export IPYTHONDIR=$PWD/.ipython
ipython profile create
file $IPYTHONDIR
export IPYTHON_PROFILE=pbs.$PBS_JOBID

# Start ipcontroller and engines
ipcontroller --init --ip='*' --nodb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 30
file $IPYTHONDIR/profile_$IPYTHON_PROFILE
mpiexec -n 256 ipengine --timeout=3000 --profile=${IPYTHON_PROFILE} &
sleep 30


# Run Optimization
pids=""
for seed in 1 2 3 4 ; do
    python Optim_Main.py             \
        --seed ${seed}             \
        --input_json stage_job_config.json &
    pids+="$! "
done

wait $pids
