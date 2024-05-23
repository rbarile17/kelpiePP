#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-24

config=./scripts/config.txt

method=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
dataset=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
summarization=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
mode=$1

if [ $method != "criage" ]; then
    ./scripts/explain.sh $dataset TransE  $mode $method $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log" 2>&1 & 
fi
./scripts/explain.sh $dataset ConvE   $mode $method $summarization> "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log" 2>&1 &
./scripts/explain.sh $dataset ComplEx $mode $method $summarization> "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
wait
