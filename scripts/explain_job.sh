#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

method=kelpie
dataset=DB100K
summarization=no
mode=necessary

./scripts/explain.sh $dataset TransE  $mode $method $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}_long_tail.log" 2>&1 & 
wait
