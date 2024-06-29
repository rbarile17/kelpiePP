#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-3

config=./scripts/config.txt

method=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
mode=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $3}' $config)
dataset=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $4}' $config)
summarization=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $5}' $config)
entity_density=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $6}' $config)
pred_rank=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $7}' $config)

if [[ $method = "criage" ]]; then
    ./scripts/explain.sh $dataset ConvE   $mode $method $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset ComplEx $mode $method $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
fi
if [[ $method = "data_poisoning" ]]; then
    ./scripts/explain.sh $dataset ConvE   $mode $method $summarization > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset ComplEx $mode $method $summarization > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $mode $method $summarization > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}.log" 2>&1 & 
fi
if [[ $method = "kelpie" ]]; then
    ./scripts/explain.sh $dataset ConvE   $mode $method $summarization $entity_density $pred_rank > "./logs/explain/${method}_ConvE_${dataset}_${mode}_${summarization}_${entity_density}_${pred_rank}.log"   2>&1 &
    ./scripts/explain.sh $dataset ComplEx $mode $method $summarization $entity_density $pred_rank > "./logs/explain/${method}_ComplEx_${dataset}_${mode}_${summarization}_${entity_density}_${pred_rank}.log" 2>&1 &
    ./scripts/explain.sh $dataset TransE  $mode $method $summarization $entity_density $pred_rank > "./logs/explain/${method}_TransE_${dataset}_${mode}_${summarization}_${entity_density}_${pred_rank}.log"  2>&1 & 
fi
wait
