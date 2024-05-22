#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

model=$1
mode=$2

python -m src.explain --dataset FB15k-237 --model TransE --mode necessary
