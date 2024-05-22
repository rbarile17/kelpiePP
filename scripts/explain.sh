#!/bin/bash

dataset=$1
model=$2
mode=$3
method=$4

python -m src.explain --dataset $dataset --model $model --mode $mode --method $method
python -m src.evaluation.verify_explanations --dataset $dataset --model $model --mode $mode
python -m src.evaluation.compute_metrics --dataset $dataset --model $model --mode $mode
