#!/bin/bash

dataset=$1
model=$2
mode=$3
method=$4
summarization=$5
entity_density=$6
pred_rank=$7

python -m src.explain                        --dataset $dataset --model $model --mode $mode --method $method --summarization $summarization --entity-density $entity_density --pred-rank $pred_rank
python -m src.evaluation.verify_explanations --dataset $dataset --model $model --mode $mode --method $method --summarization $summarization --entity-density $entity_density --pred-rank $pred_rank
python -m src.evaluation.compute_metrics     --dataset $dataset --model $model --mode $mode --method $method --summarization $summarization --entity-density $entity_density --pred-rank $pred_rank
