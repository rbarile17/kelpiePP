import click
import json

from pathlib import Path

from .. import DATASETS, MODELS, MODES
from .. import CONFIGS_PATH
from .. import NECESSARY, SUFFICIENT

def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--mode", type=click.Choice(MODES))
def main(dataset, model, mode):
    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json" 
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    explanations_path = Path(model_config["explanations_path"])
    explanations_filepath = explanations_path / "output_end_to_end.json"

    with open(explanations_filepath, "r") as input_file:
        triple_to_details = json.load(input_file)
    if mode == NECESSARY:
        ranks = [float(details["rank"]) for details in triple_to_details]
        new_ranks = [float(details["new_rank"]) for details in triple_to_details]
    elif mode == SUFFICIENT:
        ranks = [
            float(conversion["rank"])
            for details in triple_to_details
            for conversion in details["conversions"]
        ]
        new_ranks = [
            float(conversion["new_rank"])
            for details in triple_to_details
            for conversion in details["conversions"]
        ]

    original_mrr, original_h1 = mrr(ranks), hits_at_k(ranks, 1)
    new_mrr, new_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
    mrr_delta = round(new_mrr - original_mrr, 3)
    h1_delta = round(new_h1 - original_h1, 3)

    explanations_filepath = explanations_path / "output.json"
    with open(explanations_filepath, "r") as input_file:
        explanations = json.load(input_file)
    rels = [x["#relevances"] for x in explanations]
    rels = sum(rels)

    metrics = {}
    metrics["delta_h1"] = h1_delta
    metrics["delta_mrr"] = mrr_delta
    metrics["rels"] = rels

    metrics_file = explanations_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
