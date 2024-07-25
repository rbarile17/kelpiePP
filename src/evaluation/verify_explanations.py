import copy
import click
import json

import numpy
import torch

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from .. import DATASETS, METHODS, MODELS, MODES
from .. import KELPIE
from .. import NECESSARY, SUFFICIENT

from .. import CONFIGS_PATH, MODELS_PATH, RESULTS_PATH

from ..dataset import MANY_TO_ONE, ONE_TO_ONE

from ..explanation_builders.summarization import NO_SUMMARIZATION, SUMMARIZATIONS
from ..prefilters import (
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
)

from ..dataset import Dataset
from ..link_prediction import MODEL_REGISTRY
from ..link_prediction.models import TransE
from ..utils import set_seeds

PREFILTERS = [
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
]

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--mode", type=click.Choice(MODES))
@click.option("--prefilter", type=click.Choice(PREFILTERS), default=TOPOLOGY_PREFILTER)
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS), default=NO_SUMMARIZATION)
def main(
    dataset,
    model,
    mode,
    prefilter,
    summarization
):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    prefilter_short_names = {
        TOPOLOGY_PREFILTER: "bfs",
        TYPE_PREFILTER: "type",
        WEIGHTED_TOPOLOGY_PREFILTER: "wbfs",
    }
    prefilter_short_name = prefilter_short_names[prefilter] if prefilter else "bfs"

    explanations_path = f"{model}_{dataset}_{mode}_{prefilter_short_name}_th20_{summarization}"
    explanations_path = Path(explanations_path)
    explanations_path = RESULTS_PATH / explanations_path
    with open(explanations_path / "output.json", "r") as f:
        explanations = json.load(f)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []
    triple_to_best_rule = {}

    if mode == SUFFICIENT:
        triple_to_convert_set = {}
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)

            entities = explanation["entities_to_convert"]
            entities = [dataset.entity_to_id[entity] for entity in entities]
            triple_to_convert_set[pred] = entities

            if len(explanation["rule_to_relevance"]) > 0:
                tmp = explanation["rule_to_relevance"][0]
                if len(tmp) == 3:
                    _, best_rule, _ = tmp
                else:
                    best_rule, _ = tmp
                best_rule = [dataset.ids_triple(triple) for triple in best_rule]

                triple_to_best_rule[pred] = best_rule
            else:
                triple_to_best_rule[pred] = []

        triples_to_add = []
        triples_to_convert = []

        triple_to_convert_to_added = {}
        for pred in preds:
            s, _, o = pred
            entities = triple_to_convert_set[pred]
            best_rule = triple_to_best_rule[pred]

            cur_triples_to_convert = []
            for entity in entities:
                triple_to_convert = Dataset.replace_entity_in_triple(pred, s, entity)
                cur_triples_to_convert.append(triple_to_convert)
                cur_triples_to_add = Dataset.replace_entity_in_triples(
                    best_rule, s, entity
                )
                triples_to_add.extend(cur_triples_to_add)
                triple_to_convert_to_added[triple_to_convert] = cur_triples_to_add

            triples_to_convert.extend(cur_triples_to_convert)
            triple_to_convert_set[pred] = cur_triples_to_convert

        new_dataset = copy.deepcopy(dataset)

        for s, p, o in triples_to_add:
            if new_dataset.relation_to_type[p] in [MANY_TO_ONE, ONE_TO_ONE]:
                for existing_o in new_dataset.train_to_filter[(s, p)]:
                    new_dataset.remove_training_triple((s, p, existing_o))

        new_dataset.add_training_triples(triples_to_add)

        batch_size = 2048
        if len(triples_to_convert) > batch_size and isinstance(model, TransE):
            batch_start = 0
            results = []
            num_triples = len(triples_to_convert)
            with tqdm(total=num_triples, unit="ex", leave=False) as p:
                while batch_start < num_triples:
                    batch_end = min(len(triples_to_convert), batch_start + batch_size)
                    cur_batch = triples_to_convert[batch_start:batch_end]
                    results += model.predict_triples(numpy.array(cur_batch))
                    batch_start += batch_size
                    p.update(batch_size)
        else:
            results = model.predict_triples(numpy.array(triples_to_convert))
        results = {
            triple: result for triple, result in zip(triples_to_convert, results)
        }

        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)
        hp = model_config["evaluation"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=new_model, hp=optimizer_params)

        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()
        batch_size = 2048
        if len(triples_to_convert) > batch_size and isinstance(new_model, TransE):
            batch_start = 0
            new_results = []
            num_triples = len(triples_to_convert)
            with tqdm(total=num_triples, unit="ex", leave=False) as p:
                while batch_start < num_triples:
                    batch_end = min(len(triples_to_convert), batch_start + batch_size)
                    cur_batch = triples_to_convert[batch_start:batch_end]
                    new_results += new_model.predict_triples(numpy.array(cur_batch))
                    batch_start += batch_size
                    p.update(batch_size)
        else:
            new_results = new_model.predict_triples(numpy.array(triples_to_convert))
        new_results = {
            triple: result for triple, result in zip(triples_to_convert, new_results)
        }

        evaluations = []
        for pred in preds:
            triples_to_convert = triple_to_convert_set[pred]
            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
            }
            conversions = []
            for pred in triples_to_convert:
                result = results[pred]
                new_result = new_results[pred]

                score = result["score"]["tail"]
                rank = result["rank"]["tail"]
                new_score = new_result["score"]["tail"]
                new_rank = new_result["rank"]["tail"]

                conversions.append(
                    {
                        "triples_to_add": [
                            dataset.labels_triple(triple)
                            for triple in triple_to_convert_to_added[pred]
                        ],
                        "score": str(score),
                        "rank": str(rank),
                        "new_score": str(new_score),
                        "new_rank": str(new_rank),
                    }
                )
            evaluation["conversions"] = conversions
            evaluations.append(evaluation)

    elif mode == NECESSARY:
        triple_to_best_rule = defaultdict(list)
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)
            tmp = explanation["rule_to_relevance"][0]
            if len(tmp) == 3:
                _, best_rule, _ = tmp
            else:
                best_rule, _ = tmp
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[pred] = best_rule

        triples_to_remove = []

        for pred in preds:
            triples_to_remove += triple_to_best_rule[pred]

        new_dataset = copy.deepcopy(dataset)

        new_dataset.remove_training_triples(triples_to_remove)

        results = model.predict_triples(numpy.array(preds))
        results = {triple: result for triple, result in zip(preds, results)}
        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)

        hp = model_config["evaluation"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=new_model, hp=optimizer_params)
        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()

        new_results = new_model.predict_triples(numpy.array(preds))
        new_results = {triple: result for triple, result in zip(preds, new_results)}

        evaluations = []
        for pred in preds:
            result = results[pred]
            new_result = new_results[pred]

            score = result["score"]["tail"]
            rank = result["rank"]["tail"]
            new_score = new_result["score"]["tail"]
            new_rank = new_result["rank"]["tail"]

            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
                "rule": [
                    dataset.labels_triple(triple)
                    for triple in triple_to_best_rule[pred]
                ],
                "score": str(score),
                "rank": str(rank),
                "new_score": str(new_score),
                "new_rank": str(new_rank),
            }

            evaluations.append(evaluation)

    with open(explanations_path / "output_end_to_end.json", "w") as outfile:
        json.dump(evaluations, outfile, indent=4)


if __name__ == "__main__":
    main()
