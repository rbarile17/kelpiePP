import click
import json

import torch

from pathlib import Path

from . import DATASETS, METHODS, MODELS, MODES
from . import CONFIGS_PATH, MODELS_PATH, CORRECT_PREDICTIONS_PATH, RESULTS_PATH
from . import KELPIE, DATA_POISONING, CRIAGE
from . import NECESSARY, SUFFICIENT
from .link_prediction import MODEL_REGISTRY
from .prefilters import (
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
)

from .dataset import Dataset
from .explanation_builders import CriageBuilder, DataPoisoningBuilder, StochasticBuilder
from .explanation_builders.summarization import NO_SUMMARIZATION, SUMMARIZATIONS
from .pipeline import NecessaryPipeline, SufficientPipeline
from .prefilters import (
    CriagePreFilter,
    NoPreFilter,
    TopologyPreFilter,
    TypeBasedPreFilter,
    WeightedTopologyPreFilter,
)
from .relevance_engines import (
    NecessaryCriageEngine,
    SufficientCriageEngine,
    NecessaryDPEngine,
    SufficientDPEngine,
    NecessaryPostTrainingEngine,
    SufficientPostTrainingEngine,
)
from .utils import set_seeds

PREFILTERS = [
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
]


def build_pipeline(model, dataset, hp, mode, method, prefilter, xsi, summarization):
    prefilter_map = {
        TOPOLOGY_PREFILTER: TopologyPreFilter,
        TYPE_PREFILTER: TypeBasedPreFilter,
        NO_PREFILTER: NoPreFilter,
        WEIGHTED_TOPOLOGY_PREFILTER: WeightedTopologyPreFilter,
    }

    if mode == NECESSARY:
        if method == CRIAGE:
            prefilter = CriagePreFilter(dataset)
            engine = NecessaryCriageEngine(model, dataset)
            builder = CriageBuilder(engine)
        elif method == DATA_POISONING:
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = NecessaryDPEngine(model, dataset, hp["lr"])
            builder = DataPoisoningBuilder(engine)
        elif method == KELPIE:
            DEFAULT_XSI_THRESHOLD = 5
            xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = NecessaryPostTrainingEngine(model, dataset, hp)
            builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = NecessaryPipeline(dataset, prefilter, builder)
    elif mode == SUFFICIENT:
        criage = False
        if method == CRIAGE:
            prefilter = CriagePreFilter(dataset)
            engine = SufficientCriageEngine(model, dataset)
            builder = CriageBuilder(engine)
            criage = True
        elif method == DATA_POISONING:
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = SufficientDPEngine(model, dataset, hp["lr"])
            builder = DataPoisoningBuilder(engine)
        elif method == KELPIE:
            DEFAULT_XSI_THRESHOLD = 0.9
            xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = SufficientPostTrainingEngine(model, dataset, hp)
            builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = SufficientPipeline(dataset, prefilter, builder, criage=criage)

    return pipeline


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option(
    "--preds",
    type=click.Path(exists=True),
    help="Path of the predictions to explain.",
)
@click.option(
    "--coverage",
    type=int,
    default=10,
    help="Number of entities to convert (sufficient mode only).",
)
@click.option(
    "--skip",
    type=int,
    default=-1,
    help="Number of predictions to skip.",
)
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--mode", type=click.Choice(MODES), required=True)
@click.option(
    "--relevance_threshold",
    type=float,
    help="The relevance acceptance threshold.",
)
@click.option("--prefilter", type=click.Choice(PREFILTERS), default=TOPOLOGY_PREFILTER)
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS), default=NO_SUMMARIZATION)
@click.option(
    "--prefilter_threshold",
    type=int,
    default=20,
    help=f"The number of triples to select in pre-filtering.",
)
def main(
    dataset,
    model,
    preds,
    coverage,
    method,
    mode,
    prefilter,
    relevance_threshold,
    prefilter_threshold,
    summarization,
    skip
):
    set_seeds(42)

    model_config_file = CONFIGS_PATH / f"{model}_{dataset}.json"
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model_name = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model_name}_{dataset}.pt")

    prefilter_short_names = {
        TOPOLOGY_PREFILTER: "bfs",
        TYPE_PREFILTER: "type",
        WEIGHTED_TOPOLOGY_PREFILTER: "wbfs",
    }
    prefilter_short_name = prefilter_short_names[prefilter] if prefilter else "bfs"
    method = method if method else "kelpie"
    output_dir = f"{method}_{model}_{dataset}_{mode}_{summarization}"

    print("Reading preds...")
    if preds is None:
        preds = CORRECT_PREDICTIONS_PATH / f"{model}_{dataset}.csv"
    with open(preds, "r") as preds:
        preds = [x.strip().split("\t") for x in preds.readlines()]

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model_name]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    pipeline_hps = model_config["kelpie"]
    if method == DATA_POISONING and model_name == "TransE":
        pipeline_hps = model_config["data_poisoning"]
    pipeline = build_pipeline(
        model,
        dataset,
        pipeline_hps,
        mode,
        method,
        prefilter,
        relevance_threshold,
        summarization,
    )

    Path(RESULTS_PATH / output_dir).mkdir(exist_ok=True)

    explanations = []
    for i, pred in enumerate(preds):
        if i <= skip:
            continue
        s, p, o = pred
        print(f"\nExplaining pred {i}: <{s}, {p}, {o}>")
        pred = dataset.ids_triple(pred)
        explanation = pipeline.explain(pred=pred, prefilter_k=prefilter_threshold)

        explanations.append(explanation)

        output_path = RESULTS_PATH / output_dir / "output.json"
        with open(output_path, "w") as f:
            json.dump(explanations, f, indent=4)

if __name__ == "__main__":
    main()
