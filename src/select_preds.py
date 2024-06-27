import click

import pandas as pd

from . import DATASETS, ENTITY_DENSITIES, MODELS, PRED_RANKS
from . import PREDICTIONS_PATH, SELECTED_PREDICTIONS_PATH

from . import FIRST
from . import LONG_TAIL

from .dataset import Dataset

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--entity-density", type=click.Choice(ENTITY_DENSITIES))
@click.option("--pred-rank", type=click.Choice(PRED_RANKS))
def main(dataset, model, entity_density, pred_rank):
    dataset_name = dataset
    dataset = Dataset(dataset)

    preds_path = PREDICTIONS_PATH / f"{model}_{dataset_name}.csv"
    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    if pred_rank == FIRST:
        preds = preds[preds["o_rank"] == 1]
    if entity_density == LONG_TAIL:
        entity_to_triples_num = {entity: len(dataset.entity_to_training_triples[entity]) for entity in dataset.entity_ids}
        average_density = sum(entity_to_triples_num.values()) / len(entity_to_triples_num)
        long_tail_entities = [entity for entity, triples_num in entity_to_triples_num.items() if triples_num <= max((average_density / 5), 1)]
        long_tail_entities = [dataset.id_to_entity[id] for id in long_tail_entities]
        preds = preds.loc[preds["s"].isin(long_tail_entities)]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    output_path = SELECTED_PREDICTIONS_PATH / f"{model}_{dataset_name}_{entity_density}_{pred_rank}.csv"
    preds.to_csv(output_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
