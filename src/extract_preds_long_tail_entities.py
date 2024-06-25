import click

import pandas as pd

from . import DATASETS, MODELS
from . import PREDICTIONS_PATH, CORRECT_PREDICTIONS_PATH

from .dataset import Dataset

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    dataset_name = dataset
    dataset = Dataset(dataset_name)

    entity_to_triples_num = {entity: len(dataset.entity_to_training_triples[entity]) for entity in dataset.entity_ids}
    average_density = sum(entity_to_triples_num.values()) / len(entity_to_triples_num)
    long_tail_entities = [entity for entity, triples_num in entity_to_triples_num.items() if triples_num < (average_density / 5)]
    long_tail_entities = [dataset.id_to_entity[id] for id in long_tail_entities]
    preds_path = PREDICTIONS_PATH / f"{model}_{dataset_name}.csv"
    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    # preds = preds[preds["o_rank"] == 1]
    preds = preds.loc[preds["s"].isin(long_tail_entities)]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    output_path = CORRECT_PREDICTIONS_PATH / f"{model}_{dataset_name}_long_tail.csv"
    preds.to_csv(output_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
