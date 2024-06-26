import click

import pandas as pd

from . import DATASETS, MODELS
from . import PREDICTIONS_PATH, SELECTED_PREDICTIONS_PATH

from .dataset import Dataset

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--long-tail/--all_entities", default=False)
@click.option("--correct/--correct-and-wrong", default=True)
def main(dataset, model, long_tail, correct):
    dataset_name = dataset
    dataset = Dataset(dataset)

    preds_path = PREDICTIONS_PATH / f"{model}_{dataset_name}.csv"
    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    if correct:
        preds = preds[preds["o_rank"] == 1]
    if long_tail:
        entity_to_triples_num = {entity: len(dataset.entity_to_training_triples[entity]) for entity in dataset.entity_ids}
        average_density = sum(entity_to_triples_num.values()) / len(entity_to_triples_num)
        long_tail_entities = [entity for entity, triples_num in entity_to_triples_num.items() if triples_num < (average_density / 5)]
        long_tail_entities = [dataset.id_to_entity[id] for id in long_tail_entities]
        preds = preds.loc[preds["s"].isin(long_tail_entities)]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    output_path = SELECTED_PREDICTIONS_PATH / f"{model}_{dataset_name}_{'long-tail-entities' if long_tail else 'all-entities'}_{'correct' if correct else 'correct-and-wrong'}.csv"
    preds.to_csv(output_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
