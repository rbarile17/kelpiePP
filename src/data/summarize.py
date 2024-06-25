import click

import pandas as pd

from .. import DATASETS
from .. import DATA_PATH

from ..dataset import Dataset
from ..summarization import Simulation

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    dataset = Dataset(dataset=dataset)
    summarization = Simulation(dataset)
    q_triples = summarization.summarize()

    q_triples = (
        ([dataset.id_to_entity[s] for s in sq], dataset.id_to_relation[p], [dataset.id_to_entity[o] for o in oq])
        for sq, p, oq in q_triples
    )

    output_path = DATA_PATH / dataset.name / "train_summarization.txt"
    pd.DataFrame(q_triples).to_csv(output_path, index=False, header=False, sep="\t")

if __name__ == "__main__":
    main()
