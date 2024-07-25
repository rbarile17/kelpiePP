import click

import pandas as pd

from . import DATASETS, MODELS
from . import PREDICTIONS_PATH, CORRECT_PREDICTIONS_PATH

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    preds_path = PREDICTIONS_PATH / f"{model}_{dataset}.csv"
    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    preds = preds[preds["o_rank"] == 1]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    output_path = CORRECT_PREDICTIONS_PATH / f"{model}_{dataset}.csv"
    preds.to_csv(output_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
