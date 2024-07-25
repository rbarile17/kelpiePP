# Kelpie++

This repository contains the official code for the paper ["Explanation of Link Predictions on Knowledge Graphs via Levelwise Filtering and Graph Summarization"](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640175.pdf) that presents Kelpie++.

Kelpie++ is a post-hoc local explainability tool for Link Prediction (LP) on Knowledge Graphs (KGs) through embedding-based models. It explains a prediction $\langle s, p, o\rangle$ by identifying the smallest set of facts that enabled such inference.

## Explanation Scenarios

Kelpie++ generates two types of explanation:

* **necessary**: triples without which the model can not make the prediction
* **sufficient**: triples leading to replicate the prediction on other entities

## Architecture

Kelpie++ is structured around three components.

* **Pre-Filter**: selects the most useful triples featuring _s_
* **Explanation Builder**: combines the pre-filtered triples into candidate explanations and identifies sufficiently relevant ones
* **Relevance Engine**: estimates the _relevance_ of a candidate explanation

Kelpie++ introduces two extensions of Kelpie:

* a utility measure enriched with _semantic similarity_ in the **Pre-Filter**
* a summarization step in the **Explanation Builder** with three different (incremental) levels of granularity

## Additional Information

Check also:

* [models README](./models/README.md)
* [data README](./data/README.md)

## Getting started

```bash
# Clone the repository
git clone https://github.com/rbarile17/kelpiePP.git

# Navigate to the repository directory
cd kelpiePP

# Install the required dependencies
pip install -r requirements.txt
```

## Usage Instructions

Follow the steps in this section to run the Kelpie++ pipeline.
All commands require the parameters `dataset`, `model`.
Find in `data` the datasets DB50K, DB100K, and YAGO4-20.
You can also experiment with your own datasets! (structured as explained in data [README](./data/README.md))
Instead, the supported models are: ComplEx, ConvE, TransE.
You can extend the class `Model` to add models!

Run the commands with the --help option to inspect the possible values for all the parameters!

### Preliminary

Create a `<model>_<dataset>.json` in `configs` specifying the config for training, explanation, and evaluation. Check out [configs README](./configs/README.md) for information and examples on configurations.

### Link Prediction

#### Train a model

```python
python -m src.link_prediction.train --dataset <dataset> --model <model> --valid <validation_epochs>
```

`<valid>` is the frequency (in epochs) of evaluation of the model on the validation set to determine whether to apply early stopping

#### Make **predictions** (compute the rank of test triples)

```python
python -m src.link_prediction.test --dataset <dataset> --model <model>
```

#### Select and sample 100 **correct** predictions (top-ranked test triples)

```python
python -m src.extract_correct_predictions --dataset <dataset> --model <model>
```

### Explanation and Evaluation

The commands in this section also require:

* `<mode>`: **necessary** or **sufficient**
* `<prefilter>` is the utility measure to adopt in the **Pre-Filter**, choose between:
  * `topology_based` (default)
  * `weighted_topology_based`: enriched with _semantic similarity_
* `<summarization>` is the summarization solution to adopt in the **Explanation Builder**, choose between the following values (ordered by increasing granularity):
  * `simulation`
  * `bisimulation_d1`
  * `bisimulation`
  * `no` (default)

#### Generate explanations

```python
python -m src.explain --dataset <dataset> --model <model> --mode <mode> --prefilter <prefilter> --summarization <summarization>
```

#### Assses the impact of explanations on ranks

```python
python -m src.evaluation.verify_explanations --dataset <dataset> --model <model> --mode <mode> --prefilter <prefilter> --summarization <summarization>
```

#### Compute evaluation metrics

```python
python -m src.evaluation.compute_metrics --dataset <dataset> --model <model> --mode <mode> --prefilter <prefilter> --summarization <summarization>
```

## Experiments

To reproduce the experiments in the paper use:

* the datasets [DB50K](./data/DB50K), [DB100K](./data/DB100K), [YAGO4-20](./data/YAGO4-20)
* our [configs](./experiments/.configs) specifying the hyperparameters found as described in Appendix B of the paper
* our [pre-trained models](https://figshare.com/s/83a1022644a5183244d8)
* our [sampled correct preds](./experiments/.correct_preds)

We executed the experiments on an environment configured as follows:

* OS: Pop!_OS 22.04 LTS
* Python 3.11.3
* CUDA Version: 11.7
* Driver Version: 535.113.01

## Repository Structure

  ├── README.md          <- The top-level README for developers using this project.
  ├── data
  ├── notebooks          <- Jupyter notebooks.
  ├── requirements.txt   <- The requirements file for reproducing the environment
  │
  └── src                <- Source code.
