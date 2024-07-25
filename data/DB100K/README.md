
# DB100K

In this repository DB100K is an adaption of the dataset provided in the original [DB100K repository](https://github.com/iieir-km/ComplEx-NNE_AER).
The original repository solely contains the files of RDF triples
Here additional files regarding classes and schema are provided.
Moreover, in the RDF triples the IDs are modified.

We describe the process to obtain the adapted dataset.
Download the [files of RDF triples](https://github.com/iieir-km/ComplEx-NNE_AER/tree/master/datasets/DB100K).

The triples are defined using entity IDs from Wikidata instead of DBpedia, so, replace the IDs (output in the same files)

```python
python -m src.semantic.DB100K.retrieve_mapping
python -m src.semantic.DB100K.replace_mapped_entities
```

We ketp a copy of the original files in `wikidata triples`.

Then, generate the additional files through the following steps.

Download the [`DBpedia schema`](https://databus.dbpedia.org/ontologies/dbpedia.org/ontology--DEV/2023.11.27-081000/ontology--DEV_type=parsed.owl).

Retrieve the entity classes (output in `entities.csv`)

```python
python -m src.semantic.retrieve_classes
```

Integrate the triples, the entity classes and the schema (output in in `DB100K.owl`)

```python
python -m src.semantic.load_ontology
```

Run HermiT to obtain the enriched version of the KG (output in `reasoned/DB100K_reasoned.owl`)

```python
python -m src.semantic.reason
```

Query the enriched KG to obtain the entity classes also including implicit ones found through reasoning (output in `reasoned/entities.csv`)

```python
python -m src.semantic.reasoned_onto_get_classes
```

However, executing this process the reasoner threw an error about inconsistency of the KG. We manually repaired it modifying `entities.csv`, `train.txt` and `DBpedia.owl` and we executed the reasoner again. In this repository we provide the repaired version of the files.
`DB100K.owl` and `DB100K_reasoned.owl` are too big for GitHub, so we uploaded them [here](https://figshare.com/s/ece7729bc1a7a8f64b26).
