
# DB50K

In this repository DB100K is an adaption of the dataset provided in the original [DB50K repository](https://github.com/bxshi/ConMask).
The original repository solely contains the files of RDF triples
Here additional files regarding classes and schema are provided.

We describe the process to obtain the adapted dataset.
Download the [files of RDF triples](https://drive.google.com/drive/folders/1YBKw4nOnbscpDeTD_gWxfcpHRFG3MY20).

Download the [`DBpedia schema`](https://databus.dbpedia.org/ontologies/dbpedia.org/ontology--DEV/2023.11.27-081000/ontology--DEV_type=parsed.owl).

Retrieve the entity classes (output in `entities.csv`)

```python
python -m src.semantic.retrieve_classes
```

Integrate the triples, the entity classes and the schema (output in in `DB50K.owl`)

```python
python -m src.semantic.load_ontology
```

Run HermiT to obtain the enriched version of the KG (output in `reasoned/DB50K_reasoned.owl`)

```python
python -m src.semantic.reason
```

Query the enriched KG to obtain the entity classes also including implicit ones found through reasoning (output in `reasoned/entities.csv`)

```python
python -m src.semantic.reasoned_onto_get_classes
```

However, executing this process the reasoner threw an error about inconsistency of the KG. We manually repaired it modifying `entities.csv`, `train.txt` and `DBpedia.owl` and we executed the reasoner again. In this repository we provide the repaired version of the files.
