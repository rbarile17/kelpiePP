# Datasets

The datasets in this repository are DB50K, DB100K, YAGO4-20.
DB50K and DB100K are adaptations of their original versions: [DB50K](https://github.com/bxshi/ConMask), [DB100K](https://github.com/iieir-km/ComplEx-NNE_AER).
Specifically, we provide additional files regarding entity classes and schema.
YAGO4-20 is a sample of the KG [YAGO4](https://yago-knowledge.org/downloads/yago-4). Such sample is then enriched with additional files for entity classes.

Each one includes:

- a README file to check for dataset specific information
- RDF triples in the files `train.txt`, `valid.txt`, `test.txt` where each line is a triple structured as follows:

  ```rdf
  subject'\t'predicate'\t'object
  ```

- entity classes in `entities.csv`
- the schema in one or more files depending on the KG
- the integration of the triples with the schema
- a `reasoned` directory containing:
  - the integrated dataset enriched after reasoning
  - the entity classes including implicit ones obtained through reasoning in `entities.csv`

All such information is available in this directory and Kelpie++ is ready to execute! Anyway, in each dataset specific README we describe the process to obtain the datasets in this repository starting from the existing ones or the full KG in the case of YAGO4-20.

We also provide the diff files listing all the modifications that we made to make the datasets consistent

* `DB50Kentityclasses.txt`
* `DB50KTBox.txt`
* `DB100Kentityclasses.txt`
* `DB100KTBox.txt`
* `DB100Ktraintriples.txt`
