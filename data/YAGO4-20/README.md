# YAGO4-20

Download [YAGO4](https://yago-knowledge.org/data/yago4/en/2020-02-24/).

Load the files `yago-wd-schema.nt` and `yago-wd-facts.nt` and `yago-wd-class.nt` in [GraphDB](https://www.ontotext.com/products/graphdb/).

Execute all the cells in the [YAGO 4 notebook](`../../notebooks/yago4.ipynb`) to sample the complete KG and split such samples in three files, thus obtaining `train.txt`, `valid.txt`, `test.txt`.
Execute all the cells in the [YAGO 4 schema notebook](`../../notebooks/yago4_onto.ipynb`) to obtain `YAGO4-20.nt`.
The notebooks assume that GraphDB is running on localhost:7200.

Run HermiT to obtain the enriched version of the KG (output in YAGO4-20/YAGO4-20_reasoned.nt)

```python
python -m src.semantic.reason
```

Query the enriched KG to obtain the entity classes also including implicit ones found through reasoning (output in /reasoned/entities.csv)

```python
python -m src.semantic.reasoned_onto_get_classes_yago
```

However, executing this process the reasoner threw an error about unsatisfiability of the KG. We manually repaired it by modifying some axioms in `yago-wd-class.nt`. You can download the [repaired file](https://figshare.com/s/ece7729bc1a7a8f64b26).
