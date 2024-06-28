from itertools import product

from .dataset import Dataset

class FactBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.load_summary()

    def build_triples(self, entity, pred):
        qe = self.dataset.get_quotient_entity(entity)
        qtriples = self.dataset.quotient_entity_to_triples[qe]

        qtriples = [
            (self.dataset.quotient_entities[qs], p, self.dataset.quotient_entities[qo])
            for qs, p, qo in qtriples
        ]

        triples = [(entity, p, o) for _, p, qo in qtriples for o in qo]
        triples = [t for t in triples if t != pred]

        return triples
