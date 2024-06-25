from .dataset import Dataset

class FactBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.load_summary()

    def build_triples(self, entity):
        qe = self.dataset.get_quotient_entity(entity)
        qtriples = self.dataset.quotient_entity_to_triples[qe]

        triples = [
            (s, p, o)
            for qs, p, qo in qtriples
            for s in qs
            for o in qo
        ]

        return triples


if __name__ == "__main__":
    dataset = Dataset(dataset="DB100K")
    fact_builder = FactBuilder(dataset)
    fact_builder.build_facts(31500)
