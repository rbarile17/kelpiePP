import itertools

import numpy as np

from tqdm import tqdm

class QuotientGraph:
    def build_quotient_graph(self, partition, condition=any):
        triples = self.dataset.training_triples
        ss = triples[:, 0]
        os = triples[:, 2]
        # test = lambda p: condition(any((u, p, v) in triples for v in V) for u in U)

        pairs = list(itertools.product(partition, repeat=2))

        q_triples = []
        for U, V in tqdm(pairs):
            mask_s = np.isin(ss, U, kind='table')
            mask_o = np.isin(os, V, kind='table')

            ps = set(triples[np.logical_and(mask_s, mask_o), 1])

            q_triples.extend([(U, p, V) for p in ps])

        return q_triples

    # def plot(self, graph, filename="graph", format="svg", quotient=False):
    #     agraph = to_agraph(graph)
    #     agraph.graph_attr["rankdir"] = "LR"
    #     agraph.graph_attr["pad"] = 0.01

    #     for node in agraph.nodes():
    #         node.attr["shape"] = "rectangle"
    #         node.attr["style"] = "rounded"
    #     for edge in agraph.edges():
    #         edge.attr["arrowsize"] = 0.3
    #         edge.attr["color"] = "red"

    #     agraph.draw(f"pictures/{filename}.{format}", prog="dot", format=format)

    def set_quotient_triple_to_triples(self, q_triples, filter):
        self.quotient_triple_to_triples = {}

        for s_part, p, o_part in q_triples:
            triples = [(s, p, o) for s in s_part for o in o_part]
            triples = [triple for triple in triples if triple in filter]

            if len(triples) > 0:
                self.quotient_triple_to_triples[(s_part, p, o_part)] = triples

    def map_rule(self, rule):
        triples = []
        for q_triple in rule:
            triples += self.quotient_triple_to_triples[q_triple]
        return triples
