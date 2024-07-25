from .relevance_engines import PostTrainingEngine


class Pipeline:
    def __init__(self, dataset, prefilter, builder):
        self.dataset = dataset
        self.prefilter = prefilter
        self.builder = builder

        self.engine = self.builder.engine
        self.model = self.engine.model

    def explain(self, pred, prefilter_k=-1):
        if isinstance(self.engine, PostTrainingEngine):
            self.engine.set_cache()
        print("\tRunning prefilter...")
        return self.prefilter.select_triples(pred=pred, k=prefilter_k)


class NecessaryPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)

    def explain(self, pred, prefilter_k=-1):
        filtered_triples = super().explain(pred=pred, prefilter_k=prefilter_k)

        result = self.builder.build_explanations(pred, filtered_triples)

        return result


class SufficientPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder, criage):
        super().__init__(dataset, prefilter, builder)
        self.criage = criage

    def fail(self, pred):
        return  {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": [],
            "#relevances": 0,
            "execution_time": 0,
            "entities_to_convert": []
        }

    def explain(self, pred, prefilter_k=50, to_convert_k=10):
        self.engine.select_entities_to_convert(pred, to_convert_k, 200, criage=self.criage)
        if self.engine.entities_to_convert == []:
            return  self.fail(pred)

        filtered_triples = super().explain(pred, prefilter_k)

        result = self.builder.build_explanations(pred, filtered_triples)
        entities_to_conv = self.engine.entities_to_convert
        entities_to_conv = [self.dataset.id_to_entity[x] for x in entities_to_conv]

        result["entities_to_convert"] = entities_to_conv

        return result
