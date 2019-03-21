from allennlp.common.util import JsonDict
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('conll_tagger')
class ConllTaggerPredictor(Predictor):
    """"
    Predictor wrapper for the Tagger

    This class simply dumps the tags out in a convenient format
    for CoNLL tagging datasets.
    """
    def dump_line(self, outputs: JsonDict) -> str:
        return "\n".join(outputs["tags"]) + "\n\n"
