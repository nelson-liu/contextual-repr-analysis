from typing import Iterable
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance  # noqa


@Vocabulary.register("vocabulary_with_lm")
class VocabularyWithLM(Vocabulary):
    """
    Taken from Calypso:
    https://github.com/allenai/calypso/blob/ac934b6881787387581efaa8a646531278010652/calypso/allennlp_bridge.py#L31-L48

    Augment the allennlp Vocabulary with a pre-trained LM.
    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['Instance'] = None):
        # set the LM piece
        lm_vocab_file = params.pop('lm_vocab_file')
        oov_token = params.pop('oov_token')
        namespace = params.pop('namespace', 'lm')
        vocab = super(VocabularyWithLM, cls).from_params(params, instances)
        # if `lm_vocab_file` is a URL, redirect to the cache
        lm_vocab_file = cached_path(lm_vocab_file)
        vocab.set_from_file(lm_vocab_file, is_padded=True, oov_token=oov_token,
                            namespace=namespace)
        return vocab
