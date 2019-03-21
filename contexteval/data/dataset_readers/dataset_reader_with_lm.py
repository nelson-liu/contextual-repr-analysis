from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common import Registrable

from contexteval.data.fields import BidirectionalLanguageModelField


@DatasetReader.register("datasetreader_with_lm")
class DatasetReaderWithLM(DatasetReader):
    """
    Taken from calypso:
    https://github.com/allenai/calypso/blob/ac934b6881787387581efaa8a646531278010652/calypso/allennlp_bridge.py#L134-L168

    This dataset reader augments a given dataset reader with a TokenIndexer and adds a
    BidirectionalLanguageModelField used for multi-tasking with a language modeling objective
    at the "lm_targets" key. To make this work, it also adds a TextField at the "tokens" key.

    NOTE: This can only really be constructed from_params.
    """
    @classmethod
    def from_params(cls, params):
        subreader_type = params.pop("subreader_type")
        # get the class definition for the subreader
        subreader_cls = Registrable._registry[DatasetReader][subreader_type]
        # build the token indexer beforehand
        all_token_indexer_params = params.pop('token_indexers', {})
        token_indexers = {}
        for token_indexer_name, token_indexer_params in all_token_indexer_params.items():
            token_indexers[token_indexer_name] = TokenIndexer.from_params(token_indexer_params)

        # Create a new class that adds the BidirectionalLanguageModelField
        # and a TextField
        class ForReturn(subreader_cls):
            _token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

            def text_to_instance(self, *args, **kwargs):
                instance = super().text_to_instance(*args, **kwargs)
                if 'tokens' in kwargs:
                    tokens = kwargs['tokens']
                else:
                    tokens = args[0]

                # Make the TextField
                if isinstance(tokens[0], str):
                    text_field = TextField([Token(t) for t in tokens],
                                           self._token_indexers)
                else:
                    text_field = TextField(tokens, self._token_indexers)
                instance.add_field('tokens', text_field)

                # Make the LM field
                if isinstance(tokens[0], str):
                    lm_field = BidirectionalLanguageModelField([Token(t) for t in tokens])
                else:
                    lm_field = BidirectionalLanguageModelField(tokens)

                # add it to the instance
                instance.add_field('lm_targets', lm_field)
                return instance

        reader = ForReturn.from_params(params)

        params.assert_empty('reader_with_lm')

        return reader
