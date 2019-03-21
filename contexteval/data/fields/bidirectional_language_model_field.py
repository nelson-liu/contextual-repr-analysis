import torch
import numpy as np

from typing import Dict, List, Optional

from spacy.tokens import Token as SpacyToken

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import SequenceField
from allennlp.data.fields.text_field import TokenList


class BidirectionalLanguageModelField(SequenceField[Dict[str, torch.Tensor]]):
    """
    Taken from calypso:
    https://github.com/allenai/calypso/blob/ac934b6881787387581efaa8a646531278010652/calypso/allennlp_bridge.py#L51-L131

    Field for adding targets for a BidirectionalLM in a multi-task setting.
    """
    def __init__(self,
                 tokens: List[Token]) -> None:
        self.tokens = tokens
        self._indexed_tokens: Optional[Dict[str, TokenList]] = None
        self._directions = ['forward_targets', 'backward_targets']

        if not all([isinstance(x, (Token, SpacyToken)) for x in tokens]):
            raise ConfigurationError("TextFields must be passed Tokens. "
                                     "Found: {} with types {}.".format(tokens, [type(x) for x in tokens]))

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def index(self, vocab: Vocabulary):
        # self._indexed_tokens = {'forward_targets': array of next token id,
        #                         'backward_targets': array of prev token id}

        tokens = [token.text for token in self.tokens]
        token_ids = [vocab.get_token_index(token, namespace='lm')
                     for token in tokens]

        n_tokens = len(tokens)
        forward_targets = np.zeros(n_tokens, dtype='int32')
        forward_targets[0:-1] = token_ids[1:]

        backward_targets = np.zeros(n_tokens, dtype='int32')
        backward_targets[1:n_tokens] = token_ids[0:(n_tokens - 1)]

        self._indexed_tokens = {
            direction: target for direction, target in
            zip(self._directions, [forward_targets, backward_targets])
        }

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # the padding length is the number of tokens in the sequence
        if self._indexed_tokens is None:
            raise ValueError("self._indexed_tokens is None.")
        return {'num_lm_targets': len(self._indexed_tokens['forward_targets'])}

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1) -> Dict[str, torch.Tensor]:

        desired_num_tokens = padding_lengths['num_lm_targets']
        if self._indexed_tokens is None:
            raise ValueError("self._indexed_tokens is None.")
        unpadded_num_tokens = len(self._indexed_tokens['forward_targets'])

        # need to pad indexed_tokens to the specified length then return
        # as torch tensor on device
        tensors = {}

        for k in ['forward_targets', 'backward_targets']:
            padded_array = np.zeros(desired_num_tokens, dtype=np.int32)
            padded_array[:unpadded_num_tokens] = self._indexed_tokens[k]

            tensor = torch.LongTensor(padded_array)
            if cuda_device >= 0:
                tensor = tensor.cuda(cuda_device)
            tensors[k] = tensor

        return tensors

    @overrides
    def empty_field(self, vocab):
        text_field = BidirectionalLanguageModelField([])
        text_field.index(vocab)
        return text_field

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)
