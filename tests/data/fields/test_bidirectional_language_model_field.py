import codecs
import numpy as np

from allennlp.common.params import Params
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.data.fields import BidirectionalLanguageModelField


def _create_vocab_with_lm(vocab_filename):
    # make an instance
    token_indexer = SingleIdTokenIndexer("tokens")
    text_field = TextField([Token(t) for t in ["a", "a", "a", "a", "b", "b", "c", "c", "c"]],
                           {"tokens": token_indexer})
    instances = [Instance({"text": text_field})]

    with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
        vocab_file.write('<S>\n')
        vocab_file.write('</S>\n')
        vocab_file.write('<UNK>\n')
        vocab_file.write('a\n')
        vocab_file.write('word\n')
        vocab_file.write('another\n')

    params = Params({
        "type": "vocabulary_with_lm",
        "lm_vocab_file": vocab_filename,
        "oov_token": "<UNK>"
    })

    vocab = Vocabulary.from_params(params, instances)

    return vocab


class TestBidirectionalLanguageModelField(CustomTestCase):
    def test_bidirectional_lm_field(self):
        tokens = [Token(token) for token in ['a', 'unk', 'word']]

        bilm_field = BidirectionalLanguageModelField(tokens)
        vocab = _create_vocab_with_lm(self.TEST_DIR / 'vocab_file')

        bilm_field.index(vocab)
        expected_targets = {
            'forward_targets': np.array([3, 5, 0], dtype=np.int32),
            'backward_targets': np.array([0, 4, 3], dtype=np.int32)
        }

        self.assertEqual(list(sorted(bilm_field._indexed_tokens.keys())),
                         list(sorted(expected_targets.keys())))
        for k in expected_targets.keys():
            self.assertTrue(
                (expected_targets[k] == bilm_field._indexed_tokens[k]).all())

        self.assertEqual(
            bilm_field.get_padding_lengths(), {'num_lm_targets': 3})

        self.assertEqual(bilm_field.sequence_length(), 3)

        expected_tensors = {
            'forward_targets': np.array([3, 5, 0, 0, 0], dtype=np.int32),
            'backward_targets': np.array([0, 4, 3, 0, 0], dtype=np.int32)
        }
        tensors = bilm_field.as_tensor({'num_lm_targets': 5})
        for k in expected_tensors.keys():
            self.assertTrue(
                (expected_tensors[k] == tensors[k].numpy()).all())
