import codecs

from allennlp.common.params import Params
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from contexteval.common.custom_test_case import CustomTestCase


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


class TestVocabularyWithLM(CustomTestCase):
    def test_vocabulary_with_lm(self):
        vocab_filename = self.TEST_DIR / 'vocab_file'
        vocab = _create_vocab_with_lm(vocab_filename)

        # check the the vocabs!
        self.assertEqual(vocab.get_index_to_token_vocabulary('lm'),
                         {0: '@@PADDING@@',
                          1: '<S>',
                          2: '</S>',
                          3: '@@UNKNOWN@@',
                          4: 'a',
                          5: 'word',
                          6: 'another'})

        self.assertEqual(vocab.get_index_to_token_vocabulary(),
                         {0: '@@PADDING@@', 1: '@@UNKNOWN@@', 2: 'a',
                          3: 'c', 4: 'b'})
