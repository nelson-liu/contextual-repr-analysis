import logging
from typing import List

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
import numpy
import torch

from contexteval.contextualizers import Contextualizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Contextualizer.register("glove_contextualizer")
class GloveContextualizer(Contextualizer):
    """
    This "contextualizer" simply assigns each word to its pre-trained GloVe vector.

    Parameters
    ----------
    glove_path: str
        Path to the GloVe embeddings.
    embedding_dim: int
        The dimensionality of the GloVe embeddings.
    trainable: bool
        Whether to update the GloVe embeddings.
    """
    def __init__(self,
                 glove_path: str,
                 embedding_dim: int,
                 trainable: bool = False) -> None:
        super(GloveContextualizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        # Read the GloVe file, and produce a dictionary of tokens to indices, a dictionary
        # of indices to tokens, and a PyTorch Embedding object.
        self.token_to_idx = {DEFAULT_OOV_TOKEN: 0}
        self.idx_to_token = {0: DEFAULT_OOV_TOKEN}

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        logger.info("Reading pretrained embeddings from file")
        embeddings = {}
        with EmbeddingsTextFile(glove_path) as embeddings_file:
            for line in Tqdm.tqdm(embeddings_file):
                token = line.split(' ', 1)[0]
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != self.embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                                   self.embedding_dim, len(fields) - 1, line)
                    continue

                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector
                self.token_to_idx[token] = len(self.token_to_idx)
                self.idx_to_token[len(self.idx_to_token)] = token
        if not embeddings:
            raise ConfigurationError("No embeddings of correct dimension found; you probably "
                                     "misspecified your embedding_dim parameter, or didn't "
                                     "pre-populate your Vocabulary")

        all_embeddings = numpy.asarray(list(embeddings.values()))
        embeddings_mean = float(numpy.mean(all_embeddings))
        embeddings_std = float(numpy.std(all_embeddings))
        # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
        # then filling in the word vectors we just read.
        vocab_size = len(self.token_to_idx)
        logger.info("Initializing pre-trained embedding layer")
        embedding_matrix = torch.FloatTensor(vocab_size, self.embedding_dim).normal_(embeddings_mean,
                                                                                     embeddings_std)
        # Start at 1, since the 0th token is OOV, and fill in the embedding matrix
        for i in range(1, vocab_size):
            embedding_matrix[i] = torch.FloatTensor(embeddings[self.idx_to_token[i]])
        self.weight = torch.nn.Parameter(embedding_matrix, requires_grad=self.trainable)

    def forward(self, sentences: List[List[str]]) -> torch.FloatTensor:
        """
        Parameters
        ----------
        sentences: List[List[str]]
            A batch of sentences. len(sentences) is the batch size, and each sentence
            itself is a list of strings (the constituent words). If the batch is padded,
            the expected padding token in the Python ``None``.

        Returns
        -------
        representations: List[FloatTensor]
            A list with the contextualized representations of all words in an input sentence.
            Each inner FloatTensor is of shape (seq_len, repr_dim), and an outer List
            is used to store the representations for each input sentence.
        """
        batch_representations = []

        for sentence in sentences:
            sentence_tensor = torch.FloatTensor(
                len(sentence), self.embedding_dim)
            for i, word in enumerate(sentence):
                word_index = self.token_to_idx.get(word, 0)
                sentence_tensor[i] = self.weight[word_index]
            batch_representations.append(sentence_tensor)
        return batch_representations
