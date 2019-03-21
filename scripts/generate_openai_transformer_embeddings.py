"""
Given a pre-processed input text file, this command outputs the internal
layers used to compute OpenAI transformer representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (<http://docs.h5py.org/en/latest/>) where, with the --all flag, each
sentence is a size (13, num_tokens, 768) array with the transformer representations.

For information, see

.. code-block:: console

    usage: python scripts/generate_openai_transformer_embeddings.py
                                                    [-h]
                                                    [--all] [--top] [--average]
                                                    [--model-path MODEL_PATH]
                                                    [--batch-size BATCH_SIZE]
                                                    [--cuda_device CUDA_DEVICE]
                                                    [--forget_sentences FORGET_SENTENCES]
                                                    [--use-sentence-keys USE_SENTENCE_KEYS]
                                                    input_path output_path

    Generate OpenAI transformer embeddings

    positional arguments:
    input_path
    output_path

   optional arguments:
     -h, --help            show this help message and exit
     --all                 Output all 13 vectors.
     --top                 Output the top vector.
     --average             Output the average of the vectors.
     --model-path MODEL_PATH
                           A path to the trained OpenAI transformer model.
     --batch-size BATCH_SIZE
                           The batch size to use.
     --cuda-device CUDA_DEVICE
                           The cuda_device to run on.
     --forget-sentences    If this flag is specified, and --use-sentence-keys is
                           not, remove the string serialized JSON dictionary that
                           associates sentences with their line number (its HDF5
                           key) that is normally placed in the
                           "sentence_to_index" HDF5 key.
     --use-sentence-keys   Normally a sentence's line number is used as the HDF5
                           key for its embedding. If this flag is specified, the
                           sentence itself will be used as the key.
"""
# pylint: disable=line-too-long,redefined-outer-name,invalid-name

import argparse
import json
import logging
from typing import List, Iterator
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy
import torch

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import OpenaiTransformerBytePairIndexer
from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.nn.util import get_range_vector, get_device_of

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_MODEL_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
DEFAULT_BATCH_SIZE = 64

def main(input_path: str,
         output_path: str,
         output_format: str = 'all',
         model_path: str = DEFAULT_MODEL_PATH,
         batch_size: int = DEFAULT_BATCH_SIZE,
         cuda_device: int = -1,
         forget_sentences: bool = False,
         use_sentence_keys: bool = False) -> None:

    assert output_format in ['all', 'top', 'average'], f"unknown output format {output_format}"

    model_path = cached_path(model_path)
    indexer = OpenaiTransformerBytePairIndexer(model_path=model_path)
    transformer = OpenaiTransformer(model_path=model_path)
    if cuda_device >= 0:
        transformer = transformer.cuda(device=cuda_device)

    with open(cached_path(input_path), 'r') as input_file:
        sentences = [line.strip() for line in Tqdm.tqdm(input_file)]

    blank_lines = [i for (i, line) in enumerate(sentences) if line == ""]
    if blank_lines:
        raise ConfigurationError(f"Your input file contains empty lines at indexes "
                                 f"{blank_lines}. Please remove them.")
    split_sentences = [[Token(word) for word in sentence.split()] for sentence in Tqdm.tqdm(sentences)]

    # Uses the sentence index as the key.

    if use_sentence_keys:
        logger.warning("Using sentences as keys can fail if sentences "
                       "contain forward slashes or colons. Use with caution.")
        embedded_sentences = zip(sentences, embed_sentences(indexer, transformer, split_sentences, output_format, batch_size, cuda_device))
    else:
        embedded_sentences = ((str(i), x) for i, x in
                              enumerate(embed_sentences(indexer, transformer, split_sentences, output_format, batch_size, cuda_device)))

    sentence_to_index = {}
    logger.info("Processing sentences.")
    with h5py.File(output_path, 'w') as fout:
        for key, embeddings in Tqdm.tqdm(embedded_sentences):
            if use_sentence_keys and key in fout.keys():
                raise ConfigurationError(f"Key already exists in {output_path}. "
                                         f"To encode duplicate sentences, do not pass "
                                         f"the --use-sentence-keys flag.")

            if not forget_sentences and not use_sentence_keys:
                sentence = sentences[int(key)]
                sentence_to_index[sentence] = key

            if output_format == "all":
                output = embeddings
            elif output_format == "top":
                output = embeddings[-1]
            elif output_format == "average":
                output = numpy.average(embeddings, axis=0)


            fout.create_dataset(
                    str(key),
                    output.shape, dtype='float32',
                    data=output
            )
        if not forget_sentences and not use_sentence_keys:
            sentence_index_dataset = fout.create_dataset(
                    "sentence_to_index",
                    (1,),
                    dtype=h5py.special_dtype(vlen=str))
            sentence_index_dataset[0] = json.dumps(sentence_to_index)

def embed_sentences(indexer: OpenaiTransformerBytePairIndexer,
                    transformer: OpenaiTransformer,
                    sentences: List[List[Token]],
                    output_format: str,
                    batch_size: int,
                    cuda_device: int) -> Iterator[torch.Tensor]:
    vocab = Vocabulary()
    # Set the transformer to eval mode.
    transformer.eval()

    for batch in lazy_groups_of(iter(sentences), batch_size):
        instances = []

        for tokens in batch:
            field = TextField(tokens, {'index': indexer})
            instance = Instance({"openai": field})
            instances.append(instance)

        dataset = Batch(instances)
        dataset.index_instances(vocab)
        tensors = dataset.as_tensor_dict()

        inputs = tensors['openai']['index']
        offsets = tensors['openai']['index-offsets']  # (batch_size, original_sequence_length)
        mask = tensors['openai']['mask']

        if cuda_device >= 0:
            inputs = inputs.cuda(cuda_device)
            offsets = offsets.cuda(cuda_device)

        _, num_timesteps = inputs.size()

        # vocab_size, vocab_size + 1, ...
        vocab_size = transformer.vocab_size - transformer.n_ctx
        positional_encodings = get_range_vector(512, device=get_device_of(inputs)) + vocab_size
        positional_encodings = positional_encodings.unsqueeze(0).expand(len(instances), num_timesteps)

        # Combine the inputs with positional encodings
        batch_tensor = torch.stack([
                inputs,   # (batch_size, num_timesteps)
                positional_encodings
        ], dim=-1)

        # Embeddings is num_output_layers x (batch_size, num_timesteps, embedding_dim)
        layer_activations = transformer(batch_tensor)
        embeddings = torch.stack(layer_activations, 1)  # (batch_size, num_output_layers, num_timesteps, embedding_dim)
        embeddings = embeddings.transpose(1, 2)         # (batch_size, num_timesteps, num_output_layers, embedding_dim)
        range_vector = get_range_vector(len(instances), device=get_device_of(inputs)).unsqueeze(1)
        embeddings = embeddings[range_vector, offsets]  # (batch_size, num_tokens, num_output_layers, embedding_dim)
        embeddings = embeddings.transpose(1, 2)         # (batch_size, num_output_layers, num_tokens, embedding_dim)

        for i in range(len(instances)):
            # embeddings[i] is (num_output_layers, num_tokens, embedding_dim)
            num_tokens = torch.sum(mask[i]).item()

            if output_format == "all":
                output = embeddings[i, :, :num_tokens]
            elif output_format == "top":
                output = embeddings[i, -1, :num_tokens]
            elif output_format == "average":
                output = numpy.average(embeddings[i, :, :num_tokens], axis=0)

            yield output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OpenAI transformer embeddings')
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--all", action='store_true')
    parser.add_argument("--top", action='store_true')
    parser.add_argument("--average", action='store_true')
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--forget-sentences", type=bool, default=False)
    parser.add_argument("--use-sentence-keys", type=bool, default=False)
    args = parser.parse_args()

    formats = [f for f in [args.all, args.top, args.average] if f]
    assert len(formats) <= 1, "please choose only one format"

    if args.top:
        output_format = "top"
    elif args.average:
        output_format = "average"
    else:
        output_format = "all"

    main(input_path=args.input_path,
         output_path=args.output_path,
         output_format=output_format,
         model_path=args.model_path,
         batch_size=args.batch_size,
         cuda_device=args.cuda_device,
         forget_sentences=args.forget_sentences,
         use_sentence_keys=args.use_sentence_keys)
