"""
Given a configuration, return the tokenized sentences to pre-contextualize.
"""
import argparse
from copy import deepcopy
import logging
import os
import sys

from allennlp.common import Params
from allennlp.data import DatasetReader
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import contexteval.data.dataset_readers  # noqa: F401


def main():
    all_sentences = set()

    for config_path in args.config_paths:
        params = Params.from_file(config_path)
        dataset_reader_params = params.pop('dataset_reader')
        # Remove contextualizer from dataset reader params
        dataset_reader_params.pop("contextualizer", None)
        # Set "include_raw_tokens" to true if it is false
        dataset_reader_params["include_raw_tokens"] = True
        # Deepcopy so dataset_reader_params is not modified
        train_reader = DatasetReader.from_params(deepcopy(dataset_reader_params))

        train_data_path = params.get('train_data_path', None)
        train_sentences = set()
        if train_data_path:
            for instance in train_reader.read(train_data_path):
                train_sentences.add(tuple(token.metadata for token in
                                          instance.fields["raw_tokens"].field_list))
        else:
            logger.warning("Did not find \"train_data_path\" key in experiment config, skipping")
        all_sentences.update(train_sentences)

        evaluation_reader_params = params.pop('validation_dataset_reader',
                                              dataset_reader_params)
        # Remove contextualizer from dataset reader params
        evaluation_reader_params.pop("contextualizer", None)
        # Set "include_raw_tokens" to true if it is false
        evaluation_reader_params["include_raw_tokens"] = True
        evaluation_reader = DatasetReader.from_params(deepcopy(evaluation_reader_params))

        validation_data_path = params.get('validation_data_path', None)
        validation_sentences = set()
        if validation_data_path:
            for instance in evaluation_reader.read(validation_data_path):
                validation_sentences.add(tuple(token.metadata for token in
                                               instance.fields["raw_tokens"].field_list))
        else:
            logger.warning("Did not find \"validation_data_path\" key in experiment config, skipping")
        all_sentences.update(validation_sentences)

        test_data_path = params.get('test_data_path', None)
        test_sentences = set()
        if test_data_path:
            for instance in evaluation_reader.read(test_data_path):
                test_sentences.add(tuple(token.metadata for token in
                                         instance.fields["raw_tokens"].field_list))
        else:
            logger.warning("Did not find \"test_data_path\" key in experiment config, skipping")
        all_sentences.update(test_sentences)

    sorted_sentences = sorted(all_sentences)

    # Write all_sentences to a file
    with open(args.output_path, "w") as output_file:
        for sentence in sorted_sentences:
            output_file.write("{}\n".format(" ".join(sentence)))
    logger.info("Wrote {} sentences to {}".format(len(all_sentences), args.output_path))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)

    # Path to project root directory
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir)))

    parser = argparse.ArgumentParser(
        description=("Given a configuration, return the tokenized "
                     "sentences to pre-contextualize."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config-paths", type=str, required=True, nargs="+",
                        help=("Path to the configuration to return "
                              "tokenized sentences to pre-contextualize."))
    parser.add_argument("--output-path", type=str, required=True,
                        help=("Path write the dataset sentences."))
    args = parser.parse_args()
    main()
