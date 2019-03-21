"""
Subsample the Billion Word Benchmark (or any file in that format) to
(approximately) the specified number of tokens..
"""
import argparse
import glob
import logging
import os

from tqdm import tqdm
logger = logging.getLogger(__name__)


def main():
    sentences = []
    total_num_tokens = 0
    logger.info("Reading files in {}".format(args.benchmark_path))
    for filepath in tqdm(glob.glob(os.path.join(args.benchmark_path, "*"))):
        # Read the file and count the number of tokens.
        with open(filepath) as benchmark_file:
            for line in benchmark_file:
                sentences.append(line)
                total_num_tokens += len(line.rstrip("\n").split(" "))
                if total_num_tokens > args.num_tokens:
                    break

    logger.info("Read {} sentences, got {} tokens".format(
        len(sentences), total_num_tokens))

    # Get total number of files
    num_output_files = int(len(sentences) / args.sentences_per_file)
    total_output_files_str = str(num_output_files).zfill(5)

    # Write them out to the specified output path, with 30000
    # sentences per file.
    output_filepaths = [
        os.path.join(
            args.output_path,
            "news.en-{}-of-{}".format(str(x).zfill(5), total_output_files_str))
        for x in range(1, num_output_files + 1)]

    # Write out the sentences to files.
    logger.info("Writing output to {}".format(args.output_path))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    current_sentence_index = 0
    pbar = tqdm(total=len(sentences))
    for output_filepath in output_filepaths:
        with open(output_filepath, "w") as output_file:
            # Number of sentences written to the current file.
            num_sentences_written = 0
            while current_sentence_index != len(sentences):
                output_file.write(sentences[current_sentence_index])
                current_sentence_index += 1
                num_sentences_written += 1
                pbar.update(1)
                # Start a new file
                if num_sentences_written >= args.sentences_per_file:
                    break
    pbar.close()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)

    # Path to project root directory
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir)))

    parser = argparse.ArgumentParser(
        description=("Subsample the Billion Word Benchmark (or any "
                     "file in that format) to (approximately) the "
                     "specified number of tokens."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark-path", type=str,
                        help=("Path to the billion word benchmark directory "
                              "with text files."))
    parser.add_argument("--num-tokens", type=int, required=True,
                        help=("The number of tokens to subsample."))
    parser.add_argument("--output-path", type=str, required=True,
                        help=("The folder to write the output files to."))
    parser.add_argument("--sentences-per-file", type=int, default=300000,
                        help=("The number of sentences to includer per file."))
    args = parser.parse_args()
    main()
