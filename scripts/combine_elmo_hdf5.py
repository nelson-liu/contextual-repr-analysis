import argparse
import h5py
import json
from tqdm import tqdm


def main(args: argparse.Namespace):
    # Read the sentences at the path
    new_sentence_to_index = {}
    with h5py.File(args.output_path, 'w') as fout:
        for hdf5_path in tqdm(args.hdf5_paths):
            with h5py.File(hdf5_path, "r") as embeddings:
                sentence_to_index = json.loads(embeddings.get("sentence_to_index")[0])
                for sentence, index in tqdm(sentence_to_index.items(), leave=False):
                    # Get the embedding from the hdf5 file
                    embedding = embeddings[index][:]
                    # Get the new index to use in the combined hdf5 file
                    new_index = str(len(new_sentence_to_index))
                    # Add data to the new hdf5 file
                    if sentence in new_sentence_to_index:
                        print("WARNING: Sentence {} appears twice".format(sentence))
                    new_sentence_to_index[sentence] = new_index
                    fout.create_dataset(str(new_index),
                                        embedding.shape,
                                        dtype='float32',
                                        data=embedding)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index", (1,), dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(new_sentence_to_index)
        print("Done! {} sentences total".format(len(new_sentence_to_index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5-paths', type=str, required=True, nargs="+",
                        help=('Path to the ELMo-format HDF5 files to combine'))
    parser.add_argument('--output-path', type=str, required=True,
                        help=('Path to write the combined HDF5 file.'))
    args = parser.parse_args()
    main(args)
