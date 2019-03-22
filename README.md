[![Build Status](https://travis-ci.org/nelson-liu/contextual-repr-analysis.svg?branch=master)](https://travis-ci.org/nelson-liu/contextual-repr-analysis)
[![codecov](https://codecov.io/gh/nelson-liu/contextual-repr-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/nelson-liu/contextual-repr-analysis)

# contextual-repr-analysis

A toolkit for evaluating the linguistic knowledge and transferability of contextual word representations. Code for [_Linguistic Knowledge and Transferability of Contextual Representations_](http://nelsonliu.me/papers/liu+gardner+belinkov+peters+smith.naacl2019.pdf), to appear at NAACL 2019.


For a description of the included tasks, see [`TASKS.md`](./TASKS.md).

## Table of Contents

- [Installation](#installation)
- [Getting Started: Evaluating Representations](#getting-started-evaluating-representations)
- [References](#references)

## Installation

This project is being developed in Python 3.6, and CI runs the tests in Python
3.6 as well
(via [TravisCI](https://travis-ci.com/nelson-liu/contextual-repr-analysis)).

[Conda](https://conda.io/) will set up a virtual environment with the exact
version of Python used for development along with all the dependencies needed to
run the code.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.

    ```bash
    cd contextual-repr-analysis
    ```

3.  Create a Conda environment with Python 3.6 .

    ```bash
    conda create -n contextual_repr_analysis python=3.6
    ```

4.  Now activate the Conda environment. You will need to activate the Conda
    environment in each terminal in which you want to run code from this repo.

    ```bash
    source activate contextual_repr_analysis
    ```

5.  Install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

You should now be able to test your installation with `py.test -v`.  Congratulations!

## Getting Started: Evaluating Representations

This section walks through an example of evaluating ELMo on the English Web
Treebank (EWT) English POS tagging task.

### Step 1. Precomputing the Word Representations

The easiest way to get started with evaluating your representations is to
precompute representations for each word in each sentence in the evaluation
dataset. In each of the `data/` directories, there exists a text file of
sentences (newline delimited, and tokens are space-delimited). These are the
sentences used during training and evaluation, so getting representations for
these should be enough. If you write a new `DatasetReader` and want to generate
these sentences, use the script
at [`./scripts/get_config_sentences.py`](./scripts/get_config_sentences.py) (see
`python ./scripts/get_config_sentences.py -h` for more information on usage).

The format of the HDF5 file should be as follows:

1. The keys should be numbers (represented as strings), corresponding to line
   numbers.

2. The value associated with each key is expected to a numpy array of word
   representations. Acceptable shapes are `(sequence_length, representation_dim)` or
   `(num_layers, sequence_length, representation_dim)`.
  
3. Another key, the string value `"sentence_to_index"`, should store a
   string-serialized JSON dictionary mapping from sentences (the sentences that the
   representations in the values are calculated from) to string numbers (the
   other keys of the HDF5 file).
  
If you have a `Dict[str, str]` called `sentence_to_index` and a `Dict[str,
numpy.ndarray]` named `vectors` containing a mapping from `str` numbers to
vectors (a dictionary with consecutive numbers as keys and numpy arrays as
values), you can pass the dictionaries into the following function to produce
an HDF5 file with the proper format.

```python
def make_hdf5_file(sentence_to_index, vectors):
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key),
                embeddings.shape, dtype='float32',
                data=embeddings)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)
```

### Step 2. Creating the experiment configuration

`./experiment_configs` contains all the experiment configurations used in this
project. **TODO (nfliu): Write more about writing your own experiment config**.

### Step 3. Training the probing model

To train a probing model on top of the precomputed word representations,
we use the `allennlp train` command.

Given a single configuration file, we can train with:

```bash
allennlp train <config_path> -s <model_save_path> --include-package contexteval
```

For example, for training a contextualizer on the topmost ELMo layer (the
default) for POS tagging:

```bash
allennlp train experiment_configs/elmo_original/ewt_pos_tagging.json \
    -s ewt_pos_tagging_topmost_layer \
    --include-package contexteval
```

**Note that the precomputed contextualizers in the experiment config do not have
a layer specified**. This causes models to default to using the topmost layer.
To train on, say, the first layer (index 0), you can run this command:

```bash
allennlp train experiment_configs/elmo_original/ewt_pos_tagging.json \
    -s models/elmo_original/ewt_pos_tagging_layer_0 --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": 0}}, "validation_dataset_reader": {"contextualizer": {"layer_num": 0}}}'
```

To train on all layers, one-by-one, you can wrap the above in a bash for-loop.

```bash
for i in 0 1 2; do allennlp train experiment_configs/elmo_original/ewt_pos_tagging.json \
    -s models/elmo_original/ewt_pos_tagging_layer_${i} --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done
```

### Step 4: Evaluating the probing model on test data

To evaluate a trained probing model on test data, use the `allennlp evaluate` command.

To evaluate the three models we trained above and log the output to a file, we can run:

```bash
for i in 0 1 2; do allennlp evaluate models/elmo_original/ewt_pos_tagging_layer_${i}/model.tar.gz \
    --evaluation-data-file ./data/pos/en_ewt-ud-test.conllu --cuda-device 0 \
    --include-package contexteval 2>&1 | tee models/elmo_original/ewt_pos_tagging_layer_${i}/evaluation.log; done
```

## References

```
@InProceedings{liu-gardner-belinkov-peters-smith:2019:NAACL,
  author    = {Liu, Nelson F.  and  Gardner, Matt  and  Belinkov, Yonatan  and  Peters, Matthew E.  and  Smith, Noah A.},
  title     = {Linguistic Knowledge and Transferability of Contextual Representations},
  booktitle = {Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2019}
}
```
