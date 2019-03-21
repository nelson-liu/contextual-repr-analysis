# Scripts

This directory contains all the utility scripts and other tools used to
interface with the code and make running experiments a bit easier.

Scripts should be run in the project root directory (i.e., one level above where
this file lives).

Filename | Description
--- | ---
`contextualizers_to_sentences.txt` | Used in the `precontextualize_*.sh` scripts to provide a mapping from the filename of an input list of sentences to the filename of the desired output hdf5 file. Note that the coreference resolution task doesn't have an entry here, and should be run manually.
`evaluate_with_beaker.py` |  Script to run `allennlp train` jobs on Beaker.
`fetch_all_from_beaker.sh` | Script to download all the models for each task and each layer of representation from Beaker.
`get_config_sentences.py` | Script to generate the sentences to precompute vectors for given an input allennlp config.
`get_elmo_scalar_weights.py` | Script to analyze a serialized allennlp model to determine the scalar weights corresponding to each of the ELMo representation layers.
`precontextualize_calypso.sh` | Script to run a Calypso contextualizer over all sentences for all tasks (except for coreference, which should be done manually).
`precontextualize_elmo.sh` | Script to run an ELMo contextualizer over all sentences for all tasks (except for coreference, which should be done manually).
`precontextualize_openai_transformer.sh` | Script to run the OpenAI transformer contextualizer over all sentences for all tasks (except for coreference, which should be done manually).
`subsample_1b_benchmark.py` | Script to generate a random subsample of the 1B Word LM Benchmark.
`task_info.txt` | File with information about each task's name, its associated contextualizer name, and the path to its sentences.
`task_list.txt` | File with a list of tasks. This is simply the first column of `task_info.txt`.
`train_all_nonprecomputed_on_beaker.sh` | Train auxiliary classifiers on all tasks for a given contextualizer. This assumes that only 1 layer of representations is generated and to be evaluated (e.g., when fine-tuning ELMo while learning a scalar mix).
`train_all_precomputed.sh` | Train auxiliary classifiers on all tasks and all layers for a given contextualizer, where the contextualizer is a PrecomputedContextualizer (i.e., all the vectors have been written to disk and are static). This runs the job on the local machine.
`train_all_precomputed_on_beaker.sh` | Train auxiliary classifiers on all tasks and all layers for a given contextualizer, where the contextualizer is a PrecomputedContextualizer (i.e., all the vectors have been written to disk and are static). This runs the job on the cloud in Beaker.
`train_all_precomputed_with_learned_scalar_mix_on_beaker.sh` | Train auxiliary classifiers on all tasks and all layers for a given contextualizer, where the contextualizer is a ScalarMixedPrecomputedContextualizer (i.e., all the vectors have been written to disk and are static, but a dynamic weighting of them is learned). This runs the job on the cloud in Beaker.
