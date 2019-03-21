#!/usr/bin/env bash
for config_name in "ccg_supertagging" "conjunct_identification" "conll2000_chunking" "ptb_pos_tagging" "ptb_syntactic_dependency_arc_classification" "ptb_syntactic_dependency_arc_prediction" "semantic_dependency_arc_classification" "semantic_dependency_arc_prediction" "syntactic_constituency_grandparent_prediction" "syntactic_constituency_greatgrandparent_prediction" "syntactic_constituency_parent_prediction"; do
    python scripts/evaluate_with_beaker.py "transfer_experiment_configs/elmo_original_randomly_initialized_3/${config_name}.json" \
           --source "contexteval_data_resplit:./data" \
           --source "contexteval_contextualizers_elmo_original_randomly_initialized_3_weights:./contextualizers/elmo_original_randomly_initialized_3/elmo_original_randomly_initialized_3_weights.hdf5" \
           --source "contexteval_contextualizers_elmo_original_randomly_initialized_3_options:./contextualizers/elmo_original_randomly_initialized_3/elmo_original_randomly_initialized_3_options.json" \
           --gpu-count 1 \
           --desc "elmo_original_randomly_initialized_3 on resplit ${config_name} with contextualizer elmo_original_randomly_initialized_3" \
           --name "elmo_original_randomly_initialized_3_transfer_${config_name}"
done
