#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [config_path] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [config_path] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [config_path] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

CONFIG_PATH=$1
PRETRAINED_FILE_PREFIX=$2
NAME_PREFIX=$3

echo "Experiment config path: ${CONFIG_PATH}"
echo "Pre-trained file prefix: ${PRETRAINED_FILE_PREFIX}"
echo "Name prefix: ${NAME_PREFIX}"

# These are the tasks that we have pretrained contextualizers for.
for task_name in "ccg_supertagging" "conjunct_identification" "conll2000_chunking" "ptb_pos_tagging" "ptb_syntactic_dependency_arc_classification" "ptb_syntactic_dependency_arc_prediction" "semantic_dependency_arc_classification" "semantic_dependency_arc_prediction" "syntactic_constituency_grandparent_prediction" "syntactic_constituency_greatgrandparent_prediction" "syntactic_constituency_parent_prediction"; do
    export pretrained_file="./pretrained_files/${PRETRAINED_FILE_PREFIX}_${task_name}/model.tar.gz"
    echo "pretrained_file : ${pretrained_file}"
    python scripts/evaluate_with_beaker.py "${CONFIG_PATH}" \
           --source "contexteval_data:./data" \
           --source "contexteval_contextualizers_elmo_small_weights:./contextualizers/elmo_small/elmo_small_weights.hdf5" \
           --source "contexteval_contextualizers_elmo_small_options:./contextualizers/elmo_small/elmo_small_options.json" \
           --source "contexteval_pretrained_files_${PRETRAINED_FILE_PREFIX}_${task_name}:./pretrained_files/${PRETRAINED_FILE_PREFIX}_${task_name}/model.tar.gz" \
           --env "pretrained_file=./pretrained_files/${PRETRAINED_FILE_PREFIX}_${task_name}/model.tar.gz" \
           --gpu-count 1 \
           --desc "${NAME_PREFIX} on ${CONFIG_PATH} with contextualizer ${CONTEXTUALIZER_PREFIX} transferred from ${task_name}" \
           --name "${NAME_PREFIX}_${task_name}"
done
