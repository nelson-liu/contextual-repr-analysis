#!/usr/bin/env bash
# Run the experiments with taking ELMo and retraining the softmax on top
# on beaker. This is custom since we need to add the vocabulary path

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [config_path] [contextualizer_prefix] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [config_path] [contextualizer_prefix] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [config_path] [contextualizer_prefix] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fouth command line argument is null"
    echo "Please call the script with: script [config_path] [contextualizer_prefix] [name_prefix] [num_layers]"
    exit 1
fi

CONFIG_PATH=$1
CONTEXTUALIZER_PREFIX=$2
NAME_PREFIX=$3
NUM_LAYERS=$4

echo "Config path: ${CONFIG_PATH}"
echo "Contextualizer prefix: ${CONTEXTUALIZER_PREFIX}"
echo "Name prefix: ${NAME_PREFIX}"
echo "Number of layers to evaluate: ${NUM_LAYERS}"

echo "Running language modeling task (retraining softmax)"
for ((layer_num=0; layer_num<$4; layer_num++)); do python scripts/evaluate_with_beaker.py "${CONFIG_PATH}" --layer-num ${layer_num} --source "contexteval_data:./data" --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_language_modeling:./contextualizers/${CONTEXTUALIZER_PREFIX}/language_modeling.hdf5" --source "language_modeling_vocab:./models/${CONTEXTUALIZER_PREFIX}/language_modeling_vocab" --source "language_modeling_backward_vocab:./models/${CONTEXTUALIZER_PREFIX}/language_modeling_backward_vocab" --gpu-count 1 --desc "${NAME_PREFIX} on language modeling task with layer ${layer_num}" --name "${NAME_PREFIX}_language_modeling_layer_${layer_num}"; done
