#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_info_path] [saved_models_dir] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_info_path] [saved_models_dir] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_info_path] [saved_models_dir] [name_prefix] [num_layers]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_info_path] [saved_models_dir] [name_prefix] [num_layers]"
    exit 1
fi

TASK_INFO_PATH=$1
SAVED_MODELS_DIR=$2
NAME_PREFIX=$3
NUM_LAYERS=$4

echo "Task info path: ${TASK_INFO_PATH}"
echo "Directory to save results to: ${SAVED_MODELS_DIR}"
echo "Name prefix: ${NAME_PREFIX}"
echo "Number of layers evaluated: ${NUM_LAYERS}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"

    task_name="${columns[0]}"
    echo "Fetching results for task : ${task_name}"

    for ((layer_num=0; layer_num<$NUM_LAYERS; layer_num++)); do
        echo "Fetching layer ${layer_num}"
        result_dataset_id=$(beaker experiment inspect "${NAME_PREFIX}_${task_name}_layer_${layer_num}" | jq -r '.[0].nodes[0].result_id')
        beaker dataset fetch --output "${SAVED_MODELS_DIR}"/"${NAME_PREFIX}"/"${task_name}_layer_${layer_num}" $result_dataset_id
    done
done < "${TASK_INFO_PATH}"
