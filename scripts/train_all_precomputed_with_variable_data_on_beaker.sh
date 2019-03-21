#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [num_layers] [max_instances]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [num_layers] [max_instances]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [num_layers] [max_instances]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [num_layers] [max_instances]"
    exit 1
fi

if [ -z "$5" ]
then
    echo "Fifth command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [num_layers] [max_instances]"
    exit 1
fi

TASK_INFO_PATH=$1
CONFIG_DIR=$2
CONTEXTUALIZER_PREFIX=$3
NUM_LAYERS=$4
MAX_INSTANCES=$5

echo "Task info path: ${TASK_INFO_PATH}"
echo "Experiment config directory: ${CONFIG_DIR}"
echo "Contextualizer prefix: ${CONTEXTUALIZER_PREFIX}"
echo "Number of layers to evaluate: ${NUM_LAYERS}"
echo "Max instances: ${MAX_INSTANCES}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"

    task_name="${columns[0]}"
    # e.g., adposition_supersense_tagging.hdf5
    task_contextualizer_filename="${columns[1]}"
    # remove extra path items from task_contextualizer_filename just in case
    task_contextualizer_filename_without_path=$(basename -- "${task_contextualizer_filename}")
    # remove the extension to get adposition_supersense_tagging
    task_contextualizer_name="${task_contextualizer_filename_without_path%.*}"

    echo "Running task : ${task_name}"
    echo "Task contextualizer name: ${task_contextualizer_name}"
    for ((layer_num=0; layer_num<$NUM_LAYERS; layer_num++)); do python scripts/evaluate_with_beaker.py "${CONFIG_DIR}"/"${task_name}".json --layer-num ${layer_num} --max-instances ${MAX_INSTANCES} --source "contexteval_data:./data" --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_${task_contextualizer_name}:./contextualizers/${CONTEXTUALIZER_PREFIX}/${task_contextualizer_filename}" --gpu-count 1 --desc "${CONTEXTUALIZER_PREFIX} on ${task_name} task with layer ${layer_num} and max instances ${MAX_INSTANCES}" --name "${CONTEXTUALIZER_PREFIX}_${task_name}_max_instances_${MAX_INSTANCES}_layer_${layer_num}"; done
done < "${TASK_INFO_PATH}"
