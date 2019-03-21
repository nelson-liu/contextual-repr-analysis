#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

TASK_INFO_PATH=$1
CONFIG_DIR=$2
CONTEXTUALIZER_PREFIX=$3
NAME_PREFIX=$4

echo "Task info path: ${TASK_INFO_PATH}"
echo "Experiment config directory: ${CONFIG_DIR}"
echo "Contextualizer prefix: ${CONTEXTUALIZER_PREFIX}"
echo "Name prefix: ${NAME_PREFIX}"

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
    python scripts/evaluate_with_beaker.py "${CONFIG_DIR}"/"${task_name}".json --source "contexteval_data:./data" --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_${task_contextualizer_name}:./contextualizers/${CONTEXTUALIZER_PREFIX}/${task_contextualizer_filename}" --gpu-count 1 --desc "${NAME_PREFIX} on ${task_name} task with learned scalar mix" --name "${NAME_PREFIX}_${task_name}"
done < "${TASK_INFO_PATH}"
