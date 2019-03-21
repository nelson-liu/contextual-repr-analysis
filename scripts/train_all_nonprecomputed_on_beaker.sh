#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [contextualizer_prefix] [name_prefix]"
    exit 1
fi

TASK_LIST_PATH=$1
CONFIG_DIR=$2
CONTEXTUALIZER_PREFIX=$3
NAME_PREFIX=$4

echo "Task list path: ${TASK_LIST_PATH}"
echo "Experiment config directory: ${CONFIG_DIR}"
echo "Contextualizer prefix: ${CONTEXTUALIZER_PREFIX}"
echo "Name prefix: ${NAME_PREFIX}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Running task : ${line}"
    python scripts/evaluate_with_beaker.py "${CONFIG_DIR}"/"${line}".json --source "contexteval_data:./data" --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_weights:./contextualizers/${CONTEXTUALIZER_PREFIX}/${CONTEXTUALIZER_PREFIX}_weights.hdf5" --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_options:./contextualizers/${CONTEXTUALIZER_PREFIX}/${CONTEXTUALIZER_PREFIX}_options.json" --gpu-count 1 --desc "${NAME_PREFIX} on ${line} task" --name "${NAME_PREFIX}_${line}"
done < "${TASK_LIST_PATH}"
