#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_info_path] [config_dir] [name_prefix]"
    exit 1
fi



TASK_INFO_PATH=$1
CONFIG_DIR=$2
NAME_PREFIX=$3

echo "Task info path: ${TASK_INFO_PATH}"
echo "Experiment config directory: ${CONFIG_DIR}"
echo "Name prefix: ${NAME_PREFIX}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"

    task_name="${columns[0]}"
    echo "Running task : ${task_name}"
    python scripts/evaluate_with_beaker.py "${CONFIG_DIR}"/"${task_name}".json \
           --source "contexteval_data:./data" \
           --gpu-count 1 \
           --desc "${NAME_PREFIX} on ${task_name} task" \
           --name "${NAME_PREFIX}_${task_name}"
done < "${TASK_INFO_PATH}"
