#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [lm_name] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [lm_name] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [lm_name] [name_prefix]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [lm_name] [name_prefix]"
    exit 1
fi

TASK_LIST_PATH=$1
CONFIG_DIR=$2
LM_NAME=$3
NAME_PREFIX=$4

echo "Task list path: ${TASK_LIST_PATH}"
echo "Experiment config directory: ${CONFIG_DIR}"
echo "LM name: ${LM_NAME}"
echo "Name prefix: ${NAME_PREFIX}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Running task : ${line}"
    python scripts/evaluate_with_beaker.py "${CONFIG_DIR}"/"${line}".json \
           --source "contexteval_data:./data" \
           --source "${LM_NAME}:./language_models/${LM_NAME}"\
           --gpu-count 1 \
           --desc "${NAME_PREFIX} on ${line} task" --name "${NAME_PREFIX}_${line}"
done < "${TASK_LIST_PATH}"
