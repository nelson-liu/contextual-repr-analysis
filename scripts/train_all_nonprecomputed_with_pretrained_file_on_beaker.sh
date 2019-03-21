#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_path] [contextualizer_prefix] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_path] [contextualizer_prefix] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_path] [contextualizer_prefix] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_path] [contextualizer_prefix] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

if [ -z "$5" ]
then
    echo "Fifth command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_path] [contextualizer_prefix] [pretrained_file_prefix] [name_prefix]"
    exit 1
fi

TASK_LIST_PATH=$1
CONFIG_PATH=$2
CONTEXTUALIZER_PREFIX=$3
PRETRAINED_FILE_PREFIX=$4
NAME_PREFIX=$5

echo "Task list path: ${TASK_LIST_PATH}"
echo "Experiment config path: ${CONFIG_PATH}"
echo "Contextualizer prefix: ${CONTEXTUALIZER_PREFIX}"
echo "Pre-trained file prefix: ${PRETRAINED_FILE_PREFIX}"
echo "Name prefix: ${NAME_PREFIX}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    export pretrained_file="./pretrained_files/${PRETRAINED_FILE_PREFIX}_${line}/model.tar.gz"
    echo "pretrained_file : ${pretrained_file}"
    python scripts/evaluate_with_beaker.py "${CONFIG_PATH}" \
           --source "contexteval_data:./data" \
           --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_weights:./contextualizers/${CONTEXTUALIZER_PREFIX}/${CONTEXTUALIZER_PREFIX}_weights.hdf5" \
           --source "contexteval_contextualizers_${CONTEXTUALIZER_PREFIX}_options:./contextualizers/${CONTEXTUALIZER_PREFIX}/${CONTEXTUALIZER_PREFIX}_options.json" \
           --source "contexteval_pretrained_files_${PRETRAINED_FILE_PREFIX}_${line}:./pretrained_files/${PRETRAINED_FILE_PREFIX}_${line}/model.tar.gz" \
           --env "pretrained_file=./pretrained_files/${PRETRAINED_FILE_PREFIX}_${line}/model.tar.gz" \
           --gpu-count 1 \
           --desc "${NAME_PREFIX} on ${CONFIG_PATH} with contextualizer ${CONTEXTUALIZER_PREFIX} transferred from ${PRETRAINED_FILE_PREFIX}" \
           --name "${NAME_PREFIX}_transfer_${PRETRAINED_FILE_PREFIX}_${line}"
done < "${TASK_LIST_PATH}"
