#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [save_dir] [num_layers]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [save_dir] [num_layers]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [save_dir] [num_layers]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [task_list_path] [config_dir] [save_dir] [num_layers]"
    exit 1
fi

echo "Task file path: ${1}"
echo "Experiment config directory: ${2}"
echo "Save directory: ${3}"

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Running task : ${line}"
    for ((layer_num=0; layer_num<$4; layer_num++)); do allennlp train "${2}"/${line}.json -s "${3}"/${line}_layer_${layer_num} --include-package contexteval --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${layer_num}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${layer_num}'}}}'; done
done < "$1"
