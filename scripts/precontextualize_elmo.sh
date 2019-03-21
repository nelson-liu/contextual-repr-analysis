#!/usr/bin/env bash
set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [output_dir] [weights_path] [options_file] [cuda_device]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [output_dir] [weights_path] [options_file] [cuda_device]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [output_dir] [weights_path] [options_file] [cuda_device]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [output_dir] [weights_path] [options_file] [cuda_device]"
    exit 1
fi

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"
    echo "column[0]:'${columns[0]}' column[1]:'${columns[1]}'"
    allennlp elmo ${columns[1]} "${1}"/${columns[0]} --all --weight-file "${2}" --cuda-device "${4}" --options-file "${3}"
done < "scripts/contextualizers_to_sentences.txt"
