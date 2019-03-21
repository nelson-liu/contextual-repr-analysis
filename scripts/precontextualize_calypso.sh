#!/usr/bin/env bash
set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

if [ -z "$5" ]
then
    echo "Fifth command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

if [ -z "$6" ]
then
    echo "Sixth command line argument is null"
    echo "Please call the script with: script [calypso_dump_lm_embeddings_path] [output_dir] [weights_path] [options_file] [vocab_file] [cuda_device]"
    exit 1
fi

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"
    echo "column[0]:'${columns[0]}' column[1]:'${columns[1]}'"
    python ${1} ${columns[1]} "${2}"/${columns[0]} --weight-file "${3}" --options-file "${4}" --vocab-file "${5}" --cuda-device "${6}"
done < "scripts/contextualizers_to_sentences.txt"
