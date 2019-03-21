#!/usr/bin/env bash
set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [output_dir] [cuda_device]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [output_dir] [cuda_device]"
    exit 1
fi

OUTPUT_DIR=$1
CUDA_DEVICE=$2

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"
    echo "column[0]:'${columns[0]}' column[1]:'${columns[1]}'"
    python scripts/generate_openai_transformer_embeddings.py ${columns[1]} ${OUTPUT_DIR}/${columns[0]} --cuda-device ${CUDA_DEVICE}
done < "scripts/contextualizers_to_sentences.txt"
