#!/usr/bin/env bash
set -e

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [bert_extract_features_path] [output_dir] [init_checkpoint] [bert_config_file] [vocab_file]"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Second command line argument is null"
    echo "Please call the script with: script [bert_extract_features_path] [output_dir] [init_checkpoint] [bert_config_file] [vocab_file]"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Third command line argument is null"
    echo "Please call the script with: script [bert_extract_features_path] [output_dir] [init_checkpoint] [bert_config_file] [vocab_file]"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Fourth command line argument is null"
    echo "Please call the script with: script [bert_extract_features_path] [output_dir] [init_checkpoint] [bert_config_file] [vocab_file]"
    exit 1
fi

if [ -z "$5" ]
then
    echo "Fifth command line argument is null"
    echo "Please call the script with: script [bert_extract_features_path] [output_dir] [init_checkpoint] [bert_config_file] [vocab_file]"
    exit 1
fi

BERT_EXTRACT_FEATURES_PATH=$1
OUTPUT_DIR=$2
INIT_CHECKPOINT=$3
BERT_CONFIG_FILE=$4
VOCAB_FILE=$5

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"
    echo "column[0]:'${columns[0]}' column[1]:'${columns[1]}'"
    python ${BERT_EXTRACT_FEATURES_PATH} --input_file=${columns[1]} \
           --output_file=${OUTPUT_DIR}/${columns[0]} \
           --vocab_file=${VOCAB_FILE} \
           --bert_config_file=${BERT_CONFIG_FILE} \
           --init_checkpoint=${INIT_CHECKPOINT} \
           --max_seq_length=384 \
           --batch_size=1 \
           --do_lower_case=false
done < "scripts/contextualizers_to_sentences.txt"
