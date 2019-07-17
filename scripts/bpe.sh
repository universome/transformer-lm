#!/bin/bash

dataset=$1
num_bpes=$2
bpes_path="data/generated/bpes-$num_bpes.txt"

echo "Dataset: $dataset"
echo "Num bpes: $num_bpes"
echo "BPEs will be saved into $bpes_path"

subword-nmt learn-bpe -s $num_bpes < $dataset > $bpes_path
subword-nmt apply-bpe -c $bpes_path < $dataset > $dataset.bpe
