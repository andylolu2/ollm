#!/bin/bash
set -ex

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

download_path=$(
    huggingface-cli download --revision v2 --repo-type dataset andylolu24/arxiv-ol \
    train_eval_split/train_graph.json \
    train_eval_split/test_graph.json \
    train_test_split/test_graph.json
)

echo "Downloaded to $download_path"
output_path=out/data/arxiv/final

mkdir -p $output_path
ln -s $download_path/train_eval_split/train_graph.json $output_path/train_graph.json
ln -s $download_path/train_eval_split/test_graph.json $output_path/eval_graph.json
ln -s $download_path/train_test_split/test_graph.json $output_path/test_graph.json

ls -lh $output_path
