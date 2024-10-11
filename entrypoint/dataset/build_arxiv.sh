#!/bin/bash
set -ex

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

arxiv_dir=out/data/arxiv

python ollm/dataset/arxiv/build_categories.py \
    --output_dir $arxiv_dir/categories

python ollm/dataset/arxiv/build_pages.py \
    --output_dir $arxiv_dir/pages \
    --date_min "2020-01-01" \
    --date_max "2022-12-31"

python ollm/dataset/arxiv/export_graph.py \
    --categories_file $arxiv_dir/categories/raw_categories.json \
    --pages_file $arxiv_dir/pages/papers_with_citations.jsonl \
    --min_citations 10 \
    --output_dir $arxiv_dir/full

python ollm/dataset/train_test_split.py \
    --graph_file $arxiv_dir/full/full_graph.json \
    --split_depth 1 \
    --split_prop 0.5 \
    --output_dir $arxiv_dir/train_test_split \
    --seed 0

python ollm/dataset/train_test_split.py \
    --graph_file $arxiv_dir/train_test_split/train_graph.json \
    --split_depth 1 \
    --split_prop 0.3 \
    --output_dir $arxiv_dir/train_eval_split \
    --seed 0

mkdir -p $arxiv_dir/final
ln -s $arxiv_dir/train_eval_split/train_graph.json $arxiv_dir/final/train_graph.json
ln -s $arxiv_dir/train_eval_split/test_graph.json $arxiv_dir/final/eval_graph.json
ln -s $arxiv_dir/train_test_split/test_graph.json $arxiv_dir/final/test_graph.json

ls -lh $arxiv_dir/final