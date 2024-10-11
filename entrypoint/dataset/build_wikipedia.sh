#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

wiki_dir=out/data/wikipedia

python ollm/dataset/wikipedia/build_categories.py \
    --max_depth 3 \
    --output_dir $wiki_dir/categories

python ollm/dataset/wikipedia/build_pages.py \
    --categories_file $wiki_dir/categories/raw_categories.jsonl \
    --output_dir $wiki_dir/pages

python ollm/dataset/wikipedia/export_graph.py \
    --categories_file $wiki_dir/categories/raw_categories.jsonl \
    --pages_file $wiki_dir/pages/raw_pages.jsonl \
    --output_dir $wiki_dir/full

python ollm/dataset/train_test_split.py \
    --graph_file $wiki_dir/full/graph_depth_3.json \
    --split_depth 1 \
    --split_prop 0.5 \
    --output_dir $wiki_dir/train_test_split

python ollm/dataset/train_test_split.py \
    --graph_file $wiki_dir/train_test_split/train_graph.json \
    --split_depth 1 \
    --split_prop 0.3 \
    --output_dir $wiki_dir/train_eval_split

mkdir -p $wiki_dir/final
ln -s $wiki_dir/train_eval_split/train_graph.json $wiki_dir/final/train_graph.json
ln -s $wiki_dir/train_eval_split/test_graph.json $wiki_dir/final/eval_graph.json
ln -s $wiki_dir/train_test_split/test_graph.json $wiki_dir/final/test_graph.json

ls -lh $wiki_dir/final