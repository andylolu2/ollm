#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] \
    ([ $1 != "eval" ] && [ $1 != "test" ]); then
echo \
"Usage: $0 <split>

split: eval | test
"
exit 1
fi

split=$1

num_concepts=$(
python - <<EOF
from ollm.dataset import data_model

G = data_model.load_graph("out/data/wikipedia/final/${split}_graph.json")
print(len(G.nodes))
EOF
)
echo "Number of ground truth concepts: $num_concepts"

python ollm/experiments/baselines/llms4ol/concept_discovery/export_concepts.py \
    --raw_prediction_file out/experiments/baselines/llms4ol/wikipedia/$split/concept_discovery/categorised_pages.jsonl \
    --top_k $num_concepts \
    --output_dir out/experiments/baselines/llms4ol/wikipedia/${split}
