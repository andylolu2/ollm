#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || [ -z "$2" ] || \
    ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]) || \
    ([ $2 != "eval" ] && [ $2 != "test" ]); then
echo \
"Usage: $0 <dataset> <split>

dataset: wikipedia | arxiv
split: eval | test
"
exit 1
fi

dataset=$1
split=$2

num_concepts=$(
python - <<EOF
from ollm.dataset import data_model

G = data_model.load_graph("out/data/$dataset/final/${split}_graph.json")
print(len(G.nodes))
EOF
)
echo "Number of ground truth concepts: $num_concepts"

# dataset=$1
# split=$2

python ollm/experiments/baselines/llms4ol/concept_discovery/export_concepts.py \
    --raw_prediction_file out/experiments/baselines/llms4ol/$dataset/$split/concept_discovery/categorised_pages.jsonl \
    --top_k $(( $num_concepts * 2 )) \
    --output_dir out/experiments/baselines/llms4ol/$dataset/${split}
