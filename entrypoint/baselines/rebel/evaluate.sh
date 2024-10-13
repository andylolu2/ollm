#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]); then
echo \
"Usage: $0 <dataset>

dataset: wikipedia | arxiv
"
exit 1
fi

dataset=$1
exp_dir=out/experiments/baselines/rebel/$dataset

# Find the best k
best_k=$(python - <<EOF
import json

best = (0, 0)  # (score, k)
for k in (5, 10, 15, 20, 25, 50, 100, 150, 200, 250):
    with open(f"$exp_dir/eval/k_{k}/hp_search.jsonl") as f:
        best = max(best, *((json.loads(line)["continuous_f1"], k) for line in f))
print(best[1])
EOF
)
echo "Best k found: $best_k"

python ollm/experiments/baselines/rebel/export_graph_with_ground_truth.py \
    --input_file $exp_dir/test/categorised_pages.jsonl \
    --graph_true out/data/$dataset/final/test_graph.json \
    --k $best_k \
    --output_dir $exp_dir/test

python ollm/eval/test_metrics.py \
    --graph $exp_dir/test/k_${best_k}/graph.json \
    --graph_true out/data/$dataset/final/test_graph.json \
    --hp_search_result $exp_dir/eval/k_${best_k}/hp_search.jsonl \
    --output_file $exp_dir/test_metrics.json
