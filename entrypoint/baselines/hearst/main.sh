#!/bin/bash
set -e

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
exp_dir=out/experiments/baselines/hearst/$dataset/$split

python ollm/experiments/baselines/hearst/make_txt.py \
    --graph_file out/data/$dataset/final/train_graph.json \
    --graph_file out/data/$dataset/final/${split}_graph.json \
    --output_dir $exp_dir/abstracts

dir $exp_dir/abstracts/*.txt | sort -V > $exp_dir/abstract-list.txt

# Download Stanford CoreNLP
if [ ! -d corenlp/stanford-corenlp-4.5.6 ]; then
    zip_file=$(
        huggingface-cli download --revision v4.5.6 stanfordnlp/CoreNLP stanford-corenlp-latest.zip
    )
    unzip -o $zip_file -d corenlp
fi
    
java \
    -classpath "corenlp/stanford-corenlp-4.5.6/*" \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,pos,lemma,tokensregex \
    -tokensregex.rules ollm/experiments/baselines/hearst/hearst.rules \
    -fileList $exp_dir/abstract-list.txt \
    -outputDirectory $exp_dir/extractions \
    -outputFormat conll \
    -output.columns ner \
    -threads 16

# Add -noClobber if you want to avoid overwriting existing files