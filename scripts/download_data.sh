#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/sherlock

mkdir -p $data/sherlock/raw

# download the file directly to the name "data/sherlock/raw/sherlock.txt"
curl -L https://www.gutenberg.org/cache/epub/1661/pg1661.txt -o $data/sherlock/raw/sherlock.txt

# preprocess slightly: read sherlock.txt, send its contents to preprocess_raw.py, save the output as sherlock.cleaned.txt

cat $data/sherlock/raw/sherlock.txt | python $base/scripts/preprocess_raw.py > $data/sherlock/raw/sherlock.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/sherlock/raw/sherlock.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/sherlock/raw/sherlock.preprocessed.txt

# split into train, valid and test
head -n 400 $data/sherlock/raw/sherlock.preprocessed.txt > $data/sherlock/valid.txt
head -n 800 $data/sherlock/raw/sherlock.preprocessed.txt | tail -n 400 > $data/sherlock/test.txt
tail -n +801 $data/sherlock/raw/sherlock.preprocessed.txt > $data/sherlock/train.txt
