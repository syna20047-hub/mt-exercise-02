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

mkdir -p $data/alice

mkdir -p $data/alice/raw

# download the file directly to the name "data/alice/raw/alice.txt"
curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o $data/alice/raw/alice.txt

# preprocess slightly: read alice.txt, send its contents to preprocess_raw.py, save the output as alice.cleaned.txt

cat $data/alice/raw/alice.txt | python $base/scripts/preprocess_raw.py > $data/alice/raw/alice.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/alice/raw/alice.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/alice/raw/alice.preprocessed.txt

# split into train, valid and test

head -n 200 $data/alice/raw/alice.preprocessed.txt > $data/alice/valid.txt
head -n 400 $data/alice/raw/alice.preprocessed.txt | tail -n 200 > $data/alice/test.txt
tail -n +401 $data/alice/raw/alice.preprocessed.txt > $data/alice/train.txt
