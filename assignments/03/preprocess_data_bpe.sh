#!/bin/bash
# -*- coding: utf-8 -*-

set -e

src=fr
tgt=en
data=data/$tgt-$src

# create preprocessed directory
mkdir -p $data/bpe/preprocessed/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/bpe/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/bpe/preprocessed/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/bpe/preprocessed/tm.$src --corpus $data/bpe/preprocessed/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/bpe/preprocessed/tm.$tgt --corpus $data/bpe/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/bpe/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/bpe/preprocessed/tm.$src > $data/bpe/preprocessed/train.$src 
cat $data/bpe/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/bpe/preprocessed/tm.$tgt > $data/bpe/preprocessed/train.$tgt

# variables for BPE learning
train_file=$data/bpe/preprocessed/train
num_operations=10000
codes_file=$data/bpe/preprocessed/bpe_codes
vocab_file=$data/bpe/preprocessed/vocab

# learn byte pair encoding on the concatenation of the training text, and get resulting vocabulary for each
cat $train_file.$src $train_file.$tgt | subword-nmt learn-bpe -s $num_operations -o $codes_file
subword-nmt apply-bpe -c $codes_file < $train_file.$src | subword-nmt get-vocab > $vocab_file.$src
subword-nmt apply-bpe -c $codes_file < $train_file.$tgt | subword-nmt get-vocab > $vocab_file.$tgt

# re-apply byte pair encoding with vocabulary filter
subword-nmt apply-bpe -c $codes_file --vocabulary $vocab_file.$src --vocabulary-threshold 50 < $train_file.$src > $train_file.BPE.$src
subword-nmt apply-bpe -c $codes_file --vocabulary $vocab_file.$tgt --vocabulary-threshold 50 < $train_file.$tgt > $train_file.BPE.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/bpe/preprocessed/tm.$src > $data/bpe/preprocessed/$split.$src
    subword-nmt apply-bpe -c $codes_file --vocabulary $vocab_file.$src --vocabulary-threshold 50 < $data/bpe/preprocessed/$split.$src > $data/bpe/preprocessed/$split.BPE.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/bpe/preprocessed/tm.$tgt > $data/bpe/preprocessed/$split.$tgt
    subword-nmt apply-bpe -c $codes_file --vocabulary $vocab_file.$tgt --vocabulary-threshold 50 < $data/bpe/preprocessed/$split.$tgt > $data/bpe/preprocessed/$split.BPE.$tgt
done

# prepare all files for model training

# BPE data, build dict
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/bpe/prepared/ --train-prefix $data/bpe/preprocessed/train.BPE --valid-prefix $data/bpe/preprocessed/valid.BPE --test-prefix $data/bpe/preprocessed/test.BPE --tiny-train-prefix $data/bpe/preprocessed/tiny_train.BPE --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 

# remove temporary files
rm $data/bpe/preprocessed/train.$src.p
rm $data/bpe/preprocessed/train.$tgt.p

echo "done!"
