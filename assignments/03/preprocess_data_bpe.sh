#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=C:/Users/Nadia\ Timoleon/Documents/GitHub/atmt_2023
src=fr
tgt=en
data=$base/data/$tgt-$src

# change into base directory to ensure paths are valid
cd "$base"
# create preprocessed directory
mkdir -p $data/preprocessed/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

# concatenate and preprocess training data for BPE learning
cat $data/preprocessed/train.$src $data/preprocessed/train.$tgt > $data/preprocessed/train.concat
train_file=$data/preprocessed/train.concat
num_operations=10000
codes_file=$data/preprocessed/bpe_codes
vocab_file=$data/preprocessed/vocab

# learn BPE codes
subword-nmt learn-bpe -s $num_operations -o $codes_file < $train_file

# apply BPE codes and generate vocabularies
subword-nmt apply-bpe -c $codes_file < $data/preprocessed/train.$src > $data/preprocessed/train.$src.bpe
subword-nmt apply-bpe -c $codes_file < $data/preprocessed/train.$tgt > $data/preprocessed/train.$tgt.bpe

# generate vocabularies
subword-nmt get-vocab < $data/preprocessed/train.$src.bpe > $vocab_file.$src
subword-nmt get-vocab < $data/preprocessed/train.$tgt.bpe > $vocab_file.$tgt

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
done

# preprocess all files for model training
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

# remove temporary files
rm $data/preprocessed/train.concat
rm $data/preprocessed/train.$src.bpe
rm $data/preprocessed/train.$tgt.bpe

echo "done!"
