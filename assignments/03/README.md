# Assignment 3: Improving Low-Resource NMT
## Method 1: Byte Pair Encoding

We implemented Byte Pair Encoding (BPE) to improve the performance of the baseline model. 
We used the subword-nmt library to train the BPE model on the training data and then applied the BPE model to the training, validation and test data.
Then, we extract the vocabulary to be used by the neural network and train (1) a new model using the same hyperparameters as the baseline model (BPE_model). and (2) a new model using batch size equal to 100 (BPE_model_100), keeping the rest of the parameters the same as the baseline model.
Finally, we run inference on the test data and evaluate the model using sacreBLEU.
The cells below show the commands used to run each step of the process, differentiating between the two versions of the BPE model wherever necessary.

### Data Preprocessing
Here, we use a modified version of the original preprocessing script to apply BPE to the data.
```
bash assignments/03/preprocess_data_bpe.sh
```

### Train model
#### Batch size = 1
```
python train.py `
--data data/en-fr/bpe/prepared `
--source-lang fr `
--target-lang en `
--save-dir assignments/03/BPE_model/checkpoints
```

#### Batch size = 100
```
python train.py `
--data data/en-fr/bpe/prepared `
--source-lang fr `
--target-lang en `
--batch-size 100 `
--save-dir assignments/03/BPE_model_100/checkpoints
```


### Run inference
#### Batch size = 1
```
python translate.py `
--data data/en-fr/bpe/prepared `
--dicts data/en-fr/bpe/prepared `
--checkpoint-path assignments/03/BPE_model/checkpoints/checkpoint_last.pt `
--output assignments/03/BPE_model/tatoeba_translations.txt
```

#### Batch size = 100
```
python translate.py `
--data data/en-fr/bpe/prepared `
--dicts data/en-fr/bpe/prepared `
--checkpoint-path assignments/03/BPE_model_100/checkpoints/checkpoint_last.pt `
--output assignments/03/BPE_model_100/tatoeba_translations.txt
```

### Postprocessing
#### Batch size = 1
```
cat assignments/03/BPE_model/tatoeba_translations.txt | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l en > assignments/03/BPE_model/tatoeba_translations.p.txt
```

#### Batch size = 100
```
cat assignments/03/BPE_model_100/tatoeba_translations.txt | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l en > assignments/03/BPE_model_100/tatoeba_translations.p.txt
```

### Evaluation
#### Batch size = 1
```
cat `
assignments/03/BPE_model/tatoeba_translations.p.txt `
| sacrebleu data/en-fr/raw/test.en
```

#### Batch size = 100
```
cat `
assignments/03/BPE_model_100/tatoeba_translations.p.txt `
| sacrebleu data/en-fr/raw/test.en
```

#### Batch size = 1
```
{
 "name": "BLEU",
 "score": 15.8,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
 "verbose_score": "46.6/20.4/11.0/6.0 (BP = 1.000 ratio = 1.108 hyp_len = 4313 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.3.1"
}
```
#### Batch size = 100
```
{
 "name": "BLEU",
 "score": 17.1,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
 "verbose_score": "50.3/22.0/12.0/6.4 (BP = 1.000 ratio = 1.004 hyp_len = 3908 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.3.1"
}
```

## Method 2: Hyperparameter Tuning
In a second attempt to improve the performance of the baseline model, we decided to tune the hyperparameters of the model.
We therefore decided to use a smaller learning rate equal to $0.0001$ and an increased patience of $4$.


### Data Preprocessing
Here we use the same preprocessing script as in the baseline model.
```
bash assignments/03/preprocess_data.sh
```

### Train model
```
python train.py `
--data data/en-fr/prepared `
--source-lang fr `
--target-lang en `
--save-dir assignments/03/slow_model/checkpoints `
--lr 0.0001 `
--patience 4
```

### Run inference
```
python translate.py `
--data data/en-fr/prepared `
--dicts data/en-fr/prepared `
--checkpoint-path assignments/03/slow_model/checkpoints/checkpoint_last.pt `
--output assignments/03/slow_model/tatoeba_translations.txt
```

### Postprocessing
```
cat assignments/03/slow_model/tatoeba_translations.txt | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l en > assignments/03/slow_model/tatoeba_translations.p.txt 
```

### Evaluation
```
cat `
assignments/03/slow_model/tatoeba_translations.p.txt `
| sacrebleu data/en-fr/raw/test.en
```

The evaluation yielded the following results:
```
{
 "name": "BLEU",
 "score": 17.2,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
 "verbose_score": "47.7/23.2/12.2/6.5 (BP = 1.000 ratio = 1.222 hyp_len = 4757 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.3.1"
}
```