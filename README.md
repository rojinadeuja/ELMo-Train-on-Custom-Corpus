# ELMo-Train-on-Custom-Corpus

ELMo model trained on custom corpus (IMDb) using tensorflow

The tensorflow implementation of ELMo is available in this repo by AllenNLP: <a> https://github.com/allenai/bilm-tf </a>

# bilm-tf
Tensorflow implementation of the pretrained biLM used to compute ELMo
representations from ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).

This repository supports both training biLMs and using pre-trained models for prediction.

We also have a pytorch implementation available in [AllenNLP](http://allennlp.org/).

You may also find it easier to use the version provided in [Tensorflow Hub](https://www.tensorflow.org/hub/modules/google/elmo/2) if you just like to make predictions.

Citation:

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```
## Installing
Install python version 3.5 or later, tensorflow version 1.2 and h5py:

```
pip install tensorflow-gpu==1.2 h5py
python setup.py install
```

Ensure the tests pass in your environment by running:
```
python -m unittest discover tests/
```

# Training a biLM on a new corpus

Broadly speaking, the process to train and use a new biLM is:

1.  Prepare input data and a vocabulary file.
2.  Train the biLM.
3.  Test (compute the perplexity of) the biLM on heldout data.
4.  Write out the weights from the trained biLM to a hdf5 file.
5.  See the instructions above for using the output from Step #4 in downstream models.


#### 1.  Prepare input data and a vocabulary file.

Detailed Instructions Available Here: <a> https://appliedmachinelearning.blog/2019/11/30/training-elmo-from-scratch-on-custom-data-set-for-generating-embeddings-tensorflow/ </a>

To train and evaluate a biLM, you need to provide:

* a vocabulary file
* a set of training files
* a set of heldout files

The vocabulary file is a a text file with one token per line.  It must also include the special tokens `<S>`, `</S>` and `<UNK>` (case sensitive) in the file.

<i>IMPORTANT</i>: the vocabulary file should be sorted in descending order by token count in your training data.  The first three lines should be the special tokens (`<S>`, `</S>` and `<UNK>`), then the most common token in the training data, ending with the least common token.

<i>NOTE</i>: the vocabulary file used in training may differ from the one use for prediction.

The training data should be randomly split into many training files,
each containing one slice of the data.  Each file contains pre-tokenized and
white space separated text, one sentence per line.
Don't include the `<S>` or `</S>` tokens in your training data.

All tokenization/normalization is done before training a model, so both
the vocabulary file and training files should include normalized tokens.
As the default settings use a fully character based token representation, in general we do not recommend any normalization other then tokenization.

Finally, reserve a small amount of the training data as heldout data for evaluating the trained biLM.

#### 2.  Train the biLM.
The hyperparameters used to train the ELMo model can be found in `bin/train_elmo.py`.

The ELMo model was trained on 3 GPUs.
To train a new model with the same hyperparameters, first download the training data from the [1 Billion Word Benchmark](http://www.statmt.org/lm-benchmark/).
Then download the [vocabulary file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt).
Finally, run:

```
export CUDA_VISIBLE_DEVICES=0,1,2
python bin/train_elmo.py \
    --train_prefix='/path/to/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*' \
    --vocab_file /path/to/vocab-2016-09-10.txt \
    --save_dir /output_path/to/checkpoint
```

#### 3. Evaluate the trained model.

Use `bin/run_test.py` to evaluate a trained model, e.g.

```
export CUDA_VISIBLE_DEVICES=0
python bin/run_test.py \
    --test_prefix='/path/to/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-000*' \
    --vocab_file /path/to/vocab-2016-09-10.txt \
    --save_dir /output_path/to/checkpoint
```

#### 4. Convert the tensorflow checkpoint to hdf5 for prediction with `bilm` or `allennlp`.

First, create an `options.json` file for the newly trained model.  To do so,
follow the template in an existing file (e.g. the [original `options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json) and modify for your hyperpararameters.

**Important**: always set `n_characters` to 262 after training (see below).

Then Run:

```
python bin/dump_weights.py \
    --save_dir /output_path/to/checkpoint
    --outfile /output_path/to/weights.hdf5
```
