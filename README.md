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
