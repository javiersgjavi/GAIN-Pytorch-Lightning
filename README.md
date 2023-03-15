[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![HitCount](https://hits.dwyl.com/javiersgjavi/GAIN-Pytorch-Lightning.svg?style=flat-square&show=unique)](http://hits.dwyl.com/javiersgjavi/GAIN-Pytorch-Lightning)

# Pytorch Lightning implementation for "Generative Adversarial Imputation Networks (GAIN)"

Original authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Paper: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GAIN: Missing Data Imputation using Generative Adversarial Nets," 
International Conference on Machine Learning (ICML), 2018.
 
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf

This directory contains implementations of GAIN framework for imputation
using three UCI datasets.

-   UCI Letter (https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
-   UCI Spam (https://archive.ics.uci.edu/ml/datasets/Spambase)
-   UCI Credit (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
-   UCI Breast Cancer (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

To run the pipeline for training and evaluation on GAIN framwork, simply run 
``python3 -m main.py``.

Note that any model architecture can be used as the generator and 
discriminator model such as multi-layer perceptrons or CNNs. 

## How to run it:

### Creation of a Docker container:

If you want to run the code in a Docker container, you can use the following commands:

1. Give execution permissions to the setup.sh file:
```shell
$ chmod +x setup.sh
```
2. Run the setup.sh file:

```shell
$ ./setup.sh
```

If you have exited the container, you can access it again by running the setup.sh file again.

### Command inputs:

-   data_name: letter or credit
-   miss_rate: probability of missing components
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations

### Example command

```shell
$ python3 main.py --data_name spam 
--miss_rate: 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100
--iterations 10000
```

### Outputs

-   rmse: Root Mean Squared Error
