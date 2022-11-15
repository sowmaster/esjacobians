# Partial Zeroth-Order based Bilevel Optimizers
Implementations of the algorithms described in the NeurIPS 22 paper [**On the Convergence Theory for Hessian-Free Bilevel Algorithms**](https://arxiv.org/abs/2110.07004). 

This repository is built on [hypertorch](https://github.com/prolearner/hypertorch). 
You can get started with the simple examples in IPython notebooks HyperRepresentation.ipynb and DeepHyperRepresentation.ipynb. 

Appropriate datasets will be downloaded and put into `data` folder. 

Examples:

To run the deep hyper-representation experiment with PZOBO-S algorithm on the MNIST dataset, please run the following command: 
```
python bilevel_training_mnist.py --dataset MNIST 
```
To run few-shot meta-learning experiment with PZOBO algorithm on the omniglot dataset, please run the following command: 
```
python meta_learning.py --dataset miniimagenet
```
Other supported dataset for few-shot meta-learning are Omniglot and FC100. Please, check the file `meta-learning.py` for other command-line arguments that can be set. 
