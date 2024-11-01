# Partial Zeroth-Order based Bilevel Optimizers
Implementations of the algorithms described in the NeurIPS 22 paper [**On the Convergence Theory for Hessian-Free Bilevel Algorithms**](https://arxiv.org/abs/2110.07004). 

## Authors

Daouda Sow, Kaiyi Ji, and Yingbin Liang 

## Quickstart

This repository is built on [hypertorch](https://github.com/prolearner/hypertorch). 
You can get started with the simple examples in IPython notebooks HyperRepresentation.ipynb and DeepHyperRepresentation.ipynb. 

Appropriate datasets will be downloaded and put into `data` folder. 

## Examples

To run the deep hyper-representation experiment with PZOBO-S algorithm on the MNIST dataset, please run the following command: 
```
python bilevel_training_mnist.py --dataset MNIST 
```
To run few-shot meta-learning experiment with PZOBO algorithm on the MiniImageNet dataset, please run the following command: 
```
python meta_learning.py --dataset miniimagenet 
```
Other supported dataset for few-shot meta-learning are Omniglot and FC100. Please, check the file `meta-learning.py` for other command-line arguments that can be set. 

## Cite 
If this code is useful for your research, please cite the following papers: 
```
@article{sow2022convergence,
  title={On the convergence theory for hessian-free bilevel algorithms},
  author={Sow, Daouda and Ji, Kaiyi and Liang, Yingbin},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={4136--4149},
  year={2022}
}
```
```
@inproceedings{grazzi2020iteration,
  title={On the Iteration Complexity of Hypergradient Computation},
  author={Grazzi, Riccardo and Franceschi, Luca and Pontil, Massimiliano and Salzo, Saverio},
  journal={Thirty-seventh International Conference on Machine Learning (ICML)},
  year={2020}
}
```



