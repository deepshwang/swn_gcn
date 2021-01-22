# WN-GCN
Arxiv | Code

This repository is a PyTorch Implementation of our submission to ICML'21

***Towards Deep Rotation Equivariant and Invariant Representation Learning using Graph Convolutional Network***

## Abstract

*Training a Convolutional Neural Network (CNN) to be robust against rotation has mostly been done with data augmentation. In this paper, another progressive vision of research direction is highlighted to encourage independence from data augmentation by achieving structural rotational invariance of a network. We propose an equivariance-bridged SO(2) invariant network, which consists of two main parts to echo such vision. Weighted Nearest Neighbor Graph Convolutional Network (WN-GCN) is proposed to implement Graph Convolutional Network (GCN) on graph representations of images to acquire rotationally equivariant representations. Then, invariant representation is eventually obtained with Global Average Pooling (GAP) over the equivariant set of vertices retrieved from WN-GCN. Our method achieves the state-of-the-art performance on rotated MNIST and CIFAR-10 image classification, where the models are trained with a non-augmented dataset only. Then, quantitative and qualitative validations over invariance and equivariance of the representations are conducted, respectively.*
![alt text](figures/fig_problem-2.png)


## Conda Environment Set-Up

```
$ conda env create -f environment.yml
$ conda activate wn_gcn
```

## Test over Pre-trained Models

Download pre-trained model for [R-MNIST](https://kaistackr-my.sharepoint.com/:u:/g/personal/shwang_14_kaist_ac_kr/EeJa9ABKh3lHiwGB-cR97dwBYOz_k1exJOf1D-8ROFpwqQ?e=ujFg99) and [R-CIFAR-10](https://kaistackr-my.sharepoint.com/:u:/g/personal/shwang_14_kaist_ac_kr/EZZnIl_6z5ZPhBDp00rzEP0BVE99btFH9Xp9jHRJ4BZ-qg?e=hwNcgR) 

Test classification over

* R-MNIST
```
python train.py --test_only True --test_dataset 'RotNIST' --test_model_name './data/saved_models/wngcn_mnist.pth.tar'
```

* R-CIFAR-10
```
python train.py --test_only True --test_dataset 'RotCIFAR10' --test_model_name './data/saved_models/wngcn_cifar10.pth.tar'
```

## Training a Model

* MNIST
```
python train.py --train_dataset 'MNIST' --test_dataset 'RotNIST' --m 0.4 --save_bestmodel_name './data/saved_models/wngcn_mnist.pth.tar'
```

* CIFAR-10
```
python train.py --train_dataset 'CIFAR10' --test_dataset 'RotCIFAR10' --m 0.05 --save_bestmodel_name './data/saved_models/wngcn_cifar10.pth.tar'
```
