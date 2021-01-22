# WN-GCN

Title: *Towards Deep Rotation Equivariant and Invariant Representation Learning using Graph Convolutional Network*

![alt text] (figures/fig_problem-2.png)

This repository is a PyTorch Implementation of our submission to ICML'21.

## Abstract

*Training a Convolutional Neural Network (CNN) to be robust against rotation has mostly been done with data augmentation. In this paper, another progressive vision of research direction is highlighted to encourage independence from data augmentation by achieving structural rotational invariance of a network. We propose an equivariance-bridged SO(2) invariant network, which consists of two main parts to echo such vision. Weighted Nearest Neighbor Graph Convolutional Network (WN-GCN) is proposed to implement Graph Convolutional Network (GCN) on graph representations of images to acquire rotationally equivariant representations. Then, invariant representation is eventually obtained with Global Average Pooling (GAP) over the equivariant set of vertices retrieved from WN-GCN. Our method achieves the state-of-the-art performance on rotated MNIST and CIFAR-10 image classification, where the models are trained with a non-augmented dataset only. Then, quantitative and qualitative validations over invariance and equivariance of the representations are conducted, respectively.*


## Conda Environment Set-Up

```
$ conda env create -f environment.yml
$ conda activate wn_gcn
```
