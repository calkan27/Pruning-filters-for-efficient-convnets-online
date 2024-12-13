# Pruning Filters for Efficient ConvNets

This repository contains the original implementation of the paper *"Pruning Filters for Efficient ConvNets"* by Hao Li et al., 2017. The primary goal of this work is to reduce the computational and storage costs of convolutional neural networks (CNNs) by pruning filters while maintaining model performance.

## Requirements

- Python 3
- PyTorch 1.x
- torchvision

## Installation

Clone the repository to your local machine:

```bash
git clone [repository-url]
cd [repository-directory]

## Dataset

The CIFAR-10 dataset is used for both training and testing the network. The dataset contains ten classes of 32x32 color images and will be automatically downloaded if not already available in the specified directory.

## Running the Code
### Training

To train the network from scratch:

bash
Copy code
