# Pruning Filters for Efficient ConvNets

This folder contains the original implementation of the "Pruning Filters for Efficient ConvNets" paper by Hao Li et al., 2017. The implementation demonstrates the paper's proposed method for reducing the computational and storage costs of convolutional neural networks (CNNs) through filter pruning, while maintaining high performance.

For a comparison between the original implementation and an improved version of the method, refer to the main repository. The primary dataset used in this project is CIFAR-10.

## Requirements

- Python 3
- PyTorch 1.x
- torchvision

## Installation

Clone the repository to your local machine:

```bash
git clone [repository-url]
cd [repository-directory]
```

## Dataset

The CIFAR-10 dataset is used for both training and testing the network. The dataset contains ten classes of 32x32 color images and will be automatically downloaded if not already available in the specified directory.

## Run
### Training the Network

To train the VGG network from scratch:

```
python main.py --train-flag --save-path /path/to/save/model
```

To resume training from a saved checkpoint:

```
python main.py --resume-flag --load-path /path/to/load/model --save-path /path/to/save/model
```

## Pruning the Network
To prune the trained network:
```
python main.py --retrain-flag --load-path /path/to/load/pruned_model --save-path /path/to/save/retrained_model
``
Retraining fine-tunes the pruned network to recover any lost performance.

## Testying the Network


