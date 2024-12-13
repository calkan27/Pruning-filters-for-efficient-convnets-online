# Pruning Filters for Efficient ConvNets

This repository contains the implementation of the paper "Pruning Filters for Efficient ConvNets", focusing on the application of pruning techniques to reduce the size and complexity of Convolutional Neural Networks (ConvNets) without significant loss of accuracy. The primary dataset used in this implementation is CIFAR-10.

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

The CIFAR-10 dataset is used for both training and testing the network. The dataset will be automatically downloaded when you run the training or testing script if it is not available in the specified directory.

## Configure

Adjust the training settings such as epochs, batch size, learning rate, and the path for saving models in the args configuration in the parameter.py script.

## Training the Network

To train the network from scratch:

```
python main.py --train-flag --save-path /path/to/save/model
```

```
You can resume training from a saved model checkpoint:
```

```
python main.py --resume-flag --load-path /path/to/load/model --save-path /path/to/save/model
```

## Pruning the Network

To prune the trained network:

```
python main.py --prune-flag --load-path /path/to/load/model --save-path /path/to/save/pruned_model
```

## Retraining the Network after Pruning

To retrain the pruned network:

```
python main.py --retrain-flag --load-path /path/to/load/pruned_model --save-path /path/to/save/retrained_model
```

## Testing the Network

To test the performance of the network:

```
python main.py --load-path /path/to/load/model
```

## Implementation Details

The implementation uses a variant of the VGG network adapted for CIFAR-10. It incorporates several modules:

- VGG: The network architecture, including feature extractors and classifiers.

- train_network: Handles the training process, including data loading, forward and backward passes.

- prune_network: Implements the pruning logic to selectively disable filters based on their importance to the accuracy.

- test_network: Evaluates the model's performance on the testing set.

The pruning technique aims to reduce the computational cost and model size by removing less important filters without compromising the model's predictive power.

## Improvements over the Base Model

This implementation enhances the efficiency of the ConvNet by pruning redundant filters, potentially decreasing both inference time and space complexity, which is critical for deployment in resource-constrained environments.
