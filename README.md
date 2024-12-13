# Improving Pruning Filters for Efficient ConvNets

This repository contains my submission for the final project of the COSI 159A "Computer Vision" course at Brandeis University. The project required implementing an academic paper in the field of computer vision, and to improve upon the paper's results. I was assigned the paper "Pruning Filters for Efficient ConvNets" by Hao Li et al., 2017. The original implementation of this paper can be found in the original_implementation folder.

The paper's primary goal was to reduce the computational and storage costs of convolutional neural networks (CNNs) while preserving their performance, even as these networks become deeper and more complex. My improvements focused on applying advanced pruning techniques to further reduce the size and complexity of ConvNets without significantly compromising accuracy. A detailed qualitative discussion of the original paper and my enhancements can be found in the PDF document, "Improving Pruning Filters for Efficient ConvNets."

All code files located outside the original_implementation folder represent the improvements I made, while all code within the original_implementation folder corresponds to the original implementation of the paper. Below, you’ll find instructions on how to run the improved implementation. Instructions for running the original implementation are available in the README file within the original_implementation folder. The primary dataset used for this project is CIFAR-10.

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

The CIFAR-10 dataset is used for both training and testing the network. The dataset will be automatically downloaded when you run the training or testing script if it is not available in the specified directory. The CIFAR10 dataset is known for its ten classes of 32x32 color images.


## Run

### Configuration

Adjust the training settings such as epochs, batch size, learning rate, and the path for saving models in the args configuration in the parameter.py script.

### Training the Network

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

### Pruning the Network

To prune the trained network:

```
python main.py --prune-flag --load-path /path/to/load/model --save-path /path/to/save/pruned_model
```

### Retraining the Network after Pruning

To retrain the pruned network:

```
python main.py --retrain-flag --load-path /path/to/load/pruned_model --save-path /path/to/save/retrained_model
```

### Testing the Network

To test the performance of the network:

```
python main.py --load-path /path/to/load/model
```

## Implementation Details

Our implementation uses a variant of the VGG network adapted for CIFAR-10. It incorporates several modules:

- VGG: The network architecture, including feature extractors and classifiers.

- train_network: Handles the training process, including data loading, forward and backward passes.

- prune_network: Implements the pruning logic to selectively disable filters based on their importance to the accuracy.

- test_network: Evaluates the model's performance on the testing set.

The pruning technique aims to reduce the computational cost and model size by removing less important filters without compromising the model's predictive power.


## Improvements Over Prior Work
The code improves upon the earlier "PRUNING FILTERS FOR EFFICIENT CONVNETS" by introducing an online clustering approach during training, which dynamically clusters 
filters based on their feature similarities rather than just pruning based on magnitude or other static measures. This method aims to maintain or even enhance model 
accuracy by ensuring that only redundant filters that contribute minimally to the output accuracy are pruned. Moreover, instead of just focusing on the final layer or
fully connected layers, this method evaluates and prunes across multiple layers of the network which can lead to significant computational savings without a drop in performance.

## Original Implementation
An implementation of the original algorithm presented in "PRUNING FILTERS FOR EFFICIENT CONVNETS" can be found in the folder `original_implementation`.

## Results and Analysis
Analysis comparing our implementation with the original implementation can be found in the file `analysis.ipynb`. Results from experiments involving our implementation can be found in `results.zip` and results from the original implementation can be found in `original_implementation/results.zip`.

The results and analysis found in these files are described in more detail in the write up titled `Improving_Pruning_Filters_for_Efficient_ConvNets.pdf`.
