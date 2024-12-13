# Improving Pruning Filters for Efficient ConvNets

This repository contains my submission for the final project of the COSI 159A "Computer Vision" course at Brandeis University. The project required implementing an academic paper in the field of computer vision. For extra credit, we were tasked with improving upon the paper's results. My work is based on the paper "Pruning Filters for Efficient ConvNets" by Hao Li et al., 2017. The original implementation is included in the original_implementation folder for reference, while the improved implementation is located in the main repository.

The paper's primary goal was to reduce the computational and storage costs of convolutional neural networks (CNNs) while preserving their performance, even as these networks become deeper and more complex. My improvements focused on applying advanced pruning techniques and enhancements designed to further reduce the size and complexity of convolutional neural networks (CNNs) without significantly compromising accuracy. These enhancements include dynamic online clustering during training, multi-layer pruning, and additional functionality like multi-GPU support and improved file management. A detailed discussion of the original paper and my enhancements can be found in the accompanying PDF document, "Improving_Pruning_Filters_for_Efficient_ConvNets.pdf".

The primary dataset used for this project is CIFAR-10. Instructions for running the improved implementation are outlined below. To run the original implementation, refer to the README in the original_implementation folder.

I included this project in my GitHub because it was the most intellectually challenging, labor-intensive, and open-ended project I completed as a student at Brandeis University. It also marked my first experience with academic research, where I learned to implement a paper's methodology and address gaps in external descriptions. The pursuit of improvements challenged me to think innovatively, devising novel ideas that offered meaningful enhancements. Completing this project was a pivotal milestone in my computer science journey, shaping my ability to tackle complex problems and think creatively.

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

Adjust the training settings such as epochs, batch size, learning rate, and the path for saving models in the args configuration in the parameter.py script.  The improved implementation also supports Multi-GPU Training and Dynamic Clustering for filter pruning.



### Training the Network

To train the network from scratch on a single GPU or CPU:

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
The code improves upon the earlier "Pruning Filters for Efficient ConvNets" by introducing an online clustering approach during training, which dynamically clusters 
filters based on their feature similarities rather than just pruning based on magnitude or other static measures. This method aims to maintain or even enhance model 
accuracy by ensuring that only redundant filters that contribute minimally to the output accuracy are pruned. Moreover, instead of just focusing on the final layer or
fully connected layers, this method evaluates and prunes across multiple layers of the network which can lead to significant computational savings without a drop in performance.


## Results and Analysis
Analysis comparing our implementation with the original implementation can be found in the file `analysis.ipynb`. Results from experiments involving our implementation can be found in `results.zip` and results from the original implementation can be found in `original_implementation/results.zip`.

The results and analysis found in these files are described in more detail in the write up titled `Improving_Pruning_Filters_for_Efficient_ConvNets.pdf`.
