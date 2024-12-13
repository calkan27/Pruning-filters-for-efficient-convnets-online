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
```
Retraining fine-tunes the pruned network to recover any lost performance.

## Testying the Network

To evaluate the network on the CIFAR-10 test set:

```
python main.py --load-path /path/to/load/model
```

## Implementation Details

This implementation is based on the methodology outlined in the original paper. Key components include:

- A version of the VGG network adapted for the CIFAR-10 dataset.
- Filters are pruned based on their importance, which is determined by their absolute magnitude.
- Scripts to train, prune, retrain, and evaluate the model.

## Results and Analysis
The results of experiments using this implementation can be found in the results.zip file in this folder. For a detailed analysis, refer to the main repository or the write-up "Improving_Pruning_Filters_for_Efficient_ConvNets.pdf" in the parent directory.


