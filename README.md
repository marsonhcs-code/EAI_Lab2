# EAI Lab 2: Model Pruning

This repository contains implementations of neural network pruning techniques for the EAI (Embedded AI) Lab 2 assignment.

## Overview

Model pruning is a technique to reduce the size and computational requirements of neural networks by removing unnecessary parameters while maintaining accuracy. This project demonstrates:

1. **Magnitude-based Pruning**: Removes weights with smallest absolute values
2. **Structured Pruning**: Removes entire filters or neurons
3. **Performance Comparison**: Evaluates accuracy vs. model size trade-offs

## Project Structure

```
.
├── model.py           # SimpleCNN model definition
├── train.py           # Training script for MNIST
├── prune.py           # Pruning implementation and evaluation
├── test_pruning.py    # Unit tests for pruning functionality
├── visualize.py       # Visualization of pruning impact
├── demo.py            # Complete demonstration workflow
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Recommended)

Run the complete demonstration to see all pruning techniques:

```bash
python demo.py
```

This provides a comprehensive walkthrough of all pruning features without requiring dataset downloads.

### 1. Quick Test (No Training Required)

To verify the pruning implementation works:

```bash
python test_pruning.py
```

To visualize the impact of pruning:

```bash
python visualize.py
```

### 2. Train the Base Model

To train a model on MNIST (requires internet to download dataset):

```bash
python train.py
```

This will:
- Download the MNIST dataset (if not already present)
- Train a simple CNN for 5 epochs
- Save the trained model to `models/mnist_cnn.pth`

### 3. Apply Pruning and Evaluate

Run the pruning script to compare different pruning strategies:

```bash
python prune.py
```

This will:
- Load the trained model
- Apply different pruning amounts (30%, 50%, 70%, 90%)
- Compare magnitude-based vs. structured pruning
- Display accuracy and sparsity metrics
- Save a 50% pruned model to `models/mnist_cnn_pruned_50.pth`

## Results

The pruning script evaluates models with different sparsity levels and compares:

- **Accuracy**: Classification accuracy on MNIST test set
- **Sparsity**: Percentage of zero parameters
- **Model Size**: Memory footprint in MB
- **Parameter Count**: Total and non-zero parameters

### Example Output

```
ORIGINAL MODEL
==============================================================
Accuracy: 98.50%
Model size: 0.4321 MB
Total parameters: 109,386
Non-zero parameters: 109,386

MAGNITUDE-BASED PRUNING
==============================================================

Pruning amount: 30%
Accuracy: 98.45% (Δ: -0.05%)
Sparsity: 30.00%
Non-zero parameters: 76,570 / 109,386

Pruning amount: 50%
Accuracy: 98.30% (Δ: -0.20%)
Sparsity: 50.00%
Non-zero parameters: 54,693 / 109,386
```

## Pruning Techniques

### Magnitude-based Pruning

Removes individual weights with the smallest absolute values:
- **Pros**: Simple, effective, achieves high sparsity
- **Cons**: Requires sparse matrix support for speedup

### Structured Pruning

Removes entire filters or neurons:
- **Pros**: Direct reduction in computation, no special hardware needed
- **Cons**: May require more careful tuning to maintain accuracy

## Key Findings

1. Models can typically be pruned by 30-50% with minimal accuracy loss
2. Magnitude-based pruning generally preserves accuracy better
3. Structured pruning provides direct computational benefits
4. Higher pruning rates (>70%) may require fine-tuning to recover accuracy

## Fine-tuning Pruned Models

For better results with high pruning rates, consider:
1. Pruning gradually (iterative pruning)
2. Fine-tuning after each pruning step
3. Using learning rate scheduling
4. Applying pruning-aware training techniques

## References

- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

## License

This project is for educational purposes as part of the EAI course.
