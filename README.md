# makemore-from-scratch

A deep-dive implementation and analysis of character-level language models built from first principles in PyTorch, based on Andrej Karpathy's **Building makemore** educational series.

---

## 📌 Overview

This repository contains step-by-step implementations and experiments exploring fundamental deep learning mechanics—ranging from simple bigram probability counting to building neural networks, custom activation initializations, batch normalization layers, and manual analytical backpropagation.

---

## 📚 Repository Structure & Topics Covered

| Notebook / Script | Description & Key Concepts |
| :--- | :--- |
| **`part1_bigram.ipynb`** | **Bigram Language Model & Neural Net Baseline**<br>• Character-level tokenization and dataset construction.<br>• Frequency counting via lookup matrices (`torch.Tensor`).<br>• Single-layer neural network trained with cross-entropy/NLL loss.<br>• Manual Softmax activation implementation. |
| **`part2_mlp.ipynb`** | **Multilayer Perceptron (MLP)**<br>• Bengio et al. (2003) architecture for character-level modeling.<br>• Character embeddings (`torch.nn.Embedding`) & context windowing.<br>• Non-linear activation functions (`tanh`) and hidden layer dynamics.<br>• Learning rate decay schedules and train/validation/test dataset splits. |
| **`part3_activations_bn.ipynb`** | **Activations, Initialization, & Batch Normalization**<br>• Diagnosing network dynamics: activation distributions and saturation.<br>• Weight initialization scaling (Kaiming / He initialization) to prevent vanishing/exploding gradients.<br>• Custom `BatchNorm1d` implementation from scratch (running mean/var tracking). |
| **`part4_backprop.ipynb`** | **Manual Analytical Backpropagation**<br>• Deep dive into tensor backpropagation without using `loss.backward()`.<br>• Hand-derived gradients ($\frac{\partial L}{\partial W}$) for linear layers, embeddings, activations, `BatchNorm`, and `CrossEntropy` loss.<br>• Gradient verification against PyTorch's autograd engine (`torch.autograd`). |

*(Note: Part 5 / WaveNet architecture is intentionally omitted.)*

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and PyTorch installed:

```bash
pip install torch matplotlib jupyter
