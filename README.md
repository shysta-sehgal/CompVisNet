# Convolutional Neural Network Comparison for Image Classification

This repository contains a Python project for comparing the performance of convolutional neural networks (CNNs) with different layers and configurations on image classification tasks. It supports experimentation with one-layer, two-layer, and three-layer CNNs on the MNIST and CIFAR-10 datasets.

## Project Structure

- `conv_config_cifar.py`: Contains the configurations for CIFAR-10 models.
- `conv_config_mnist.py`: Contains the configurations for MNIST models.
- `mass_sims.py`: Script to run simulations across various model configurations.
- `net_structures.py`: Defines the CNN structures for the models.
- `test.py`: Provides functions for testing the models.
- `train.py`: Includes functions for training the models.

## Getting Started

To get started with this project, clone the repository to your local machine and ensure that you have the necessary dependencies installed.

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision

To train and test the models with the default settings, run the following:
python mass_sims.py

## Custom Configuration

You can customize the CNN configurations by editing conv_config_cifar.py or conv_config_mnist.py according to your requirements.

## Authors

Justice Tomlinson, Shysta Sehgal 
