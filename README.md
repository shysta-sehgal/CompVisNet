# Convolutional Neural Network Comparison for Image Classification

This repository contains a Python project for comparing the performance of convolutional neural networks (CNNs) with different layers and configurations on image classification tasks. It supports experimentation with one-layer, two-layer, and three-layer CNNs on the MNIST and CIFAR-10 datasets.

## Project Structure

- `conv_config.py`: Contains the configurations the models.
- `mass_sims.py`: Script to run simulations across various model configurations.
- `net_structures.py`: Defines the CNN structures for the models.
- `test.py`: Provides functions for testing the models.
- `train.py`: Includes functions for training the models.
- `cifar_training_metrics.csv`: Includes training accuracy and loss per epoch per model configuration on CIFAR-10
- `cifar_test_results.csv`: Includes testing accuracy and loss per model configuration on CIFAR-10
- `mnist_training_metrics.csv`: Includes training accuracy and loss per epoch per model configuration on MNIST
- `mnist_test_results.csv`: Includes testing accuracy and loss per model configuration on MNIST
- `analysis.ipynb`: Includes several data visualisations for interpretation of results obtained in the CSVs

## Getting Started

To get started with this project, clone the repository to your local machine and ensure that you have the necessary dependencies installed.

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision

To train and test the models with the default settings, run the following:
python3 mass_sims.py

## Custom Configuration

You can customize the CNN configurations by editing conv_config.py according to your requirements.

## Authors

Justice Tomlinson, Shysta Sehgal 
