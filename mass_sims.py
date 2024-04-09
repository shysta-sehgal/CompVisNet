import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from conv_config import one_layer_configs, two_layer_configs, three_layer_configs
from net_structures import OneLayerCNN, TwoLayerCNN, ThreeLayerCNN
from test import test_model
from train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Functions for loading MNIST and CIFAR-10 datasets


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Function for counting the trainable weights per layer and per model


def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Number of Parameters: {param.numel()}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


# Function for running simulations on different CNN architectures

def mass_test_cnn_models(train_loader, test_loader, one_layer_config, two_layer_config, three_layer_config,
                         num_classes, num_epochs):
    one_layer_results, two_layer_results, three_layer_results = [], [], []

    print("Now Training One Layer Models")
    for config in one_layer_config:
        model_type = '1 hidden'
        in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1 = config

        model = OneLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                            num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, train_loader, device, num_epochs, config, model_type=model_type)

        print("Testing model...")
        accuracy = test_model(model, test_loader, device, config, model_type=model_type)

        one_layer_results.append((config, accuracy))

    print("Now Training Two Layer Models")
    for config in two_layer_config:
        model_type = '2 hidden'

        (in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1, out_channels2, kernel_size2, stride2,
            padding2, dilation2) = config

        model = TwoLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                            out_channels1, out_channels2, kernel_size2, stride2, padding2, dilation2,
                            num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, train_loader, device, num_epochs, config, model_type=model_type)

        print("Testing model...")
        accuracy = test_model(model, test_loader, device, config, model_type=model_type)

        two_layer_results.append((config, accuracy))

    print("Now Training Three Layer Models")
    for config in three_layer_config:
        model_type = '3 hidden'

        (in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
            in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
            in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3) = config

        model = ThreeLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                              in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                              in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3,
                              num_classes)
        model.to(device)
        count_parameters(model)
        print(f"Training model: conv_config={config}")
        train_model(model, train_loader, device, num_epochs, config, model_type=model_type)

        print("Testing model...")
        accuracy = test_model(model, test_loader, device, config, model_type=model_type)

        three_layer_results.append((config, accuracy))

    return one_layer_results, two_layer_results, three_layer_results


# trainLoader, testLoader = get_mnist_loaders(batch_size=64)
trainLoader, testLoader = get_cifar10_loaders(batch_size=64)

# results_cifar = mass_test_cnn_models(trainLoader, testLoader, one_layer_configs, two_layer_configs, three_layer_configs,
#                                      num_classes=10, num_epochs=5)
results_mnist = mass_test_cnn_models(trainLoader, testLoader, one_layer_configs, two_layer_configs, three_layer_configs,
num_classes=10, num_epochs=5)
