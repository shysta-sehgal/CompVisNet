from net_structures import OneLayerCNN, TwoLayerCNN, ThreeLayerCNN
from train import train_model
from test import test_model
from conv_config import one_conv_configs, two_layer_configs, three_layer_configs

def mass_test_cnn_models(trainloader, testloader, conv_configs, fc_sizes, num_classes, num_epochs=5):
    results = []

    for conv_config in conv_configs:
        for fc_size in fc_sizes:

            in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1, \
            in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2, \
            in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3 = conv_config
            
            model = ThreeLayerCNN(in_channels1, out_channels1, kernel_size1, stride1, padding1, dilation1,
                                  in_channels2, out_channels2, kernel_size2, stride2, padding2, dilation2,
                                  in_channels3, out_channels3, kernel_size3, stride3, padding3, dilation3,
                                  num_classes)
            
            print(f"Training model: conv_config={conv_config}, fc_size={fc_size}")
            train_model(model, trainloader, epochs=num_epochs)
            
            print("Testing model...")
            accuracy = test_model(model, testloader)  
            
            # Collect the results
            results.append((conv_config, fc_size, accuracy))
    
    return results


fc_sizes = [64, 128, 256] # may want to make this variable too

results = mass_test_cnn_models(trainloader, testloader, conv_configs, fc_sizes, num_epochs=5)

