import torch
import torch.optim as optim
import csv


def train_model(model, train_loader, device, epochs, model_size, learning_rate=0.001, model_type='CNN',
                csv_file='training_metrics.csv'):
    """
    Function for training the model.
    """
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # for MNIST
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # for CIFAR-10

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        file.seek(0, 2)  
        if file.tell() == 0:
            writer.writerow(['Model Type', 'Epoch', 'Model Size', 'Total Parameters', 'Accuracy', 'Loss'])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Append epoch results to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_type, epoch + 1, model_size, sum(p.numel() for p in model.parameters()), accuracy,
                             avg_loss])

        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
        
    print('Finished Training')
