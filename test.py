import torch
import csv


def test_model(model, test_loader, device, model_size, model_type, csv_file='test_results.csv'):
    """
    Function for testing the accuracy of the model.
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'Average test loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        file.seek(0, 2) 
        if file.tell() == 0:
            writer.writerow(['Model Size', 'Model Type', 'Average Test Loss', 'Accuracy'])
        writer.writerow([model_size, model_type, avg_loss, accuracy])

    return avg_loss, accuracy
