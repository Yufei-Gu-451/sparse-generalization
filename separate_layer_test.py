import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import csv

import data_src


class Representation_Layer(nn.Module):
    def __init__(self, n):
        super(Representation_Layer, self).__init__()
        self.n = n

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n),
            nn.ReLU()
        )

        # Random Initialization
        self.features.weight.data.uniform_(-1, 1)
        self.features.bias.data.uniform_(-1, 1)
        self.features.requires_grad_(False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out


class One_Layer_Classifier(nn.Module):
    def __init__(self, n):
        super(One_Layer_Classifier, self).__init__()

        self.n = n

        self.classifier = nn.Linear(n, 10)

    def forward(self, x):
        out = self.classifier(x)

        return out


class Two_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2):
        super(Two_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n2, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.classifier(out)

        return out


class Three_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2, n3):
        super(Three_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(n2, n3),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n3, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.classifier(out)

        return out


class Four_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2, n3, n4):
        super(Four_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(n2, n3),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(n3, n4),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n4, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.classifier(out)

        return out


def train_and_evaluate_model(representation, classifier, device, train_dataloader, test_dataloader, optimizer, criterion):
    train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0

    for epoch in tqdm(range(800), desc="Processing"):
        # Model Training
        classifier.train()
        cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            features = representation(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

        train_loss = cumulative_loss / (idx + 1)
        train_acc = correct / total

    # Test Model
    classifier.eval()
    cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = representation_layer(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / (idx + 1)
    test_acc = correct / total

    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':
    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Get train and test dataset and dataloader
    train_dataset = data_src.get_train_dataset(dataset='MNIST')
    train_dataloader = data_src.get_dataloader_from_dataset(train_dataset, 128, 0)

    test_dataset = data_src.get_test_dataset(dataset='MNIST')
    test_dataloader = data_src.get_dataloader_from_dataset(test_dataset, 128, 0)

    # Save 20 representation layers
    representation_layer_list = []
    for i in range(20):
        # Initialize a representation layer for all following classification
        representation_layer = Representation_Layer(20)
        representation_layer.to(device)

        representation_layer_list.append(representation_layer)
        torch.save(representation_layer.state_dict(), f'separate_layer_test/representation_layers/model_{i}')

    # One Layer Classifier
    for i in range(20):
        one_layer_classifier = One_Layer_Classifier(20)
        one_layer_classifier.to(device)
        parameters = sum(p.numel() for p in one_layer_classifier.parameters())

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(one_layer_classifier.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        test_results = train_and_evaluate_model(representation_layer_list[i], one_layer_classifier, device,
                                                train_dataloader, test_dataloader, optimizer, criterion)

        print('One_Layer_Classifier - Parameters = %d ; '
              'Train Loss = %.3f, Train Accuracy = %.2f ; Test Loss = %.3f, Test Accuracy = %.2f\n'
              % (parameters, test_results[0], test_results[1], test_results[2], test_results[3]))

        with open('separate_layer_test/separate_layer_test.csv', 'a') as file:
            file.write(f'1,{parameters},{test_results[0]},{test_results[1]},{test_results[2]},{test_results[3]},0')

    # Two Layer Classifier
    for i in range(20):
        for n in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
            two_layer_classifier = Two_Layer_Classifier(20, n)
            two_layer_classifier.to(device)
            parameters = sum(p.numel() for p in two_layer_classifier.parameters())

            # Set the optimizer and criterion
            optimizer = torch.optim.SGD(two_layer_classifier.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Train and evaluate the model
            test_results = train_and_evaluate_model(representation_layer_list[i], two_layer_classifier, device,
                                                    train_dataloader, test_dataloader, optimizer, criterion)

            print('Two_Layer_Classifier - Parameters = %d ; '
                  'Train Loss = %.3f, Train Accuracy = %.2f ; Test Loss = %.3f, Test Accuracy = %.2f\n'
                  % (parameters, test_results[0], test_results[1], test_results[2], test_results[3]))

            with open('separate_layer_test/separate_layer_test.csv', 'a') as file:
                file.write(f'2,{parameters},{test_results[0]},{test_results[1]},{test_results[2]},{test_results[3]},{n}')

    # Three Layer Classifier
    for i in range(20):
        for n in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
            three_layer_classifier = Three_Layer_Classifier(20, n, n)
            three_layer_classifier.to(device)
            parameters = sum(p.numel() for p in three_layer_classifier.parameters())

            # Set the optimizer and criterion
            optimizer = torch.optim.SGD(three_layer_classifier.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Train and evaluate the model
            test_results = train_and_evaluate_model(representation_layer_list[i], three_layer_classifier, device,
                                                    train_dataloader, test_dataloader, optimizer, criterion)

            print('Three_Layer_Classifier - Parameters = %d ; '
                  'Train Loss = %.3f, Train Accuracy = %.2f ; Test Loss = %.3f, Test Accuracy = %.2f\n'
                  % (parameters, test_results[0], test_results[1], test_results[2], test_results[3]))

            with open('separate_layer_test/separate_layer_test.csv', 'a') as file:
                file.write(f'3,{parameters},{test_results[0]},{test_results[1]},{test_results[2]},{test_results[3]},{n}')

    # Four Layer Classifier
    for i in range(20):
        for n in [4, 6, 8, 10, 12, 16, 18, 20, 22, 24]:
            four_layer_classifier = Four_Layer_Classifier(20, n, n, n)
            four_layer_classifier.to(device)
            parameters = sum(p.numel() for p in four_layer_classifier.parameters())

            # Set the optimizer and criterion
            optimizer = torch.optim.SGD(four_layer_classifier.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Train and evaluate the model
            test_results = train_and_evaluate_model(representation_layer_list[i], four_layer_classifier, device,
                                                    train_dataloader, test_dataloader, optimizer, criterion)

            print('Four_Layer_Classifier - Parameters = %d ; '
                  'Train Loss = %.3f, Train Accuracy = %.2f ; Test Loss = %.3f, Test Accuracy = %.2f\n'
                  % (parameters, test_results[0], test_results[1], test_results[2], test_results[3]))

            with open('separate_layer_test/separate_layer_test.csv', 'a') as file:
                file.write(f'4,{parameters},{test_results[0]},{test_results[1]},{test_results[2]},{test_results[3]},{n}')

    # ------------------------------------------------------------------------------------------------------------------

    # Plot the Test Results
    test_result = {i: {'Parameters': [], 'Dimension': [], 'Train Loss': [], 'Train Accuracy': [],
                       'Test Loss': [], 'Test Accuracy': []} for i in range(1, 5)}

    with open('separate_layer_test/separate_layer_test.csv', "r", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            layer = int(row['Layers'])

            # Record parameters and dimensions
            test_result[layer]['Parameters'].append(int(row['Parameters']))
            test_result[layer]['Dimensions'].append(int(row['Dimensions']))

            test_result[layer]['Train Loss'].append(float(row['Train Loss']))
            test_result[layer]['Train Accuracy'].append(float(row['Train Accuracy']))
            test_result[layer]['Test Loss'].append(float(row['Test Loss']))
            test_result[layer]['Test Accuracy'].append(float(row['Test Accuracy']))

    # Plot the Diagram
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.set_title(f'Experiment Results on MNIST')

    ax1.set_xlabel('Number of Model Parameters (P) (*10^3)')
    ax3.set_xlabel('Number of Model Parameters (P) (*10^3)')

    ax1.scatter(test_result[0]['Parameters'], test_result[0]['Train Accuracy'],
             label='1-layer Classifier Train Accuracy', color='red')
    ax1.plot(test_result[1]['Parameters'], test_result[1]['Train Accuracy'],
             label='2-layer Classifier Test Accuracy', color='cyan')
    ax1.plot(test_result[2]['Parameters'], test_result[2]['Train Accuracy'],
             label='3-layer Classifier Test Accuracy', color='blue')
    ax1.plot(test_result[3]['Parameters'], test_result[3]['Train Accuracy'],
             label='4-layer Classifier Test Accuracy', color='purple')

    ax3.scatter(test_result[0]['Parameters'], test_result[0]['Train Loss'],
             label='1-layer Classifier Train Loss', color='red')
    ax3.plot(test_result[1]['Parameters'], test_result[1]['Train Loss'],
             label='2-layer Classifier Train Loss', color='cyan')
    ax3.plot(test_result[2]['Parameters'], test_result[2]['Train Loss'],
             label='3-layer Classifier Train Loss', color='blue')
    ax3.plot(test_result[3]['Parameters'], test_result[3]['Train Loss'],
             label='4-layer Classifier Train Loss', color='purple')

    ax1.set_ylabel('Accuracy (100%)')
    ax3.set_ylabel('Cross Entropy Loss')

    ax1.legend(loc='lower right')
    ax1.grid()

    ax3.legend(loc='upper right')
    ax3.grid()

    plt.savefig('separate_layer_test/images_Rade/Seperate_Layer_Test_Result')