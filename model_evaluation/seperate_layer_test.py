import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import os.path
import csv

import datasets

epochs = 400


class Representation_Layer(nn.Module):
    def __init__(self, n):
        super(Representation_Layer, self).__init__()

        self.n = n

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n),
            nn.ReLU()
        )

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

def train_and_evaluate_model(representation_layer, classifier, device,
                             train_dataloader, test_dataloader, optimizer, criterion):

    for epoch in tqdm(range(epochs), desc="Processing"):
        # Model Training
        classifier.train()
        cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            features = representation_layer(inputs)
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

        #print("Epoch : %d ; Train Loss : %f, Train Acc : %.3f" % (epoch, train_loss, train_acc))

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
    train_dataset = datasets.get_train_dataset('MNIST')
    test_dataset = datasets.get_test_dataset('MNIST')

    train_dataloader = datasets.DataLoaderX(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = datasets.DataLoaderX(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize a representation layer for all following classification
    representation_layer = Representation_Layer(20)
    representation_layer_path = '../assets/Seperate-Layer-Test/representation_layer'

    if os.path.isfile(representation_layer_path):
        representation_layer.load_state_dict(torch.load(representation_layer_path))
    else:
        torch.save(representation_layer.state_dict(), representation_layer_path)

    representation_layer.to(device)


    # Setup Dictionary
    dictionary = {'Layers': 0, 'Parameters': 0, 'Train Loss': 0, 'Train Accuracy': 0,
                  'Test Loss': 0, 'Test Accuracy': 0, 'N1': 0, 'N2': 0, 'N3': 0}

    with open('seperate_layer_test.csv', "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
        writer.writeheader()

    # One Layer Classifier
    one_layer_classifier = One_Layer_Classifier(20)
    one_layer_classifier.to(device)
    parameters = sum(p.numel() for p in one_layer_classifier.parameters())

    # Set the optimizer and criterion
    optimizer = torch.optim.SGD(one_layer_classifier.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Train and evaluate the model
    train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(representation_layer,
                                                                          one_layer_classifier, device,
                                                                          train_dataloader, test_dataloader,
                                                                          optimizer, criterion)

    print('One_Layer_Classifier - Parameters = %d ; Train Loss = %.3f, Train Accuracy = %.2f ; '
          'Test Loss = %.3f, Test Accuracy = %.2f\n' % (parameters, train_loss, train_acc, test_loss, test_acc))

    dictionary = {'Layers': 1, 'Parameters': parameters, 'Train Loss': train_loss, 'Train Accuracy': train_acc,
                  'Test Loss': test_loss, 'Test Accuracy': test_acc, 'N1': 20, 'N2': 0, 'N3': 0}

    with open('seperate_layer_test.csv', "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
        writer.writerow(dictionary)

    # Two Layer Classifier
    for n2 in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        two_layer_classifier = Two_Layer_Classifier(20, n2)
        two_layer_classifier.to(device)
        parameters = sum(p.numel() for p in two_layer_classifier.parameters())

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(two_layer_classifier.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(representation_layer,
                                                                              two_layer_classifier, device,
                                                                              train_dataloader, test_dataloader,
                                                                              optimizer, criterion)

        print('Two_Layer_Classifier - Parameters = %d ; Train Loss = %.3f, Train Accuracy = %.2f ; '
              'Test Loss = %.3f, Test Accuracy = %.2f\n' % (parameters, train_loss, train_acc, test_loss, test_acc))

        dictionary = {'Layers': 2, 'Parameters': parameters, 'Train Loss': train_loss, 'Train Accuracy': train_acc,
                      'Test Loss': test_loss, 'Test Accuracy': test_acc, 'N1': 20, 'N2': n2, 'N3': 0}

        with open('seperate_layer_test.csv', "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
            writer.writerow(dictionary)


    # Three Layer Classifier
    for n2 in [4, 8, 12, 16, 20, 24, 28, 32]:
        n3 = n2

        three_layer_classifier = Three_Layer_Classifier(20, n2, n3)
        three_layer_classifier.to(device)
        parameters = sum(p.numel() for p in three_layer_classifier.parameters())

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(three_layer_classifier.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(representation_layer,
                                                                              three_layer_classifier, device,
                                                                              train_dataloader, test_dataloader,
                                                                              optimizer, criterion)

        print('Three_Layer_Classifier - Parameters = %d ; Train Loss = %.3f, Train Accuracy = %.2f ; '
              'Test Loss = %.3f, Test Accuracy = %.2f\n' % (parameters, train_loss, train_acc, test_loss, test_acc))

        dictionary = {'Layers': 3, 'Parameters': parameters, 'Train Loss': train_loss, 'Train Accuracy': train_acc,
                          'Test Loss': test_loss, 'Test Accuracy': test_acc, 'N1': 20, 'N2': n2, 'N3': n3}

        with open('seperate_layer_test.csv', "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
            writer.writerow(dictionary)

    # Four Layer Classifier
    for n2 in [18, 20, 22, 24]:
        n4 = n3 = n2

        four_layer_classifier = Four_Layer_Classifier(20, n2, n3, n4)
        four_layer_classifier.to(device)
        parameters = sum(p.numel() for p in four_layer_classifier.parameters())

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(four_layer_classifier.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(representation_layer,
                                                                              four_layer_classifier, device,
                                                                              train_dataloader, test_dataloader,
                                                                              optimizer, criterion)

        print('Four_Layer_Classifier - Parameters = %d ; Train Loss = %.3f, Train Accuracy = %.2f ; '
              'Test Loss = %.3f, Test Accuracy = %.2f\n' % (parameters, train_loss, train_acc, test_loss, test_acc))

        dictionary = {'Layers': 4, 'Parameters': parameters, 'Train Loss': train_loss, 'Train Accuracy': train_acc,
                      'Test Loss': test_loss, 'Test Accuracy': test_acc, 'N1': 20, 'N2': n2, 'N3': n3}

        with open('seperate_layer_test.csv', "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
            writer.writerow(dictionary)


    # Plot the Test Results
    test_result = []
    for l in range(4):
        test_result.append({'Parameters' : [], 'Train Loss' : [], 'Train Accuracy' : [],
                                                'Test Loss' : [], 'Test Accuracy' : []})

    with open('seperate_layer_test.csv', "r", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            layer = int(row['Layers']) - 1

            test_result[layer]['Parameters'].append(int(row['Parameters']))
            test_result[layer]['Train Loss'].append(float(row['Train Loss']))
            test_result[layer]['Train Accuracy'].append(float(row['Train Accuracy']))
            test_result[layer]['Test Loss'].append(float(row['Test Loss']))
            test_result[layer]['Test Accuracy'].append(float(row['Test Accuracy']))

    print(test_result)

    # Plot the Diagram
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
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

    plt.savefig('model_evaluation/evaluation_images/Seperate_Layer_Test_Result')