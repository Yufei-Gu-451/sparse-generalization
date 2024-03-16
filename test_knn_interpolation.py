import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import numpy as np
import os

import data_src
import models
from train import test_model
from plotlib import PlotLib


def get_clean_noisy_dataset_cifar(dataset_path, noise_ratio):
    # Load Clean and Noisy Dataset
    org_train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(noise_ratio * 100)}%.pth'))

    org_train_dataset = data_src.ListDataset(list(org_train_dataset))
    noisy_train_dataset = data_src.ListDataset(list(noisy_train_dataset))

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list_c, noisy_label_list_n = [], [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list_c.append((data, org_train_dataset[i][1]))
            noisy_label_list_n.append((data, noisy_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_dataset = data_src.ListDataset(clean_label_list)
    noisy_dataset_c = data_src.ListDataset(noisy_label_list_c)
    noisy_dataset_n = data_src.ListDataset(noisy_label_list_n)

    return clean_dataset, noisy_dataset_c, noisy_dataset_n, len(noisy_dataset_c)


def get_clean_noisy_dataset_mnist(dataset_path, noise_ratio):
    # Load Clean and Noisy Dataset
    org_train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(noise_ratio * 100)}%.pth'))

    # Filter Cleand and Noisy Data Index
    clean_index, noisy_index = [], []
    for i in range(4000):
        if org_train_dataset.dataset.targets[i] == noisy_train_dataset.dataset.targets[i]:
            clean_index.append(i)
        else:
            noisy_index.append(i)

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list_c, noisy_label_list_n = [], [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list_c.append((data, org_train_dataset[i][1]))
            noisy_label_list_n.append((data, noisy_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_dataset = data_src.ListDataset(clean_label_list)
    noisy_dataset_c = data_src.ListDataset(noisy_label_list_c)
    noisy_dataset_n = data_src.ListDataset(noisy_label_list_n)

    return clean_dataset, noisy_dataset_c, noisy_dataset_n, len(noisy_dataset_c)


def test(model, test_dataloader):
    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / len(test_dataloader)
    test_acc = correct/total

    return test_loss, test_acc


def knn_prediction_test(directory, hidden_units, dataset, noise_ratio, batch_size, workers, k):
    print('\nKNN Prediction Test\n')
    dataset_path = os.path.join(directory, 'dataset')

    # Load cleand and noisy dataloader
    if dataset == 'MNIST':
        clean_dataset, noisy_dataset_c, noisy_dataset_n, n_noisy_data = \
            get_clean_noisy_dataset_mnist(dataset_path, noise_ratio=noise_ratio)
    elif dataset == 'CIFAR-10':
        clean_dataset, noisy_dataset_c, noisy_dataset_n, n_noisy_data = \
            get_clean_noisy_dataset_cifar(dataset_path, noise_ratio=noise_ratio)
    else:
        raise NotImplementedError

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = data_src.get_dataloader_from_dataset(clean_dataset, batch_size, workers)
    noisy_label_dataloader_c = data_src.get_dataloader_from_dataset(noisy_dataset_c, batch_size, workers)

    knn_accuracy_list = []

    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=dataset, hidden_unit=n)
        model.eval()

        clean_act_list, clean_labels = models.get_full_activation(model, clean_label_dataloader)
        noisy_act_list, noisy_labels = models.get_full_activation(model, noisy_label_dataloader_c)

        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(clean_act_list[1], clean_labels)

        correct = sum(1 for x, y in zip(list(knn.predict(noisy_act_list[1])), noisy_labels) if x == y)
        knn_accuracy_list.append(correct / n_noisy_data)
        print('Hidden Units = %d ; Correct = %d ; k = 5' % (n, correct))

    return knn_accuracy_list


# Plot function of Experiment Results
def plot(args, hidden_units, test_result):
    # Plot the Diagram
    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.set_title(
        f'Experiment Results on {args.dataset} (N=%d, p=%d%%)' % (args.sample_size, args.noise_ratio * 100))

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units)

    if args.dataset == 'MNIST':
        ax1.set_xscale('function', functions=plotlib.scale_function)
        ax3.set_xscale('function', functions=plotlib.scale_function)

        ax3.set_ylim([0.0, 2.0])
    elif args.dataset == 'CIFAR-10':
        ax3.set_ylim([0.0, 3.0])
    else:
        raise NotImplementedError

    # Set x_labels and x_scales
    ax1.set_xlabel(plotlib.x_label)
    ax3.set_xlabel(plotlib.x_label)

    ax1.set_xticks(plotlib.x_ticks)
    ax3.set_xticks(plotlib.x_ticks)

    # Subplot 1
    if args.test_units:
        ln1 = ax1.plot(hidden_units,
                       test_result.get_train_accuracy(),
                       label='Train Accuracy', color='red')
        ln2 = ax1.plot(hidden_units,
                       test_result.get_test_accuracy(),
                       label='Test Accuracy', color='blue')
    else:
        ln1 = ax1.plot(test_result.get_parameters()[1:],
                       test_result.get_train_accuracy()[1:],
                       label='Train Accuracy', color='red')
        ln2 = ax1.plot(test_result.get_parameters()[1:],
                       test_result.get_test_accuracy()[1:],
                       label='Test Accuracy', color='blue')

    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0.0, 1.05])

    if args.knn and args.noise_ratio > 0:
        ax2 = ax1.twinx()
        if args.test_units:
            ln3 = ax2.plot(hidden_units,
                           test_result.get_knn_accuracy(),
                           label='Prediction Accuracy', color='cyan')
        else:
            ln3 = ax2.plot(test_result.get_parameters(),
                           test_result.get_knn_accuracy(),
                           label='Prediction Accuracy', color='cyan')

        ax2.set_ylabel('KNN Label Accuracy (100%)')
        ax2.set_ylim([0.0, 1.05])

        lns = ln1 + ln2 + ln3
    else:
        lns = ln1 + ln2

    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.grid()

    # Subplot 2
    if args.test_units:
        ln6 = ax3.plot(hidden_units,
                       test_result.get_train_losses(),
                       label='Train Losses', color='red')
        ln7 = ax3.plot(hidden_units,
                       test_result.get_test_losses(),
                       label='Test Losses', color='blue')
    else:
        ln6 = ax3.plot(test_result.get_parameters()[1:],
                       test_result.get_train_losses()[1:],
                       label='Train Losses', color='red')
        ln7 = ax3.plot(test_result.get_parameters()[1:],
                       test_result.get_test_losses()[1:],
                       label='Test Losses', color='blue')

    ax3.set_ylabel('Cross Entropy Loss')

    lns = ln6 + ln7
    labs = [line.get_label() for line in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid()

    # Save Figure
    if args.test_units:
        directory = f"images_1/{args.dataset}-{args.model}-Epochs=%d-p=%d-U.png" % \
                    (args.epochs, args.noise_ratio * 100)
    else:
        directory = f"images_1/{args.dataset}-{args.model}-Epochs=%d-p=%d-P.png" % \
                    (args.epochs, args.noise_ratio * 100)

    plt.savefig(directory)

    print('Program Ends!!!')
