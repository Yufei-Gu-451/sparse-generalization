import torch
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import argparse
import csv
import os

import main
import datasets
import models


def load_model(checkpoint_path, dataset, hidden_unit):
    if dataset == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif dataset == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)
    elif dataset == 'ResNet18':
        model = models.ResNet18(hidden_unit)
    else:
        raise NotImplementedError

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model


def get_clean_noisy_dataloader_cifar(dataset_path, sample_size, noise_ratio, batch_size):
    # Load Clean and Noisy Dataset
    org_train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(noise_ratio * 100)}%.pth'))

    org_train_dataset = datasets.ListDataset(list(org_train_dataset))
    noisy_train_dataset = datasets.ListDataset(list(noisy_train_dataset))

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list_c, noisy_label_list_n = [], [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list_c.append((data, org_train_dataset[i][1]))
            noisy_label_list_n.append((data, noisy_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_dataset = datasets.ListDataset(clean_label_list)
    noisy_dataset_c = datasets.ListDataset(noisy_label_list_c)
    noisy_dataset_n = datasets.ListDataset(noisy_label_list_n)

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = main.DataLoaderX(clean_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = main.DataLoaderX(noisy_dataset_c, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = main.DataLoaderX(noisy_dataset_n, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, len(noisy_dataset_c)


def get_clean_noisy_dataloader_mnist(dataset_path, sample_size, noise_ratio, batch_size):
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

    # Filter Cleand and Noisy Data
    clean_data = org_train_dataset.dataset.data[clean_index]
    clean_label = org_train_dataset.dataset.targets[clean_index]

    noisy_data = noisy_train_dataset.dataset.data[noisy_index]
    noisy_label_n = noisy_train_dataset.dataset.targets[noisy_index]
    noisy_label_c = org_train_dataset.dataset.targets[noisy_index]

    # Create Clean and Noisy Training Dataset
    clean_dataset = datasets.ImageDataset(clean_data, clean_label)
    noisy_dataset_n = datasets.ImageDataset(noisy_data, noisy_label_n)
    noisy_dataset_c = datasets.ImageDataset(noisy_data, noisy_label_c)

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = main.DataLoaderX(clean_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = main.DataLoaderX(noisy_dataset_c, batch_size=batch_size, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = main.DataLoaderX(noisy_dataset_n, batch_size=batch_size, shuffle=False,
                                                      num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, len(noisy_dataset_c)


def get_hidden_features(dataset, model, dataloader):
    # Obtain the hidden features
    data, hidden_features, predicts, true_labels = [], [], [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            hidden_feature = model(inputs.to(torch.float32), path='half1')
            outputs = model(hidden_feature, path='half2')

            for input in inputs:
                input = input.cpu().detach().numpy()
                data.append(input)

            for hf in hidden_feature:
                hf = hf.cpu().detach().numpy()
                hidden_features.append(hf)

            for output in outputs:
                predict = output.cpu().detach().numpy().argmax()
                predicts.append(predict)

            for label in labels:
                true_labels.append(label)

    # Define image_size and feature size by Dataset
    if dataset == 'MNIST':
        image_size = 28 * 28
        feature_size = model.n_hidden_units
    elif dataset == 'CIFAR-10':
        image_size = 32 * 32 * 3
        feature_size = model.n_hidden_units * 8
    else:
        raise NotImplementedError

    # Reshape all numpy arrays
    data = np.array(data).reshape(len(true_labels), image_size)
    hidden_features = np.array(hidden_features).reshape(len(true_labels), feature_size)
    predicts = np.array(predicts).reshape(len(true_labels), )
    true_labels = np.array(true_labels).reshape(len(true_labels), )

    return data, hidden_features, predicts, true_labels


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


def knn_prediction_test(directory, hidden_units, args):
    print('\nKNN Prediction Test\n')
    dataset_path = os.path.join(directory, 'dataset')

    # Load cleand and noisy dataloader
    if args.dataset == 'MNIST':
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader_mnist(dataset_path, sample_size=args.sample_size,
                                                            noise_ratio=args.noise_ratio,
                                                            batch_size=args.batch_size)
    elif args.dataset == 'CIFAR-10':
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader_cifar(dataset_path, sample_size=args.sample_size,
                                                            noise_ratio=args.noise_ratio,
                                                            batch_size=args.batch_size)
    else:
        raise NotImplementedError

    # Create repository
    tsne_directory = os.path.join(directory, 'tsne')
    if not os.path.isdir(tsne_directory):
        os.mkdir(tsne_directory)
    if not os.path.isdir('images'):
        os.mkdir('images')

    knn_5_accuracy_list = []
    X_tsne = []
    y_tsne = []

    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        # Obtain the hidden features of the clean data set
        data, hidden_features, predicts, labels = get_hidden_features(args.dataset, model, clean_label_dataloader)
        data_2, hidden_features_2, predicts_2, labels_2 = get_hidden_features(args.dataset, model, noisy_label_dataloader_c)

        knn_5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn_5.fit(hidden_features, labels)

        correct = sum(knn_5.predict(hidden_features_2) == labels_2)
        knn_5_accuracy_list.append(correct / n_noisy_data)
        print('Hidden Units = %d ; Correct = %d ; k = 5' % (n, correct))

        # T-SNE Experiment
        if args.tsne:
            # Instantiate and fit t-SNE on the data
            tsne = TSNE(n_components=2, random_state=42)
            print(hidden_features.shape, hidden_features_2.shape)
            X_tsne.append(tsne.fit_transform(np.concatenate((hidden_features, hidden_features_2))))
            y_tsne.append(np.concatenate((labels, labels_2)))

    if args.tsne:
        # Plot the t-SNE visualization
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

        if args.model == 'SimpleFC':
            ax1.set_title(args.model + ' (k=10)')
            ax2.set_title(args.model + ' (k=20)')
            ax3.set_title(args.model + ' (k=100)')
        elif args.model == 'CNN' or args.model == 'ResNet18':
            ax1.set_title(args.model + ' (k=6)')
            ax2.set_title(args.model + ' (k=12)')
            ax3.set_title(args.model + ' (k=48)')
        else:
            raise NotImplementedError

        ax1.scatter(X_tsne[0][:800, 0], X_tsne[0][:800, 1], c=y_tsne[0][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax1.scatter(X_tsne[0][len(labels):len(labels) + 200, 0], X_tsne[0][len(labels):len(labels) + 200, 1],
                    c=y_tsne[0][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')

        ax2.scatter(X_tsne[1][:800, 0], X_tsne[1][:800, 1], c=y_tsne[1][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax2.scatter(X_tsne[1][len(labels):len(labels) + 200, 0], X_tsne[1][len(labels):len(labels) + 200, 1],
                    c=y_tsne[1][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax2.set_xlabel('t-SNE Dimension 1')

        ax3.scatter(X_tsne[2][:800, 0], X_tsne[2][:800, 1], c=y_tsne[2][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax3.scatter(X_tsne[2][len(labels):len(labels) + 200, 0], X_tsne[2][len(labels):len(labels) + 200, 1],
                    c=y_tsne[2][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax3.set_xlabel('t-SNE Dimension 1')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("jet", 10))
        sm.set_array([])
        cbar = fig.colorbar(sm)

        fig.suptitle('t-SNE Visualization of Learned Representations of Random Training Samples', fontsize=20)
        plt.savefig(os.path.join('images', args.model + '_tSNE_Visualization_org'))

    return knn_5_accuracy_list
