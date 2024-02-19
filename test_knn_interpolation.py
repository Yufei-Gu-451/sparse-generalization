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

    clean_dataset = data.ListDataset(clean_label_list)
    noisy_dataset_c = data.ListDataset(noisy_label_list_c)
    noisy_dataset_n = data.ListDataset(noisy_label_list_n)

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

    # Filter Cleand and Noisy Data
    clean_data = org_train_dataset.dataset.data[clean_index]
    clean_label = org_train_dataset.dataset.targets[clean_index]

    noisy_data = noisy_train_dataset.dataset.data[noisy_index]
    noisy_label_n = noisy_train_dataset.dataset.targets[noisy_index]
    noisy_label_c = org_train_dataset.dataset.targets[noisy_index]

    # Create Clean and Noisy Training Dataset
    clean_dataset = data_src.ImageDataset(clean_data, clean_label)
    noisy_dataset_n = data_src.ImageDataset(noisy_data, noisy_label_n)
    noisy_dataset_c = data_src.ImageDataset(noisy_data, noisy_label_c)

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


def knn_prediction_test(directory, hidden_units, args):
    print('\nKNN Prediction Test\n')
    dataset_path = os.path.join(directory, 'dataset')

    # Load cleand and noisy dataloader
    if args.dataset == 'MNIST':
        clean_dataset, noisy_dataset_c, noisy_dataset_n, n_noisy_data = \
            get_clean_noisy_dataset_mnist(dataset_path, noise_ratio=args.noise_ratio)
    elif args.dataset == 'CIFAR-10':
        clean_dataset, noisy_dataset_c, noisy_dataset_n, n_noisy_data = \
            get_clean_noisy_dataset_cifar(dataset_path, noise_ratio=args.noise_ratio)
    else:
        raise NotImplementedError

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = data_src.get_dataloader_from_dataset(clean_dataset, args.batch_size, args.num_workers)
    noisy_label_dataloader_c = data_src.get_dataloader_from_dataset(noisy_dataset_c, args.batch_size, args.num_workers)
    # noisy_label_dataloader_n = main.get_dataloader_from_dataset(noisy_dataset_n, args.batch_size, args.num_workers)

    # Create repository
    tsne_directory = os.path.join(directory, 'tsne')
    if not os.path.isdir(tsne_directory):
        os.mkdir(tsne_directory)
    if not os.path.isdir('images'):
        os.mkdir('images')

    knn_5_accuracy_list = []
    x_tsne = []
    y_tsne = []

    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        data, hidden_features, outputs, predicts, labels = models.get_model_activation(args.dataset, model,
                                                                                       clean_label_dataloader)
        data_2, hidden_features_2, outputs, predicts_2, labels_2 = models.get_model_activation(args.dataset, model,
                                                                                               noisy_label_dataloader_c)

        knn_5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn_5.fit(hidden_features, labels)

        correct = sum([knn_5.predict(hidden_features_2) == labels_2])
        knn_5_accuracy_list.append(correct / n_noisy_data)
        print('Hidden Units = %d ; Correct = %d ; k = 5' % (n, correct))

        # T-SNE Experiment
        if args.tsne:
            # Instantiate and fit t-SNE on the data
            tsne = TSNE(n_components=2, random_state=42)
            print(hidden_features.shape, hidden_features_2.shape)
            x_tsne.append(tsne.fit_transform(np.concatenate((hidden_features, hidden_features_2))))
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

        ax1.scatter(x_tsne[0][:800, 0], x_tsne[0][:800, 1], c=y_tsne[0][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax1.scatter(x_tsne[0][len(labels):len(labels) + 200, 0], x_tsne[0][len(labels):len(labels) + 200, 1],
                    c=y_tsne[0][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')

        ax2.scatter(x_tsne[1][:800, 0], x_tsne[1][:800, 1], c=y_tsne[1][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax2.scatter(x_tsne[1][len(labels):len(labels) + 200, 0], x_tsne[1][len(labels):len(labels) + 200, 1],
                    c=y_tsne[1][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax2.set_xlabel('t-SNE Dimension 1')

        ax3.scatter(x_tsne[2][:800, 0], x_tsne[2][:800, 1], c=y_tsne[2][:800],
                    marker='.', cmap=plt.cm.get_cmap("jet", 10))
        ax3.scatter(x_tsne[2][len(labels):len(labels) + 200, 0], x_tsne[2][len(labels):len(labels) + 200, 1],
                    c=y_tsne[2][len(labels):len(labels) + 200],
                    marker='*', cmap=plt.cm.get_cmap("jet", 10))
        ax3.set_xlabel('t-SNE Dimension 1')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("jet", 10))
        sm.set_array([])
        # cbar = fig.colorbar(sm)

        fig.suptitle('t-SNE Visualization of Learned Representations of Random Training Samples', fontsize=20)
        plt.savefig(os.path.join('images', args.model + '_tSNE_Visualization_org'))

    return knn_5_accuracy_list


# Plot function of Experiment Results
def plot(args, hidden_units, parameters, train_accuracy, test_accuracy, train_losses, test_losses, knn_5_accuracy_list):
    parameters = np.mean(np.array(parameters), axis=0)
    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    train_losses = np.mean(np.array(train_losses), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)

    if args.knn and args.noise_ratio > 0:
        knn_5_accuracy_list = np.mean(np.array(knn_5_accuracy_list), axis=0)

    # Plot the Diagram
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.set_title(
        f'Experiment Results on {args.dataset} (N=%d, p=%d%%)' % (args.sample_size, args.noise_ratio * 100))

    if args.dataset == 'MNIST':
        ax1.set_xscale('function', functions=scale_function)
        ax3.set_xscale('function', functions=scale_function)

        if args.test_units:
            ax1.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
            ax3.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
        else:
            ax1.set_xticks([1, 4, 20, 100, 500, 2000, 5000])
            ax3.set_xticks([1, 4, 20, 100, 500, 2000, 5000])

        ax3.set_ylim([0.0, 3.0])
    elif args.dataset == 'CIFAR-10':
        if args.test_units:
            ax1.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
            ax3.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])

        ax3.set_ylim([0.0, 3.5])
    else:
        raise NotImplementedError

    if args.test_units:
        if args.model in ['SimpleFC', 'SimpleFC_2']:
            ax1.set_xlabel('Number of Hidden Neurons (N)')
            ax3.set_xlabel('Number of Hidden Neurons (N)')
        elif args.model in ['CNN', 'ResNet18']:
            ax1.set_xlabel('Convolutional Layer Width (K)')
            ax3.set_xlabel('Convolutional Layer Width (K)')
        else:
            raise NotImplementedError
    else:
        ax1.set_xlabel('Number of Model Parameters (P) (*10^3)')
        ax3.set_xlabel('Number of Model Parameters (P) (*10^3)')

    # Subplot 1
    if args.test_units:
        ln1 = ax1.plot(hidden_units, train_accuracy, label='Train Accuracy', color='red')
        ln2 = ax1.plot(hidden_units, test_accuracy, label='Test Accuracy', color='blue')
    else:
        ln1 = ax1.plot(parameters[1:], train_accuracy[1:], label='Train Accuracy', color='red')
        ln2 = ax1.plot(parameters[1:], test_accuracy[1:], label='Test Accuracy', color='blue')

    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0.0, 1.05])

    if args.knn and args.noise_ratio > 0:
        ax2 = ax1.twinx()
        if args.test_units:
            ln3 = ax2.plot(hidden_units, knn_5_accuracy_list, label='Prediction Accuracy', color='cyan')
        else:
            ln3 = ax2.plot(parameters, knn_5_accuracy_list, label='Prediction Accuracy', color='cyan')
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
        ln6 = ax3.plot(hidden_units, train_losses, label='Train Losses', color='red')
        ln7 = ax3.plot(hidden_units, test_losses, label='Test Losses', color='blue')
    else:
        ln6 = ax3.plot(parameters[1:], train_losses[1:], label='Train Losses', color='red')
        ln7 = ax3.plot(parameters[1:], test_losses[1:], label='Test Losses', color='blue')

    ax3.set_ylabel('Cross Entropy Loss')

    lns = ln6 + ln7
    labs = [line.get_label() for line in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid()

    # Plot Title and Save
    if args.test_units:
        directory = f"images/{args.dataset}-{args.model}-Epochs=%d-p=%d-U.png" % \
                    (args.epochs, args.noise_ratio * 100)
    else:
        directory = f"images/{args.dataset}-{args.model}-Epochs=%d-p=%d-P.png" % \
                    (args.epochs, args.noise_ratio * 100)

    plt.savefig(directory)
