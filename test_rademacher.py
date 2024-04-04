import torch
import matplotlib.pyplot as plt

import numpy as np
import os

import models
import data_src
from plotlib import PlotLib


def get_class_dataloader_mnist(dataset, batch_size):
    index = [[] for _ in range(10)]
    for i in range(dataset.targets.shape[0]):
        index[dataset.targets[i]].append(i)

    dataloader_list = []

    for n in range(10):
        dataset_list = [(dataset.data[i], dataset.targets[i]) for i in index[n]]

        dataset_n = data_src.ListDataset(dataset_list)
        dataloader = data_src.get_dataloader_from_dataset(dataset_n, batch_size, 0)

        dataloader_list.append(dataloader)

    return dataloader_list


def get_class_dataloader_cifar(dataset, batch_size):
    dataset = data_src.ListDataset(list(dataset))

    index = [[] for _ in range(10)]
    for i in range(len(dataset.targets)):
        index[dataset.targets[i]].append(i)

    dataloader_list = []

    for n in range(10):
        dataset_list = [(dataset.data[i].numpy(), dataset.targets[i]) for i in index[n]]

        dataset_n = data_src.ListDataset(dataset_list)
        dataloader = data_src.get_dataloader_from_dataset(dataset_n, batch_size, 0)

        dataloader_list.append(dataloader)

    return dataloader_list


def get_class_dataloader_from_directory(args, directory):
    dataset_path = os.path.join(directory, 'dataset')

    if args.noise_ratio <= 0:
        train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    else:
        train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(args.noise_ratio * 100)}%.pth'))

    test_dataset = data_src.get_test_dataset(dataset=args.dataset)

    # Load a list of dataloaders of all classes
    if args.dataset == 'MNIST':
        train_dataloader_list = get_class_dataloader_mnist(train_dataset.dataset, batch_size=args.batch_size)

        test_dataloader_list = get_class_dataloader_mnist(test_dataset, batch_size=args.batch_size)
    elif args.dataset == 'CIFAR-10':
        train_dataloader_list = get_class_dataloader_cifar(train_dataset.dataset, batch_size=args.batch_size)

        test_dataloader_list = get_class_dataloader_cifar(test_dataset, batch_size=args.batch_size)
    else:
        raise NotImplementedError

    return train_dataloader_list, test_dataloader_list


def get_complexity(args, hidden_units, directory):
    print('\nRademacher Complexity Test\n')

    train_dataloader_list, test_dataloader_list = get_class_dataloader_from_directory(args, directory)

    n_complexity_list = []
    # Compute the Rademacher Complexity
    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        complexity_list = []
        for c in range(10):
            train_act_list, _ = models.get_full_activation(model, test_dataloader_list[c])
            test_act_list, _ = models.get_full_activation(model, test_dataloader_list[c])

            hidden_feature = np.concatenate((train_act_list[-2], test_act_list[-2]))

            np.random.shuffle(hidden_feature)
            split_hidden_feature = np.array_split(hidden_feature, hidden_feature.shape[0] // 50)

            for i, hf in enumerate(split_hidden_feature):
                rademacher_variables = np.random.choice([-1, 1], size=len(hf))

                complexity_list.append(np.max(np.abs(rademacher_variables @ hf)))

        # Calculate the empirical Rademacher complexity
        n_complexity_list.append(np.mean(complexity_list))

    print(n_complexity_list)

    return n_complexity_list


def plot_complexity(args, hidden_units, rademacher_complexity_list):
    rademacher_complexity_list = np.mean(rademacher_complexity_list, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.grid()

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units)

    ax.set_xscale('function', functions=plotlib.scale_function)
    ax.set_xticks(plotlib.x_ticks)

    ax.plot(hidden_units, rademacher_complexity_list, color='blue')

    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Estimated Rademacher Complexity')
    plt.title(f'Rademacher Complexity estimate of class functions of {args.model} trained on {args.dataset}')

    plt.savefig(f"images_2/Rade-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))
