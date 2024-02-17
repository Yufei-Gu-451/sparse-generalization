import torch
import matplotlib.pyplot as plt

import numpy as np
import os

import models
import datasets


def get_class_dataloader_mnist(dataset, batch_size):
    index = [[] for _ in range(10)]
    for i in range(dataset.targets.shape[0]):
        index[dataset.targets[i]].append(i)

    dataloader_list = []

    for n in range(10):
        data = dataset.data[index[n]]
        label = dataset.targets[index[n]]

        dataset_n = datasets.ImageDataset(data, label)
        dataloader = datasets.DataLoaderX(dataset_n,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=True)

        dataloader_list.append(dataloader)

    return dataloader_list


def get_class_dataloader_cifar(dataset, batch_size):
    dataset = datasets.ListDataset(list(dataset))

    index = [[] for _ in range(10)]
    for i in range(len(dataset.targets)):
        index[dataset.targets[i]].append(i)

    dataloader_list = []

    for n in range(10):
        dataset_list = [(dataset.data[i].numpy(), dataset.targets[i]) for i in index[n]]

        dataset_n = datasets.ListDataset(dataset_list)
        dataloader = datasets.DataLoaderX(dataset_n,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=0,
                                          pin_memory=True)

        dataloader_list.append(dataloader)

    return dataloader_list


def get_hf(dataset, model, dataloader):
    # Obtain the hidden features
    hidden_features = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            hidden_feature = model(inputs.to(torch.float32), path='half1')

            for hf in hidden_feature:
                hf = hf.cpu().detach().numpy()
                hidden_features.append(hf)

    # Define image_size and feature size by Dataset
    if dataset == 'MNIST':
        feature_size = model.n_hidden_units
    elif dataset == 'CIFAR-10':
        feature_size = model.n_hidden_units * 8
    else:
        raise NotImplementedError

    # Reshape all numpy arrays
    hidden_features = np.array(hidden_features).reshape(len(hidden_features), feature_size)

    return hidden_features


def get_complexity(args, hidden_units, directory):
    print('\nRademacher Complexity Test\n')
    dataset_path = os.path.join(directory, 'dataset')

    if args.noise_ratio <= 0:
        train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    else:
        train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(args.noise_ratio * 100)}%.pth'))

    test_dataset = datasets.get_test_dataset(DATASET=args.dataset)

    # Load cleand and noisy dataloader
    if args.dataset == 'MNIST':
        train_dataloader_list = get_class_dataloader_mnist(train_dataset.dataset, batch_size=args.batch_size)

        test_dataloader_list = get_class_dataloader_mnist(test_dataset, batch_size=args.batch_size)
    elif args.dataset == 'CIFAR-10':
        train_dataloader_list = get_class_dataloader_cifar(train_dataset.dataset, batch_size=args.batch_size)

        test_dataloader_list = get_class_dataloader_cifar(test_dataset, batch_size=args.batch_size)
    else:
        raise NotImplementedError

    n_complexity_list = []
    # Compute the Rademacher Complexity
    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        complexity_list = []
        for c in range(10):
            train_hidden_feature = get_hf(args.dataset, model, train_dataloader_list[c])
            test_hidden_feature = get_hf(args.dataset, model, test_dataloader_list[c])

            hidden_feature = np.concatenate((train_hidden_feature, test_hidden_feature))

            np.random.shuffle(hidden_feature)
            split_hidden_feature = np.array_split(hidden_feature, hidden_feature.shape[0] // 50)

            for i, hf in enumerate(split_hidden_feature):
                rademacher_variables = np.random.choice([-1, 1], size=len(hf))

                complexity_list.append(np.max(np.abs(rademacher_variables @ hf)))

        # Calculate the empirical Rademacher complexity
        n_complexity_list.append(np.mean(complexity_list))

    print(n_complexity_list)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.grid()

    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    ax.plot(hidden_units, n_complexity_list)

    if args.dataset == 'MNIST':
        ax.set_xscale('function', functions=scale_function)

        ax.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    elif args.dataset == 'CIFAR-10':
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    else:
        raise NotImplementedError

    plt.savefig(f"model_evaluation/evaluation_images/Rade-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))

    return n_complexity_list
