import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from tqdm import tqdm
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


def knn_prediction_test(args, directory, hidden_units, k):
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
    clean_label_dataloader = data_src.get_dataloader_from_dataset(clean_dataset, args.batch_size, args.workers)
    noisy_label_dataloader_c = data_src.get_dataloader_from_dataset(noisy_dataset_c, args.batch_size, args.workers)
    noisy_label_dataloader_n = data_src.get_dataloader_from_dataset(noisy_dataset_n, args.batch_size, args.workers)

    knn_c_accuracy_list, knn_n_accuracy_list = [], []

    for n in tqdm(hidden_units):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, model_name=args.model,
                                  dataset_name=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        clean_act_list, clean_labels = models.get_full_activation(model, clean_label_dataloader)
        #print(clean_act_list[-2].shape, noisy_act_list[-2].shape)

        # print("1: {}".format(torch.cuda.memory_allocated(0)))
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(clean_act_list[-2], clean_labels)
        del clean_act_list, clean_labels

        # k-NN Prediction on correct labels of noisy data
        noisy_act_list, noisy_labels = models.get_full_activation(model, noisy_label_dataloader_c)
        correct_c = sum(1 for x, y in zip(list(knn.predict(noisy_act_list[-2])), noisy_labels) if x == y)
        knn_c_accuracy_list.append(correct_c / n_noisy_data)
        del noisy_act_list, noisy_labels

        # k-NN Prediction on noisy labels of noisy data
        noisy_act_list, noisy_labels = models.get_full_activation(model, noisy_label_dataloader_n)
        correct_n = sum(1 for x, y in zip(list(knn.predict(noisy_act_list[-2])), noisy_labels) if x == y)
        knn_n_accuracy_list.append(correct_n / n_noisy_data)
        del noisy_act_list, noisy_labels

        # print("2: {}\n".format(torch.cuda.memory_allocated(0)))

        print('Hidden Units = %d ; Correct_c = %d ; Correct_n = %d ; k = 5' % (n, correct_c, correct_n))

    return knn_c_accuracy_list, knn_n_accuracy_list
