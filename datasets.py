import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.data = []
        self.targets = []

        for i in range(len(data_list)):
            self.data.append(data_list[i][0])
            self.targets.append(data_list[i][1])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def get_list(self):
        list = []
        for i in range(self.__len__()):
            list.append([self.data[i], int(self.targets[i])])

        return

class ImageDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_train_dataset(DATASET):
    if DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif DATASET == 'CIFAR-10':
        transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return datasets.CIFAR10(root='./data/CIFAR-10', train=True, download=True, transform=transform)
    else:
        raise NotImplementedError


def get_test_dataset(DATASET):
    if DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'CIFAR-10':
        transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return datasets.CIFAR10(root='./data/CIFAR-10', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError


def generate_train_dataset(dataset, sample_size, label_noise_ratio, dataset_path):
    clean_dataset_path = os.path.join(dataset_path, 'clean-dataset.pth')

    if not os.path.exists(clean_dataset_path):
        train_dataset = get_train_dataset(dataset)

        train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(sample_size))

        print('Saving Clean Dataset...')
        torch.save(train_dataset, clean_dataset_path)

    if label_noise_ratio > 0:
        noisy_dataset_path = os.path.join(dataset_path, 'noise-dataset-%d%%.pth' % (100 * label_noise_ratio))

        if not os.path.exists(noisy_dataset_path):
            print('Loading Clean Dataset...')
            train_dataset = torch.load(clean_dataset_path)

            label_noise_transform = transforms.Lambda(lambda y: np.random.randint(0, 10))

            num_samples = len(train_dataset)
            num_noisy_samples = int(label_noise_ratio * num_samples)

            noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
            for idx in noisy_indices:
                train_dataset.dataset.targets[idx] = label_noise_transform(train_dataset.dataset.targets[idx])

            print('Saving Noisy Dataset...')
            torch.save(train_dataset, noisy_dataset_path)

def load_train_dataset_from_file(label_noise_ratio, dataset_path):
    if label_noise_ratio == 0:
        dataset_path = os.path.join(dataset_path, 'clean-dataset.pth')
    elif label_noise_ratio > 0:
        dataset_path = os.path.join(dataset_path, 'noise-dataset-%d%%.pth' % (100 * label_noise_ratio))
    else:
        raise NotImplementedError

    if os.path.exists(dataset_path):
        return torch.load(dataset_path)
    else:
        raise NotImplementedError
