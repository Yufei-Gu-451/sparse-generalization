import torch
import matplotlib.pyplot as plt
import csv
import os

from datetime import datetime
import numpy as np
import random
import argparse

import models
import datasets

from model_evaluation import knn_test, rademacher_test, sparsity_test


# ------------------------------------------------------------------------------------------


# Return the train_dataloader and test_dataloader
def get_train_and_test_dataloader(args, dataset_path, noise_ratio):
    train_dataset = datasets.load_train_dataset_from_file(label_noise_ratio=noise_ratio,
                                                          dataset_path=dataset_path)

    train_dataloader = datasets.DataLoaderX(train_dataset, batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)

    test_dataset = datasets.get_test_dataset(DATASET=args.dataset)

    test_dataloader = datasets.DataLoaderX(test_dataset, batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    print(f'Load {args.dataset} dataset success;')

    return train_dataloader, test_dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------------------


def status_save(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, lr,
                time, curr_time, dictionary_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %.3f ; Train Acc : %.3f ; Test Loss : %.3f ; "
          "Test Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    dictionary = {'Hidden Neurons': hidden_unit, 'Epochs': epoch, 'Parameters': parameters,
                  'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Test Loss': test_loss, 'Test Accuracy': test_acc,
                  'Learning Rate': lr, 'Time Cost': time, 'Date-Time': curr_time}

    with open(dictionary_path, "a", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing to a csv file\n')


# ------------------------------------------------------------------------------------------


# Train and Evalute the model
def train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer, criterion,
                             dictionary_path, checkpoint_path, phase):
    start_time = datetime.now()

    n_parameters = sum(p.numel() for p in model.parameters())
    n_hidden_units = model.n_hidden_units

    for epoch in range(1, args.epochs + 1):
        # Model Training
        model.train()
        cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

        train_loss = cumulative_loss / (idx + 1)
        train_acc = correct / total

        lr = optimizer.param_groups[0]['lr']
        print("Phase %d : Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; Learning Rate : %f" %
             (phase, epoch, train_loss, train_acc, lr))

        if epoch % 50 == 0:
            if args.dataset == 'MNIST':
                optimizer.param_groups[0]['lr'] = args.lr / pow(1 + epoch // 50, 0.5)
            elif args.dataset == 'CIFAR-10':
                optimizer.param_groups[0]['lr'] = args.lr / pow(1 + epoch * 10, 0.5)
            else:
                raise NotImplementedError

            # Test Model
            model.eval()
            cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(test_dataloader):
                    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    cumulative_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.argmax(1)).sum().item()

            test_loss = cumulative_loss / (idx + 1)
            test_acc = correct / total

            curr_time = datetime.now()
            time = (curr_time - start_time).seconds / 60

            status_save(n_hidden_units, epoch, n_parameters, train_loss, train_acc, test_loss, test_acc, lr,
                            time, curr_time, dictionary_path=dictionary_path)

    models.save_model(model, checkpoint_path, n_hidden_units)

    return


# Plot function of Experiment Results
def plot(args, parameters, train_accuracy, test_accuracy, train_losses, test_losses, knn_5_accuracy_list):
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

    labs = [l.get_label() for l in lns]
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
    labs = [l.get_label() for l in lns]
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


# ------------------------------------------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Descent Experiment')
    parser.add_argument('-d', '--dataset', default='MNIST', choices=['MNIST', 'CIFAR-10'], type=str, help='dataset')
    parser.add_argument('-N', '--sample_size', default=4000, type=int, help='number of samples used as training data')
    parser.add_argument('-p', '--noise_ratio', type=float, help='label noise ratio')
    parser.add_argument('-m', '--model', default='SimpleFC', choices=['SimpleFC', 'CNN', 'ResNet18'], type=str,
                        help='neural network architecture')

    parser.add_argument('-s', '--start', type=int, help='starting number of test number')
    parser.add_argument('-e', '--end', type=int, help='ending number of test number')

    parser.add_argument('--hidden_units', action='append', type=int, help='hidden units / layer width')
    parser.add_argument('--epochs', default=4000, type=int, help='epochs of training time')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
    parser.add_argument('--opt', default='sgd', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')

    parser.add_argument('--task', choices=['initialize', 'train', 'test', 'rade', 'sparsity'],
                        help='what task to perform')
    parser.add_argument('--manytasks', default=False, type=bool, help='if use manytasks to run')
    parser.add_argument('--tsne', default=False, type=bool, help='perform T-SNE experiment test')
    parser.add_argument('--knn', default=True, type=bool, help='perform KNN noisy label test')
    parser.add_argument('--test_units', default=True, type=bool,
                        help='True: Use number of hidden units in plots; False: Use number of parameters in plots.')

    args = parser.parse_args()
    print(args)

    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Define Hidden Units
    if (args.task in ['initialize', 'train'] and not args.manytasks) \
            or (args.task in ['test', 'rade', 'sparsity'] and not args.tsne):
        if args.model in ['SimpleFC', 'SimpleFC_2']:
            hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            12, 14, 16, 18, 20, 22, 25, 30, 35, 40,
                            45, 50, 55, 60, 70, 80, 90, 100, 120, 150,
                            200, 400, 600, 800, 1000]#, 2000, 3000, 4000, 5000, 6000]
        elif args.model in ['CNN', 'ResNet18']:
            hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14,
                            16, 18, 20, 24, 28, 32, 36, 40, 44, 48,
                            52, 56, 60, 64]
        else:
            raise NotImplementedError
    elif (args.task in['test', 'rade'] and args.tsne):
        if args.model in ['SimpleFC', 'SimpleFC_2']:
            hidden_units = [10, 20, 100, 1000]
        elif args.model in ['CNN', 'ResNet18']:
            hidden_units = [6, 12, 48]
        else:
            raise NotImplementedError
    elif args.task in ['initialize', 'train'] and args.manytasks:
        hidden_units = args.hidden_units
    else:
        raise NotImplementedError

    print('Hidden_units = {}'.format(hidden_units))

    parameters, train_accuracy, test_accuracy, train_losses, test_losses = [], [], [], [], []
    knn_5_accuracy_list = []

    # Main Program
    for test_number in range(args.start, args.end + 1):
        # Setup seed for reproduction
        setup_seed(20 + test_number)

        # Define the roots and paths
        directory = f"assets/{args.dataset}-{args.model}/N=%d-Epoch=%d-p=%d-sgd-%d" % \
                    (args.sample_size, args.epochs, args.noise_ratio * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")
        dictionary_path = os.path.join(directory, 'dictionary')

        # Initialization
        if args.task == 'initialize':
            # Initialize the directories
            if not os.path.isdir(f"data"):
                os.mkdir(f"data")
            elif args.dataset == 'CIFAR-10' and not os.path.isdir(f"data/CIFAR-10"):
                os.mkdir((f"data/CIFAR-10"))

            if not os.path.isdir(f"assets"):
                os.mkdir(f"assets")
            if not os.path.isdir(f"assets/{args.dataset}-{args.model}"):
                os.mkdir(f"assets/{args.dataset}-{args.model}")

            if not os.path.isdir(directory):
                os.mkdir(directory)
            if not os.path.isdir(dataset_path):
                os.mkdir(dataset_path)
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            if not os.path.isdir(dictionary_path):
                os.mkdir(dictionary_path)

            datasets.generate_train_dataset(dataset=args.dataset,
                                            sample_size=args.sample_size,
                                            label_noise_ratio=args.noise_ratio,
                                            dataset_path=dataset_path)

            print('Dataset Generated for test number %d' % test_number)
        # Training
        elif args.task == 'train':
            # Get the training and testing data of specific sample size
            train_dataloader, test_dataloader = get_train_and_test_dataloader(args=args,
                                                                            dataset_path=dataset_path,
                                                                            noise_ratio=args.noise_ratio)

            # Main Training Unit
            for hidden_unit in hidden_units:
                # Generate the model with specific number of hidden_unit
                model = models.get_model(args.dataset, args.model, hidden_unit)
                model = model.to(device)

                # Set the optimizer and criterion
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
                criterion = torch.nn.CrossEntropyLoss()
                criterion = criterion.to(device)

                # Setup the dictionary file for n_hidden_unit
                dictionary_n_path = os.path.join(dictionary_path, "dictionary_%d.csv" % hidden_unit)

                dictionary = {'Hidden Neurons': 0, 'Epochs': 0, 'Parameters': 0, 'Train Loss': 0, 'Train Accuracy': 0,
                              'Test Loss': 0, 'Test Accuracy': 0, 'Learning Rate': 0, 'Time Cost': 0, 'Date-Time': 0}

                with open(dictionary_n_path, "w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
                    writer.writeheader()

                # Train and evaluate the model
                train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer,
                            criterion, dictionary_path=dictionary_n_path, checkpoint_path=checkpoint_path, phase=1)
        # Testing
        elif args.task == 'test':
            parameters.append([])
            train_accuracy.append([])
            test_accuracy.append([])
            train_losses.append([])
            test_losses.append([])

            for hidden_unit in hidden_units:
                # Get Parameters and dataset Losses
                dictionary_path_n = os.path.join(dictionary_path, "dictionary_%d.csv" % hidden_unit)

                with open(dictionary_path_n, "r", newline="") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        if int(row['Epochs']) == args.epochs:
                            parameters[-1].append(int(row['Parameters']) // 1000)
                            train_accuracy[-1].append(float(row['Train Accuracy']))
                            test_accuracy[-1].append(float(row['Test Accuracy']))
                            train_losses[-1].append(float(row['Train Loss']))
                            test_losses[-1].append(float(row['Test Loss']))

            # Run KNN Test
            if args.knn and args.noise_ratio > 0:
                knn_5_accuracy_list.append(knn_test.knn_prediction_test(directory, hidden_units, args))
        elif args.task == 'rade':
            n_complexity = rademacher_test.get_complexity(args, hidden_units, directory)
        elif args.task == 'sparsity':
            sparsity_test.sparsity_test(args, hidden_units, directory)
        else:
            raise NotImplementedError

    if args.task == 'test':
        plot(args, parameters, train_accuracy, test_accuracy, train_losses, test_losses, knn_5_accuracy_list)

    print('Program Ends!!!')
