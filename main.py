import torch
import csv
import os

from datetime import datetime
import numpy as np
import random
import argparse

import models
import data_src

import test_knn_interpolation
import test_rademacher
import test_sparsity


# ------------------------------------------------------------------------------------------


def make_directory():
    # Initialize the directories
    if not os.path.isdir(f"data"):
        os.mkdir(f"data")
    elif args.dataset == 'CIFAR-10' and not os.path.isdir(f"data/CIFAR-10"):
        os.mkdir(f"data/CIFAR-10")

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

# Return the train_dataloader and test_dataloader
def get_train_and_test_dataloader(args, dataset_path, noise_ratio):
    train_dataset = data_src.load_train_dataset_from_file(label_noise_ratio=noise_ratio,
                                                      dataset_path=dataset_path)

    train_dataloader = data_src.get_dataloader_from_dataset(train_dataset, args.batch_size, args.workers)

    test_dataset = data_src.get_test_dataset(DATASET=args.dataset)

    test_dataloader = data_src.get_dataloader_from_dataset(test_dataset, args.batch_size, args.workers)

    print(f'Load {args.dataset} dataset success;')

    return train_dataloader, test_dataloader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------------------


def init_dict(dictionary_n_path):
    dictionary = {'Hidden Neurons': 0, 'Epochs': 0, 'Parameters': 0, 'Train Loss': 0, 'Train Accuracy': 0,
                  'Test Loss': 0, 'Test Accuracy': 0, 'Learning Rate': 0, 'Time Cost': 0, 'Date-Time': 0}

    with open(dictionary_n_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
        writer.writeheader()

def save_dict(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, lr,
                time, curr_time, dict_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %.3f ; Train Acc : %.3f ; Test Loss : %.3f ; "
          "Test Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    state_dictionary = {'Hidden Neurons': hidden_unit, 'Epochs': epoch, 'Parameters': parameters,
                        'Train Loss': train_loss, 'Train Accuracy': train_acc,
                        'Test Loss': test_loss, 'Test Accuracy': test_acc,
                        'Learning Rate': lr, 'Time Cost': time, 'Date-Time': curr_time}

    with open(dict_path, "a", newline="") as file:
        # Create a writer object
        dict_writer = csv.DictWriter(file, fieldnames=state_dictionary.keys())

        # Write the data rows
        dict_writer.writerow(state_dictionary)
        print('Done writing to a csv file\n')

def read_dict(dict_path, hidden_units):
    parameters, train_accuracy, test_accuracy, train_losses, test_losses = [], [], [], [], []

    for hidden_unit in hidden_units:
        # Get Parameters and dataset Losses
        dictionary_path_n = os.path.join(dict_path, "dictionary_%d.csv" % hidden_unit)

        with open(dictionary_path_n, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if int(row['Epochs']) == args.epochs:
                    parameters.append(int(row['Parameters']) // 1000)
                    train_accuracy.append(float(row['Train Accuracy']))
                    test_accuracy.append(float(row['Test Accuracy']))
                    train_losses.append(float(row['Train Loss']))
                    test_losses.append(float(row['Test Loss']))

    return parameters, train_accuracy, test_accuracy, train_losses, test_losses

# ------------------------------------------------------------------------------------------


# Train and Evaluate the model
def train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer, criterion,
                             dict_path, checkpoint_path):
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
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; Learning Rate : %f" %
              (epoch, train_loss, train_acc, lr))

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

            read_dict(n_hidden_units, epoch, n_parameters, train_loss, train_acc, test_loss, test_acc, lr,
                      time, curr_time, dict_path=dictionary_path)

    models.save_model(model, checkpoint_path, n_hidden_units)

    return

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

    parser.add_argument('--task', choices=['initialize', 'train', 'test', 'rade', 'activ', 'matrix'],
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

    # Set the hidden_units
    if args.model in ['SimpleFC', 'SimpleFC_2']:
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        12, 14, 16, 18, 20, 22, 25, 30, 35, 40,
                        45, 50, 55, 60, 70, 80, 90, 100, 120, 150,
                        200, 400, 600, 800, 1000]  # , 2000, 3000, 4000, 5000, 6000]
        # hidden_units = [10, 20, 100, 1000]
    elif args.model in ['CNN', 'ResNet18']:
        hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14,
                        16, 18, 20, 24, 28, 32, 36, 40, 44, 48,
                        52, 56, 60, 64]
        # hidden_units = [6, 12, 48]

    print('Hidden_units = {}'.format(hidden_units))

    # Initialize the variables
    parameters_list, train_accuracy_list, test_accuracy_list, train_losses_list, test_losses_list = [], [], [], [], []

    knn_5_accuracy_list = []

    active_activation_ratio_list, s_value_list = [], []

    act_sim_1_list, act_sim_2_list = [], []

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
            make_directory()

            data_src.generate_train_dataset(dataset=args.dataset,
                                            sample_size=args.sample_size,
                                            label_noise_ratio=args.noise_ratio,
                                            dataset_path=dataset_path)

            init_dict(dictionary_path)

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

                # Initialize the dictionary file for n_hidden_unit
                dict_n_path = os.path.join(dictionary_path, "dictionary_%d.csv" % hidden_unit)
                init_dict(dict_n_path)

                # Train and evaluate the model
                train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer,
                                         criterion, dict_path=dict_n_path, checkpoint_path=checkpoint_path)
        # Testing
        elif args.task == 'test':
            parameters, train_accuracy, test_accuracy, train_losses, test_losses = read_dict(dictionary_path,
                                                                                             hidden_units)

            parameters_list.append(parameters)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            train_losses_list.append(train_losses)
            test_losses_list.append(test_losses)

            # Run KNN Test
            if args.knn and args.noise_ratio > 0:
                knn_5_accuracy_list.append(test_knn_interpolation.knn_prediction_test(directory, hidden_units, args))

        # Rademacher Complexity Estimation Test
        elif args.task == 'rade':
            n_complexity = test_rademacher.get_complexity(args, hidden_units, directory)

        # Activation Ratio Test
        elif args.task == 'activ':
            test_dataset = data_src.get_test_dataset(DATASET=args.dataset)

            test_dataloader = data_src.get_dataloader_from_dataset(test_dataset, args.batch_size, args.workers)

            active_activation_ratio, s_value = test_sparsity.get_activation_ratio(args,
                                                                                  test_dataloader,
                                                                                  directory,
                                                                                  hidden_units)

            active_activation_ratio_list.append(active_activation_ratio)
            s_value_list.append(s_value)

        elif args.task == 'matrix':
            similarities_1, similarities_2 = test_sparsity.get_activation_matrix(args, directory, hidden_units)
            act_sim_1_list.append(similarities_1)
            act_sim_2_list.append(similarities_2)
        else:
            raise NotImplementedError

    # Plot the Corresponding Graph
    if args.task == 'test':
        test_knn_interpolation.plot(args, hidden_units, parameters, train_accuracy, test_accuracy,
                                    train_losses, test_losses, knn_5_accuracy_list)

    elif args.task == 'activ':
        test_sparsity.plot_activation_ratio(args, hidden_units, active_activation_ratio_list)
        test_sparsity.plot_s_value(args, hidden_units, s_value_list)

    elif args.task == 'matrix':
        test_sparsity.plot_class_activation_similarities(args, act_sim_1_list, act_sim_2_list, hidden_units)

    print('Program Ends!!!')
