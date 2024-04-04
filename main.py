import argparse
import os
import random

import numpy as np
import torch

import data_src
import models
import train
from plotlib import plot_test_result

from test_knn_interpolation import knn_prediction_test
from test_rademacher import get_complexity
from test_activation import get_activation_ratio, get_activation_correlation, get_ndcg_neuron_specialization, \
    plot_activation_ratio, plot_cam_correlation, plot_ndcg_value


class TestResult:
    def __init__(self):
        self.parameters_list = []
        self.train_accuracy_list = []
        self.test_accuracy_list = []
        self.train_losses_list = []
        self.test_losses_list = []
        self.knn_accuracy_list = []
        self.rade_complexity_list = []

    def get_parameters(self):
        return np.mean(np.array(self.parameters_list), axis=0)

    def get_train_accuracy(self):
        return np.mean(np.array(self.train_accuracy_list), axis=0)

    def get_test_accuracy(self):
        return np.mean(np.array(self.test_accuracy_list), axis=0)

    def get_train_losses(self):
        return np.mean(np.array(self.train_losses_list), axis=0)

    def get_test_losses(self):
        return np.mean(np.array(self.test_losses_list), axis=0)

    def get_knn_accuracy(self):
        return np.mean(np.array(self.knn_accuracy_list), axis=0) if self.knn_accuracy_list else []

    def get_rade_complexity(self):
        return np.mean(np.array(self.rade_complexity_list), axis=0) if self.rade_complexity_list else []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Descent Experiment')
    parser.add_argument('-M', '--model', default='FCNN', choices=['FCNN', 'CNN', 'ResNet18'], type=str,
                        help='neural network architecture')
    parser.add_argument('-D', '--dataset', default='MNIST', choices=['MNIST', 'CIFAR-10'], type=str, help='dataset')
    parser.add_argument('-N', '--sample_size', default=4000, type=int, help='number of samples used as training data')
    parser.add_argument('-T', '--epochs', default=4000, type=int, help='epochs of training time')
    parser.add_argument('-p', '--noise_ratio', type=float, help='label noise ratio')

    parser.add_argument('-s', '--start', type=int, help='starting number of test number')
    parser.add_argument('-e', '--end', type=int, help='ending number of test number')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--opt', default='sgd', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate starting value')

    # parser.add_argument('--manual_bp', default=False, type=bool,
    #                     help='If Compute Backpropagation and Perform Weight Update Manually')

    parser.add_argument('--task', choices=['init', 'train', 'test', 'activ'], help='what task to perform')

    # parser.add_argument('--manytasks', default=False, type=bool, help='if use manytasks to run')
    # parser.add_argument('--hidden_units', action='append', type=int, help='hidden units used for manytasks')

    parser.add_argument('--knn', default=False, type=bool, help='perform KNN noisy label test')
    parser.add_argument('--rade', default=False, type=bool, help='perform Rademacher Complexity test')
    parser.add_argument('--test_units', default=True, type=bool,
                        help='True: Use number of hidden units in plots; False: Use number of parameters in plots.')

    args = parser.parse_args()
    print(args)

    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Set the hidden_units
    if args.model in ['FCNN']:
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        12, 14, 16, 18, 20, 22, 25, 30, 35, 40,
                        45, 50, 55, 60, 70, 80, 90, 100, 120, 150,
                        200, 400, 600, 800, 1000]
    elif args.model in ['CNN', 'ResNet18']:
        hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14,
                        16, 18, 20, 24, 28, 32, 36, 40, 44, 48,
                        52, 56, 60, 64]
    else:
        raise NotImplementedError

    print('Hidden_units = {}'.format(hidden_units))

    # Initialize the variables
    test_result = TestResult()
    active_act_ratio_list, ndcg_list, correlation_list_dict = [], [], {}

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
        if args.task == 'init':
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

            data_src.generate_train_dataset(dataset=args.dataset,
                                            sample_size=args.sample_size,
                                            label_noise_ratio=args.noise_ratio,
                                            dataset_path=dataset_path)

            print('Dataset Generated for test number %d' % test_number)
        # Training
        elif args.task == 'train':
            # Get the training and testing data of specific sample size
            train_dataloader, test_dataloader = train.get_train_and_test_dataloader(args=args,
                                                                                    dataset_path=dataset_path,
                                                                                    noise_ratio=args.noise_ratio)

            # Main Training Unit
            for hidden_unit in hidden_units:
                # Generate the model with specific number of hidden_unit
                model = models.get_model(args.model, hidden_unit)
                model = model.to(device)

                # Train and evaluate the model
                train.train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, dictionary_path,
                                               checkpoint_path, manual_bp=args.manual_bp)
        # Testing
        elif args.task == 'test':
            parameters, train_accuracy, test_accuracy, train_losses, test_losses = train.read_dict(dictionary_path,
                                                                                                   args.epochs,
                                                                                                   hidden_units)

            test_result.parameters_list.append(parameters)
            test_result.train_accuracy_list.append(train_accuracy)
            test_result.test_accuracy_list.append(test_accuracy)
            test_result.train_losses_list.append(train_losses)
            test_result.test_losses_list.append(test_losses)

            # Run k-NN Interpolation Test
            if args.knn and args.noise_ratio > 0:
                print('KNN Prediction Test')
                knn_accuracy = knn_prediction_test(directory, hidden_units, args.dataset, args.noise_ratio,
                                                   args.batch_size, args.workers, k=5)

                test_result.knn_accuracy_list.append(knn_accuracy)

            # Run Rademacher Complexity Estimation Test
            if args.rade:
                print('Rademacher Complexity Test')
                rade_complexity = get_complexity(args, hidden_units, directory)
                test_result.rade_complexity_list.append(rade_complexity)

        # Activation Ratio Test
        elif args.task == 'activ':
            test_dataset = data_src.get_test_dataset(dataset=args.dataset)
            test_dataloader = data_src.get_dataloader_from_dataset(test_dataset, args.batch_size, args.workers)

            print('Activation Ratio Test')
            active_act_ratio = get_activation_ratio(args, test_dataloader, directory, hidden_units)
            active_act_ratio_list.append(active_act_ratio)

            print('Activation NDCG Test')
            ndcg = get_ndcg_neuron_specialization(args, test_dataloader, directory, hidden_units)
            ndcg_list.append(ndcg)

            print('Activation Correlation Test')
            correlation_dict = get_activation_correlation(args, directory, hidden_units)
            correlation_list_dict['Input-Hidden'].append(correlation_dict['Input-Hidden'])
            correlation_list_dict['Hidden-Output'].append(correlation_dict['Hidden-Output'])
            # correlation_list_dict['Hidden'].append(correlation_dict['Hidden'])
        else:
            raise NotImplementedError

    # Plot the Corresponding Graph
    if args.task == 'test':
        plot_test_result(args, hidden_units, test_result)

    elif args.task == 'activ':
        plot_activation_ratio(args, hidden_units, active_act_ratio_list)
        plot_ndcg_value(args, hidden_units, ndcg_list)
        plot_cam_correlation(args, correlation_list_dict, hidden_units)

    print('Program Ends!!!')
