import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from model_evaluation import knn_test
from collections import Counter
from scipy.stats import rankdata
from tqdm import tqdm

sys.path.append('..')
import models

def get_activation_ratio(args, dataloader, directory, hidden_units):
    count_active_ratio_list = []

    s_value_list = []

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        data, hidden_features, predicts, true_labels = knn_test.get_hidden_features('MNIST', model, dataloader)

        # Extract the values of the features and convert them to a list
        n_features = hidden_features.shape[0] * hidden_features.shape[1]

        # Count the portion of parameters under certain threshold
        count_active_ratio_list.append(np.count_nonzero(hidden_features != 0) / n_features)

        # Record all activated nodes and the predicted class
        activation_class = [[] for _ in range(hidden_features.shape[1])]

        for i in range(hidden_features.shape[0]):
            for j in range(hidden_features.shape[1]):
                if hidden_features[i, j] != 0:
                    activation_class[j].append(predicts[i])

        s_value = []

        # Calculate the s_value based on class distribution
        # Equal distribution: S-Value = 5.5
        # Single distribution: S-Value = 1
        for j in range(hidden_features.shape[1]):
            class_counts = list(Counter(activation_class[j]).values())
            ranked_counts = len(class_counts) + 1 - rankdata(class_counts)
            s_value.append(sum([num * rank for num, rank in zip(class_counts, ranked_counts)]) / hidden_features.shape[0])

        s_value_list.append((np.mean(s_value) - 1) / 4.5)

    return count_active_ratio_list, s_value_list

def plot_activation_ratio(args, hidden_units, activation_ratio_list):
    # Get the activation list mean over runs
    activation_ratio_list = np.mean(activation_ratio_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.bar(np.arange(len(hidden_units)), activation_ratio_list, label='Activation Ratio', color='green')

    # Add labels and title
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Frequency')
    plt.title(f'Activation Ratio of {args.model} trained on {args.dataset}')

    # Set the xticks
    if args.model in ['SimpleFC', 'SimpleFC_2']:
        xticks = [1, 5, 12, 40, 100, 1000]
    elif args.model in ['CNN', 'ResNet18']:
        xticks = [1, 8, 20, 40, 64]
    else:
        raise NotImplementedError

    index = [index for index, value in enumerate(hidden_units) if value in xticks]
    plt.xticks(index, xticks)
    plt.ylim([0, 1])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f"model_evaluation/evaluation_images/Activation-Level-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))

def plot_s_value(args, hidden_units, s_value_list):
    # Get the activation list mean over runs
    s_value_list = np.mean(s_value_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.bar(np.arange(len(hidden_units)), s_value_list, label='S-Value', color='orange')

    # Add labels and title
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('S-Value')
    plt.title(f'Activation Ratio of {args.model} trained on {args.dataset}')

    # Set the xticks
    if args.model in ['SimpleFC', 'SimpleFC_2']:
        xticks = [1, 5, 12, 40, 100, 1000]
    elif args.model in ['CNN', 'ResNet18']:
        xticks = [1, 8, 20, 40, 64]
    else:
        raise NotImplementedError

    index = [index for index, value in enumerate(hidden_units) if value in xticks]
    plt.xticks(index, xticks)
    plt.ylim([0, 1])

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f"model_evaluation/evaluation_images/S-Value-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))







def get_weight_sparsity(model, n_hidden_units):
    state_dict = model.state_dict()

    # Get all the model parameters (weights and biases)
    features_weight = state_dict['features.1.weight']
    classifier_weight = state_dict['classifier.weight']

    # Extract the values of the parameters and convert them to a list
    weights_list = [abs(x) for x in features_weight.view(-1).tolist()] + \
                   [abs(x) for x in classifier_weight.view(-1).tolist()]
    n_parameters = len(weights_list)

    # Count the statistics
    mean = np.mean(weights_list)
    median = np.median(weights_list)

    # Count the portion of parameters under certain threshold
    p_leq_01 = sum(1 for num in weights_list if num <= 0.1) / n_parameters
    p_leq_001 = sum(1 for num in weights_list if num <= 0.01) / n_parameters
    p_leq_0001 = sum(1 for num in weights_list if num <= 0.001) / n_parameters

    return mean, median, p_leq_01, p_leq_001, p_leq_0001


def weight_sparsity_test(args, hidden_units, directory):

    weight_prob_matrix = np.zeros((len(hidden_units), 3))

    for i, n in enumerate(hidden_units):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        mean, median, p_leq_01, p_leq_001, p_leq_0001 = get_weight_sparsity(model, n)
        weight_prob_matrix[i, :] = np.array([p_leq_01, p_leq_001, p_leq_0001])

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.bar(np.arange(len(hidden_units)), weight_prob_matrix[:, 0], label='Percentage of Weights in [-0.1, 0.1]',
            color='skyblue')
    plt.bar(np.arange(len(hidden_units)), weight_prob_matrix[:, 1], label='Percentage of Weights in [-0.01, 0.01]',
            color='orange')
    plt.bar(np.arange(len(hidden_units)), weight_prob_matrix[:, 2], label='Percentage of Weights in [-0.001, 0.001]',
            color='green')

    # Add labels and title
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Frequency')
    plt.title('Bar Charts of Weight Sparsity Level')

    # Set the xticks
    xticks = [1, 5, 12, 40, 100, 1000]
    index = [index for index, value in enumerate(hidden_units) if value in xticks]
    plt.xticks(index, xticks)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig('model_evaluation/evaluation_images/Weight Sparsity Test')
