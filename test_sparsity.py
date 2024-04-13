import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import os

import models
from plotlib import PlotLib


def get_activation_ratio(args, dataloader, directory, hidden_units):
    count_active_ratio_list = []

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        act_list, _ = models.get_full_activation(model, dataloader)
        hidden_features = act_list[-2]

        # Compute the number of features
        n_features = hidden_features.shape[0] * hidden_features.shape[1]

        # Count the portion of parameters under certain threshold
        count_active_ratio_list.append(np.count_nonzero(hidden_features != 0) / n_features)

    return count_active_ratio_list


def plot_activation_ratio(args, hidden_units, activation_ratio_list):
    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units)

    # Get the activation list mean over runs
    activation_ratio_list = np.mean(activation_ratio_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # Plot the line graph
    if args.model in ['FCNN']:
        ax.set_xscale('function', functions=plotlib.scale_function)
    ax.set_xticks(plotlib.x_ticks)
    ax.plot(hidden_units, activation_ratio_list, label='Activation Ratio', color='purple')
    ax.set_ylim([0, 1.05])

    # Add a legend
    plt.legend()
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Frequency')
    plt.title(f'Activation Ratio of {args.model} trained on {args.dataset}')
    plt.grid()

    # Show the plot
    plt.savefig(f"images_Activ/Act-Ratio-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))


# ------------------------------------------------------------------------------------------------------------------

def get_dcg(class_counts):
    sorted_class_counts = sorted(class_counts, reverse=True)
    return sum(sorted_class_counts[j] / np.log2(j + 2) for j, relevance in enumerate(sorted_class_counts))


def get_neural_ndcg(args, dataloader, directory, hidden_units):
    ndcg_list = []

    uniform_dcg = get_dcg([1 for _ in range(10)]) / 10

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features and compute predicts
        act_list, _ = models.get_full_activation(model, dataloader)
        hidden_features = act_list[-2]
        predicts = np.argmax(act_list[-1], axis=1)

        # Calculate the NDCG based on class distribution
        dcg_list = []

        for j in range(hidden_features.shape[1]):
            # Calculate class counts
            activated_indices = np.nonzero(hidden_features[:, j])

            # Count class prediction frequency
            cls, class_counts = np.unique(predicts[activated_indices], return_counts=True)

            # Compute Discounted Cumulative Gain (DCG)
            dcg = get_dcg(class_counts) / sum(class_counts)
            dcg_list.append(dcg)

        ndcg_list.append((np.mean(dcg_list) - uniform_dcg) / (1 - uniform_dcg))

    return ndcg_list


def plot_neural_ndcg(args, hidden_units, ndcg_list):
    # Get the activation list mean over runs
    ndcg_list = np.mean(ndcg_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units)

    ax.set_xscale('function', functions=plotlib.scale_function)
    ax.set_xticks(plotlib.x_ticks[1:])

    # Plot the line graph
    ax.plot(hidden_units[5:], ndcg_list[5:], label='Activation Ratio', color='green')
    ax.set_ylim([0, 0.2])

    # Add a legend
    plt.legend()
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Normalized Discounted Cumulative Gain (NDCG)')
    plt.title(f'NDCG of Neurons of {args.model} trained on {args.dataset}')
    plt.grid()

    # Show the plot
    plt.savefig(f"images_Activ/NDCG-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))


# ------------------------------------------------------------------------------------------------------------------


def get_weight_sparsity(model):
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

        mean, median, p_leq_01, p_leq_001, p_leq_0001 = get_weight_sparsity(model)
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

    # Set the x_ticks
    x_ticks = [1, 5, 12, 40, 100, 1000]
    index = [index for index, value in enumerate(hidden_units) if value in [1, 5, 12, 40, 100, 1000]]
    plt.xticks(index, x_ticks)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig('Others/Weight Sparsity Test')
