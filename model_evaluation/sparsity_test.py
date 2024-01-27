import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')

import models
import datasets

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

def sparsity_test(args, hidden_units, directory):
    prob_matrix = np.zeros((len(hidden_units), 3))

    for i, n in enumerate(hidden_units):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        mean, median, p_leq_01, p_leq_001, p_leq_0001 = get_weight_sparsity(model, n)

        prob_matrix[i, :] = np.array([p_leq_01, p_leq_001, p_leq_0001])

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.bar(np.arange(len(hidden_units)), prob_matrix[:, 0], label='Percentage of Weights <= 0.1',
            color='skyblue')
    plt.bar(np.arange(len(hidden_units)), prob_matrix[:, 1], label='Percentage of Weights <= 0.01',
            color='orange')
    plt.bar(np.arange(len(hidden_units)), prob_matrix[:, 2], label='Percentage of Weights <= 0.001',
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
