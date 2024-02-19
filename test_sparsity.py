import numpy as np
import os

import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import rankdata
from tqdm import tqdm

import models
from test_rademacher import get_class_dataloader_from_directory


def get_activation_ratio(args, dataloader, directory, hidden_units):
    count_active_ratio_list = []

    s_value_list = []

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        data, hidden_features, predicts, true_labels = models.get_model_activation(args.dataset, model, dataloader)

        # Extract the values of the features and convert them to a list
        n_features = hidden_features.shape[0] * hidden_features.shape[1]

        # Count the portion of parameters under certain threshold
        count_active_ratio_list.append(np.count_nonzero(hidden_features != 0) / n_features)

        # Record all activated nodes and the predicted class
        activation_class = [[] for _ in range(hidden_features.shape[1])]

        for i in range(hidden_features.shape[0]):
            for j in range(hidden_features.shape[1]):
                if hidden_features[i, j] != 0:
                    activation_class[j].append(float(predicts[i]))

        s_value = []

        # Calculate the s_value based on class distribution
        # Equal distribution: S-Value = 5.5
        # Single distribution: S-Value = 1
        for j in range(hidden_features.shape[1]):
            class_counts = list(Counter(tuple(activation_class[j])).values())
            ranked_counts = len(class_counts) + 1 - rankdata(class_counts)
            s_value.append(sum([num * rank for num, rank in zip(class_counts, ranked_counts)])
                           / hidden_features.shape[0])

        s_value_list.append((np.mean(s_value) - 1) / 4.5)

    return count_active_ratio_list, s_value_list


def plot_activation_ratio(args, hidden_units, activation_ratio_list):
    # Get the activation list mean over runs
    activation_ratio_list = np.mean(activation_ratio_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.plot(np.arange(len(hidden_units)), activation_ratio_list, label='Activation Ratio', color='green')

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
    plt.savefig(f"evaluation_images/Act-Level-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))


def plot_s_value(args, hidden_units, s_value_list):
    # Get the activation list mean over runs
    s_value_list = np.mean(s_value_list, axis=0)

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.plot(np.arange(len(hidden_units)), s_value_list, label='S-Value', color='orange')

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
    plt.savefig(f"evaluation_images/S-Value-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))
    plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------


def get_similarities(matrices_list):
    # Calculate cosine similarity between each pair of matrices
    similarities = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            dot_product = np.dot(matrices_list[i], matrices_list[j].T)
            norm_matrix1 = np.linalg.norm(matrices_list[i], axis=1)
            norm_matrix2 = np.linalg.norm(matrices_list[j], axis=1)

            cosine_similarity = dot_product / (norm_matrix1[:, np.newaxis] * norm_matrix2[np.newaxis, :])
            cosine_similarity[np.isnan(cosine_similarity)] = 0
            cosine_similarity[np.isinf(cosine_similarity)] = 0

            similarities[i, j] = np.mean(cosine_similarity)

    # Calculate the sum of all values except those on the diagonal
    sum_except_diagonal = np.sum(similarities) - np.trace(similarities)

    # Calculate the number of elements in the matrix except those on the diagonal
    num_elements_except_diagonal = similarities.size - similarities.shape[0]

    # Calculate the average value
    average_except_diagonal = sum_except_diagonal / num_elements_except_diagonal

    return similarities, average_except_diagonal


def plot_heatmap(args, similarities, hidden_unit, layer_n):
    # Plot heatmap
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(similarities, cmap='hot')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Between 10 Classes')
    plt.xlabel('Matrix Index')
    plt.ylabel('Matrix Index')
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.savefig(f'evaluation_images/class_sparsity_heatmap/'
                f'Heatmap-{args.dataset}-{args.model}-Epochs=%d-p=%d-n=%d-%d.png'
                % (args.epochs, args.noise_ratio * 100, hidden_unit, layer_n))
    # plt.close(fig)


def get_activation_matrix(args, directory, hidden_units):
    # Load a list of dataloaders of all classes
    train_dataloader_list, test_dataloader_list = get_class_dataloader_from_directory(args, directory)
    similarities_1_list, similarities_2_list = [], []

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        matrices_1_list, matrices_2_list = [], []

        for c in range(10):
            train_dataloader = train_dataloader_list[c]
            test_dataloader = test_dataloader_list[c]

            # Extract the hidden_features
            data, hidden_features, outputs, predicts, true_labels = models.get_model_activation(args.dataset,
                                                                                                model,
                                                                                                test_dataloader)

            # Activation Matrices between input_layer and hidden_layer: For each test item (10000 * d * n)
            activation_matrix_1 = np.multiply(data[:, :, np.newaxis], hidden_features[:, np.newaxis, :])
            activation_matrix_1 = np.mean(activation_matrix_1, axis=0)

            # Activation Matrices between hidden_layer and output_layer: For each test item (10000 * n * H)
            activation_matrix_2 = np.multiply(hidden_features[:, :, np.newaxis], outputs[:, np.newaxis, :])
            activation_matrix_2 = np.mean(activation_matrix_2, axis=0)

            matrices_1_list.append(activation_matrix_1)
            matrices_2_list.append(activation_matrix_2)

        # Compute similarities between class activation matrices
        similarities_1, mean_1 = get_similarities(matrices_1_list)
        similarities_2, mean_2 = get_similarities(matrices_2_list)

        # print(similarities_2)

        # Plot heatmap on certain hidden_unit threshold
        if hidden_unit in [10, 20, 40, 100, 1000]:
            plot_heatmap(args, similarities_1, hidden_unit, 1)
            plot_heatmap(args, similarities_2, hidden_unit, 2)

        similarities_1_list.append(mean_1)
        similarities_2_list.append(mean_2)

    print(similarities_1_list)
    print(similarities_2_list)

    return similarities_1_list, similarities_2_list


def plot_class_activation_similarities(args, similarities_1_list, similarities_2_list, hidden_units):
    # Get the activation similarities list mean over runs
    similarities_1_list = np.mean(similarities_1_list, axis=0)
    similarities_2_list = np.mean(similarities_2_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the bar chart
    plt.plot(np.arange(len(hidden_units)), similarities_1_list, label='Input Layer / Hidden Layer)',
             color='orange')
    plt.plot(np.arange(len(hidden_units)), similarities_2_list, label='Hidden Layer / Output Layer',
             color='blue')

    # Add labels and title
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Mean Class Activation Similarity')
    plt.title(f'Class Activation Similarity of {args.model} trained on {args.dataset}')

    # Set the xticks
    if args.model in ['SimpleFC', 'SimpleFC_2']:
        xticks = [1, 5, 12, 40, 100, 1000]
    elif args.model in ['CNN', 'ResNet18']:
        xticks = [1, 8, 20, 40, 64]
    else:
        raise NotImplementedError

    index = [index for index, value in enumerate(hidden_units) if value in xticks]
    plt.xticks(index, xticks)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f"evaluation_images/Act-Sim-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
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
    plt.savefig('evaluation_images/Weight Sparsity Test')

# ------------------------------------------------------------------------------------------------------------------
