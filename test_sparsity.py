import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.stats import rankdata
from tqdm import tqdm

import models
from test_rademacher import get_class_dataloader_from_directory
from plot import Plot


def get_activation_ratio(args, dataloader, directory, hidden_units):
    count_active_ratio_list = []

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        hidden_features = []

        for idx, (inputs, labels) in enumerate(dataloader):
            act_list = model.forward_full(inputs)
            hidden_features.append(act_list[1].cpu().detach().numpy())

        hidden_features = np.concatenate(hidden_features)

        # Extract the values of the features and convert them to a list
        n_features = hidden_features.shape[0] * hidden_features.shape[1]

        # Count the portion of parameters under certain threshold
        count_active_ratio_list.append(np.count_nonzero(hidden_features != 0) / n_features)

    return count_active_ratio_list


def plot_activation_ratio(args, hidden_units, activation_ratio_list):
    plot_setting = Plot(args.model, args.dataset, hidden_units)

    # Get the activation list mean over runs
    activation_ratio_list = np.mean(activation_ratio_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot the line graph
    ax.set_xscale('function', functions=plot_setting.scale_function)
    ax.set_xticks(plot_setting.x_ticks)
    ax.plot(hidden_units, activation_ratio_list, label='Activation Ratio', color='green')
    ax.set_ylim([0, 1])

    # Add a legend
    plt.legend()
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Frequency')
    plt.title(f'Activation Ratio of {args.model} trained on {args.dataset}')

    # Show the plot
    plt.savefig(f"evaluation_images/Act-Ratio-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))


# ------------------------------------------------------------------------------------------------------------------


def get_ndcg_neuron_specialization(args, dataloader, directory, hidden_units):
    ndcg_list = []

    uniform_counts = [1000 for _ in range(10)]
    uniform_weights = rankdata(uniform_counts, method='min')
    uniform_dcg = sum([num * rank for num, rank in zip(uniform_counts, uniform_weights)]) / 10000

    specific_counts = [10000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    specific_weights = rankdata(specific_counts, method='min')
    specific_dcg = sum([num * rank for num, rank in zip(specific_counts, specific_weights)]) / 10000

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        hidden_features, predicts = [], []

        for idx, (inputs, labels) in enumerate(dataloader):
            act_list = model.forward_full(inputs)
            hidden_features.append(act_list[1].cpu().detach().numpy())
            predicts.append(np.argmax(act_list[2].cpu().detach().numpy(), axis=1))

        hidden_features = np.concatenate(hidden_features)
        predicts = np.concatenate(predicts)

        # Calculate the NDCG based on class distribution
        ndcg = []

        for j in range(hidden_features.shape[1]):
            # Calculate class counts
            activated_indices = np.nonzero(hidden_features[:, j])

            cls, class_counts = np.unique(predicts[activated_indices], return_counts=True)

            # Apply a discounting factor (e.g., logarithmic discounting)
            ranked_counts_weights = rankdata(class_counts, method='min')
            # print(class_counts, np.sum(class_counts))

            # Compute Discounted Cumulative Gain (DCG)
            dcg = sum([num * weight for num, weight in zip(class_counts, ranked_counts_weights)]) / sum(class_counts)
            ndcg.append(dcg)

            # print(ranked_counts_weights, dcg)

        ndcg_list.append(np.mean(ndcg))

    print(uniform_dcg, specific_dcg)
    print(ndcg_list)

    return ndcg_list


def plot_ndcg_value(args, hidden_units, ndcg_list):
    # Get the activation list mean over runs
    ndcg_list = np.mean(ndcg_list, axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plot_setting = Plot(args.model, args.dataset, hidden_units)
    ax.set_xscale('function', functions=plot_setting.scale_function)
    ax.set_xticks(plot_setting.x_ticks)

    # Plot the line graph
    ax.plot(hidden_units, ndcg_list, label='Activation Ratio', color='green')

    # Add a legend
    plt.legend()
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Normalized Discounted Cumulative Gain (NDCG)')
    plt.title(f'NDCG of Neurons of {args.model} trained on {args.dataset}')

    # Show the plot
    plt.savefig(f"evaluation_images/NDCG-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))

    # plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------


def get_cosine_similarity(matrix1, matrix2):
    # Flatten the matrices to 1D arrays
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()

    # Calculate dot product
    dot_product = np.dot(flattened_matrix1, flattened_matrix2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(flattened_matrix1)
    magnitude2 = np.linalg.norm(flattened_matrix2)

    # Calculate cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity


def get_pearson_correlation(matrix1, matrix2):
    # Flatten the matrices to 1D arrays
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()

    # Calculate means
    mean1 = np.mean(flattened_matrix1)
    mean2 = np.mean(flattened_matrix2)

    # Calculate standard deviations
    std_dev1 = np.std(flattened_matrix1)
    std_dev2 = np.std(flattened_matrix2)

    # Calculate covariance
    covariance = np.mean((flattened_matrix1 - mean1) * (flattened_matrix2 - mean2))

    # Calculate correlation coefficient
    correlation_coefficient = covariance / (std_dev1 * std_dev2)

    return correlation_coefficient


def compute_correlation(matrices_list, similarity_measure='cosine'):
    # Calculate cosine similarity between each pair of matrices
    similarities = np.zeros((10, 10))
    for i in range(10):
        for j in range(i + 1, 10):
            if similarity_measure == 'cosine':
                similarities[i, j] = get_cosine_similarity(matrices_list[i], matrices_list[j])
            elif similarity_measure == 'pearson':
                similarities[i, j] = get_pearson_correlation(matrices_list[i], matrices_list[j])

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


def get_activation_correlation(args, directory, hidden_units):
    # Load a list of dataloaders of all classes
    train_dataloader_list, test_dataloader_list = get_class_dataloader_from_directory(args, directory)
    correlation_dict = {'Input-Hidden': [], 'Hidden': [], 'Hidden-Output': []}

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        cam1_list, cam2_list, hf_list = [], [], []

        for c in range(10):
            # Extract the hidden_features
            data, hidden_features, outputs, predicts, true_labels = models.get_model_activation(args.dataset,
                                                                                                model,
                                                                                                test_dataloader_list[c])

            # Activation Matrices between input_layer and hidden_layer: For each test item (10000 * d * n)
            cam1 = np.mean(np.multiply(data[:, :, np.newaxis], hidden_features[:, np.newaxis, :]), axis=0)
            cam2 = np.mean(np.multiply(hidden_features[:, :, np.newaxis], outputs[:, np.newaxis, :]), axis=0)
            hf = np.mean(hidden_features, axis=0)

            # Append all information to the class-wise list
            cam1_list.append(cam1)
            cam2_list.append(cam2)
            hf_list.append(hf)

        # Compute similarities between class activation matrices
        corr_1, mean_1 = compute_correlation(cam1_list)
        corr_2, mean_2 = compute_correlation(cam2_list)
        corr_hf, mean_hf = compute_correlation(hf_list)

        # Plot heatmap on certain hidden_unit threshold
        if hidden_unit in [10, 20, 40, 100, 1000]:
            plot_heatmap(args, corr_1, hidden_unit, 1)
            plot_heatmap(args, corr_2, hidden_unit, 2)

        correlation_dict['Input-Hidden'].append(mean_1)
        correlation_dict['Hidden-Output'].append(mean_2)
        correlation_dict['Hidden'].append(mean_hf)

    print(correlation_dict['Hidden'])
    return correlation_dict


def plot_class_activation_similarities(args, correlation_dict, hidden_units):
    # Get the activation similarities list mean over runs
    cam_1_list = np.mean(correlation_dict['Input-Hidden'], axis=0)
    cam_2_list = np.mean(correlation_dict['Hidden-Output'], axis=0)
    hf_list = np.mean(correlation_dict['Hidden'], axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plot_setting = Plot(args.model, args.dataset, hidden_units)
    ax.set_xscale('function', functions=plot_setting.scale_function)
    ax.set_xticks(plot_setting.x_ticks)

    # Plot the line chart
    plt.plot(hidden_units, cam_1_list, label='Input Layer / Hidden Layer)',
             color='orange')
    plt.plot(hidden_units, cam_2_list, label='Hidden Layer / Output Layer',
             color='blue')
    plt.plot(hidden_units, hf_list, label='Hidden Features (alone)',
             color='red')

    # Add labels and title
    plt.xlabel('Hidden Units (U)')
    plt.ylabel('Average Class-wise Activation Correlation Level')
    plt.title(f'Class-wise Activation Correlation of {args.model} trained on {args.dataset}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f"evaluation_images/Act-Corr-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
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
