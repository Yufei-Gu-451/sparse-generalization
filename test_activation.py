import matplotlib.pyplot as plt
from tqdm import tqdm

import models
from plotlib import PlotLib

import numpy as np
import os


def get_cosine_similarity(matrix1, matrix2):
    # Flatten the matrices to 1D arrays
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()

    # Calculate dot product
    dot_product = np.dot(flattened_matrix1, flattened_matrix2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(flattened_matrix1)
    magnitude2 = np.linalg.norm(flattened_matrix2)

    if magnitude1 != 0 and magnitude2 != 0:
        # Calculate cosine similarity
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return cosine_similarity
    else:
        return 0


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


def compute_correlation(matrices_list, similarity_measure):
    # Calculate cosine similarity between each pair of matrices
    similarities = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
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


def get_activation_correlation(args, dataloader, directory, hidden_units):
    # Load a list of dataloaders of all classes
    correlation_dict = {'Input-Hidden': [], 'Hidden-Output': []}
    heatmap_1_dict, heatmap_2_dict = {}, {}

    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, model_name=args.model, hidden_unit=hidden_unit)
        model.eval()

        # Get Activation List and Predicitions
        act_list, _ = models.get_full_activation(model, dataloader)
        predicts = np.argmax(act_list[-1], axis=1)
        input_feature_size = act_list[0].shape[1]

        # Get Grouped Features based on the Class Predictions
        group_inputs = models.group_by_predicts(act_list[0], predicts)
        group_features = models.group_by_predicts(act_list[-2], predicts)
        group_outputs = models.group_by_predicts(act_list[-1], predicts)
        del act_list

        cam1_list, cam2_list, hf_list = [], [], []

        for i in range(10):
            if i in np.unique(predicts):
                # Activation Matrices between input_layer and hidden_layer: For each test item (10000 * d * n)
                cam1 = np.mean(np.multiply(group_inputs[i][:, :, np.newaxis], group_features[i][:, np.newaxis, :]), axis=0)
                cam2 = np.mean(np.multiply(group_features[i][:, :, np.newaxis], group_outputs[i][:, np.newaxis, :]), axis=0)
            else:
                cam1 = np.zeros((input_feature_size, hidden_unit))
                cam2 = np.zeros((hidden_unit, 10))

            # Append all information to the class-wise list
            cam1_list.append(cam1)
            cam2_list.append(cam2)

        del group_inputs, group_features, group_outputs

        # Compute similarities between class activation matrices
        corr_1, mean_1 = compute_correlation(cam1_list, similarity_measure='cosine')
        corr_2, mean_2 = compute_correlation(cam2_list, similarity_measure='cosine')

        # Plot heatmap on certain hidden_unit threshold
        if args.model in ['FCNN']:
            if hidden_unit in [10, 20, 40, 200]:
                heatmap_1_dict[hidden_unit] = corr_1
                heatmap_2_dict[hidden_unit] = corr_2

        correlation_dict['Input-Hidden'].append(mean_1)
        correlation_dict['Hidden-Output'].append(mean_2)

    if args.model in ['FCNN']:
        plot_heatmap(args, heatmap_1_dict, 1, vmin=0)
        plot_heatmap(args, heatmap_2_dict, 2, vmin=-0.5)

    return correlation_dict


def plot_cam_correlation(args, correlation_dict, hidden_units):
    # Get the activation similarities list mean over runs
    cam_1_list = np.mean(correlation_dict['Input-Hidden'], axis=0)
    cam_2_list = np.mean(correlation_dict['Hidden-Output'], axis=0)
    # hf_list = np.mean(correlation_dict['Hidden'], axis=0)

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units,
                      noise_ratio=args.noise_ratio)

    if args.model in ['FCNN']:
        ax.set_xscale('function', functions=plotlib.scale_function)
    ax.set_xticks(plotlib.x_ticks)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Plot the line chart
    plt.plot(hidden_units, cam_1_list, label='Input Layer / Hidden Layer', color='orange')
    plt.plot(hidden_units, cam_2_list, label='Hidden Layer / Output Layer', color='purple')

    # Add labels and title
    plt.xlabel('Hidden Units (U)', fontsize=14)
    plt.ylabel('Mean CAM Similarities', fontsize=14)
    plt.title(f'{args.model} on {args.dataset} (N=%d, p=%d%%)' %
              (args.sample_size, args.noise_ratio * 100), fontsize=18)

    # Add a legend
    plt.legend(fontsize=14)
    plt.grid()

    # Show the plot
    plt.savefig(f"images/Activation/Act-Corr-{args.dataset}-{args.model}-Epochs=%d-p=%d.png"
                % (args.epochs, args.noise_ratio * 100))


def plot_heatmap(args, heatmap_dict, layer_n, vmin):
    # Plot heatmap
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, (hidden_unit, corr) in enumerate(heatmap_dict.items()):
        im = axs[i].imshow(corr, cmap='hot', vmin=vmin, vmax=1)
        axs[i].set_xticks(range(10))
        axs[i].set_yticks(range(10))
        axs[i].set_xlabel('Class Label')
        axs[i].set_ylabel('Class Label')
        axs[i].text(0.5, -0.3, f"({i+1}) k={hidden_unit}", ha='center', fontsize=16, transform=axs[i].transAxes)

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    cbar.set_label('Cosine Similarity')

    plt.subplots_adjust(left=0.05, right=0.75, bottom=0.1, top=1)
    plt.savefig(f'images/Activation/Act-Heatmap-{args.dataset}-{args.model}-Epochs=%d-p=%d-%d.png'
                % (args.epochs, args.noise_ratio * 100, layer_n))
