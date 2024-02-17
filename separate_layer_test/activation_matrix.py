import sys
import os

from tqdm import tqdm

import test_knn_interpolation

sys.path.append('..')
import models

def get_activation_matrix(args, dataloader, directory, hidden_units):
    for hidden_unit in tqdm(hidden_units, desc="Processing"):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, dataset=args.dataset, hidden_unit=hidden_unit)
        model.eval()

        # Extract the hidden_features
        data, hidden_features, predicts, true_labels = knn_test.get_hidden_features('MNIST', model, dataloader)

        print(hidden_units, hidden_features.shape)

        # Extract the values of the features and convert them to a list
        n_features = hidden_features.shape[0] * hidden_features.shape[1]

    return