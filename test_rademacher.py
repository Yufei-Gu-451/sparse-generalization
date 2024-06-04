import numpy as np
import os

from tqdm import tqdm
import models


def get_complexity(args, test_dataloader, hidden_units, directory):
    n_complexity_list = []
    sub_sampling_size = 20

    # Compute the Rademacher Complexity
    for n in tqdm(hidden_units):
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = models.load_model(checkpoint_path, model_name=args.model, dataset_name=args.dataset, hidden_unit=n)
        model.eval()

        # Get Grouped Features based on the Class Predictions
        test_act_list, _ = models.get_full_activation(model, test_dataloader)
        predicts = np.argmax(test_act_list[-1], axis=1)

        group_features = models.group_by_predicts(test_act_list[-2], predicts)
        del test_act_list, predicts

        complexity_list = []

        for _, features in group_features.items():
            '''
            for j in range(1000):
                rademacher_variables = np.random.choice([-1, 1], size=features.shape[0])
                rademacher_complexity = np.max(np.abs(np.dot(features, rademacher_variables.T)))
                complexity_list.append(rademacher_complexity / np.sqrt(features.shape[0]))
            '''

            n_splits = features.shape[0] // sub_sampling_size

            for i in range(n_splits):
                np.random.seed(i)
                sub_features = features[i*sub_sampling_size: (i+1)*sub_sampling_size]

                method = 3
                temp_list = []
                for j in range(50):
                    if method == 1:
                        rademacher_variables = np.random.choice([-1, 1], size=sub_sampling_size)
                        rademacher_complexity = np.max(np.abs(np.dot(rademacher_variables, sub_features)))
                        complexity_list.append(rademacher_complexity / sub_sampling_size)
                    elif method == 2:
                        rademacher_variables = np.random.choice([-1, 1], size=(sub_sampling_size, n))
                        rademacher_complexity = np.mean(np.abs(np.dot(rademacher_variables, sub_features.T)))
                        temp_list.append(rademacher_complexity)
                    elif method == 3:
                        rademacher_variables = np.random.choice([-1, 1], size=sub_sampling_size)
                        rademacher_complexity = np.mean(rademacher_variables @ sub_features)
                        temp_list.append(rademacher_complexity)

                complexity_list.append(np.max(temp_list))

        # Calculate the empirical Rademacher complexity
        # print(n, np.mean(complexity_list))
        n_complexity_list.append(np.mean(complexity_list))

    print(n_complexity_list)
    return n_complexity_list
