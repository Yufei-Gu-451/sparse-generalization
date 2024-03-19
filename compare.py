import matplotlib.pyplot as plt
import os

from main import TestResult
from train import read_dict
from plotlib import PlotLib
from test_knn_interpolation import knn_prediction_test


DATASET = 'MNIST'
MODEL = 'FCNN'
SAMPLE_SIZE = 4000
EPOCH = 4000
NOISE = 0.0
KNN = True
TEST_UNITS = True

if __name__ == '__main__':
    test_number_1 = [1, 2, 3, 4, 5]
    test_number_2 = [997, 998, 999]

    # Set the hidden_units
    if MODEL in ['FCNN']:
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        12, 14, 16, 18, 20, 22, 25, 30, 35, 40,
                        45, 50, 55, 60, 70, 80, 90, 100, 120, 150,
                        200, 400, 600, 800, 1000]
    elif MODEL in ['CNN', 'ResNet18']:
        hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14,
                        16, 18, 20, 24, 28, 32, 36, 40, 44, 48,
                        52, 56, 60, 64]
    else:
        raise NotImplementedError

    print('Hidden_units = {}'.format(hidden_units))

    # Initialize the variables
    test_result_1, test_result_2 = TestResult(), TestResult()

    for test_number in test_number_1:
        directory = f"assets/{DATASET}-{MODEL}/N=%d-Epoch=%d-p=%d-sgd-%d" % \
                    (SAMPLE_SIZE, EPOCH, NOISE * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")
        dictionary_path = os.path.join(directory, 'dictionary')

        parameters, train_accuracy, test_accuracy, train_losses, test_losses = read_dict(dictionary_path,
                                                                                         EPOCH, hidden_units)

        test_result_1.parameters_list.append(parameters)
        test_result_1.train_accuracy_list.append(train_accuracy)
        test_result_1.test_accuracy_list.append(test_accuracy)
        test_result_1.train_losses_list.append(train_losses)
        test_result_1.test_losses_list.append(test_losses)

        # Run KNN Test
        if NOISE > 0:
            knn_accuracy = knn_prediction_test(directory, hidden_units, DATASET, NOISE, 128, 1, k=5)
            test_result_1.knn_accuracy_list.append(knn_accuracy)

    for test_number in test_number_2:
        directory = f"assets/{DATASET}-{MODEL}/N=%d-Epoch=%d-p=%d-sgd-%d" % \
                    (SAMPLE_SIZE, EPOCH, NOISE * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")
        dictionary_path = os.path.join(directory, 'dictionary')

        parameters, train_accuracy, test_accuracy, train_losses, test_losses = read_dict(dictionary_path,
                                                                                         EPOCH, hidden_units)

        test_result_2.parameters_list.append(parameters)
        test_result_2.train_accuracy_list.append(train_accuracy)
        test_result_2.test_accuracy_list.append(test_accuracy)
        test_result_2.train_losses_list.append(train_losses)
        test_result_2.test_losses_list.append(test_losses)

        # Run KNN Test
        if NOISE > 0:
            knn_accuracy = knn_prediction_test(directory, hidden_units, DATASET, NOISE, 128, 1, k=5)
            test_result_2.knn_accuracy_list.append(knn_accuracy)

    # ------------------------------------------------------------------------------------------------------------------

    # Plot the Diagram
    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.set_title(
        f'Experiment Results on {DATASET} (N=%d, p=%d%%)' % (SAMPLE_SIZE, NOISE * 100))

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=MODEL,
                      dataset=DATASET,
                      hidden_units=hidden_units,
                      test_units=TEST_UNITS)

    if DATASET == 'MNIST':
        ax1.set_xscale('function', functions=plotlib.scale_function)
        ax3.set_xscale('function', functions=plotlib.scale_function)

        ax3.set_ylim([0.0, 2.0])
    elif DATASET == 'CIFAR-10':
        ax3.set_ylim([0.0, 3.5])
    else:
        raise NotImplementedError

    # Subplot 1
    if TEST_UNITS:
        ln11 = ax1.plot(hidden_units,
                        test_result_1.get_train_accuracy(),
                        label='Train Accuracy (wo REG)', color='red')
        ln12 = ax1.plot(hidden_units,
                        test_result_1.get_test_accuracy(),
                        label='Test Accuracy (wo REG)', color='blue')

        ln21 = ax1.plot(hidden_units,
                        test_result_2.get_train_accuracy(),
                        label='Train Accuracy (w REG)', color='orange')
        ln22 = ax1.plot(hidden_units,
                        test_result_2.get_test_accuracy(),
                        label='Test Accuracy (w REG)', color='purple')
    else:
        ln11 = ax1.plot(test_result_1.get_parameters()[1:],
                        test_result_1.get_train_accuracy()[1:],
                        label='Train Accuracy (wo REG)', color='red')
        ln12 = ax1.plot(test_result_1.get_parameters()[1:],
                        test_result_1.get_test_accuracy()[1:],
                        label='Test Accuracy (wo REG)', color='blue')

        ln21 = ax1.plot(hidden_units,
                        test_result_2.get_train_accuracy(),
                        label='Train Accuracy (w REG)', color='orange')
        ln22 = ax1.plot(hidden_units,
                        test_result_2.get_test_accuracy(),
                        label='Test Accuracy (w REG)', color='purple')

    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0.0, 1.05])

    if KNN and NOISE > 0:
        ax2 = ax1.twinx()

        if TEST_UNITS:
            ln13 = ax2.plot(hidden_units,
                            test_result_1.get_knn_accuracy(),
                            label='Prediction Accuracy (wo REG)', color='cyan')

            ln23 = ax2.plot(hidden_units,
                            test_result_2.get_knn_accuracy(),
                            label='Prediction Accuracy (w REG)', color='cyan')
        else:
            ln13 = ax2.plot(test_result_1.get_parameters(),
                            test_result_1.get_knn_accuracy(),
                            label='Prediction Accuracy (wo REG)', color='cyan')

            ln23 = ax2.plot(test_result_2.get_parameters(),
                            test_result_2.get_knn_accuracy(),
                            label='Prediction Accuracy (w REG)', color='cyan')

        ax2.set_ylabel('KNN Label Accuracy (100%)')
        ax2.set_ylim([0.0, 1.05])

        lns = ln11 + ln12 + ln13 + ln21 + ln22 + ln23
    else:
        lns = ln11 + ln12 + ln21 + ln22

    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.grid()

    # Subplot 2
    if TEST_UNITS:
        ln16 = ax3.plot(hidden_units,
                        test_result_1.get_train_losses(),
                        label='Train Losses (wo REG)', color='red')
        ln17 = ax3.plot(hidden_units,
                        test_result_1.get_test_losses(),
                        label='Test Losses (wo REG)', color='blue')

        ln26 = ax3.plot(hidden_units,
                        test_result_2.get_train_losses(),
                        label='Train Losses (w REG)', color='orange')
        ln27 = ax3.plot(hidden_units,
                        test_result_2.get_test_losses(),
                        label='Test Losses (w REG)', color='purple')
    else:
        ln16 = ax3.plot(test_result_1.get_parameters()[1:],
                        test_result_1.get_train_losses()[1:],
                        label='Train Losses (wo REG)', color='red')
        ln17 = ax3.plot(test_result_1.get_parameters()[1:],
                        test_result_1.get_test_losses()[1:],
                        label='Test Losses (wo REG)', color='blue')

        ln26 = ax3.plot(test_result_2.get_parameters()[1:],
                        test_result_2.get_train_losses(),
                        label='Train Losses (w REG)', color='orange')
        ln27 = ax3.plot(test_result_2.get_parameters()[1:],
                        test_result_2.get_test_losses(),
                        label='Test Losses (w REG)', color='purple')

    ax3.set_ylabel('Cross Entropy Loss')

    lns = ln16 + ln17 + ln26 + ln27
    labs = [line.get_label() for line in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid()

    # Set x_labels and x_scales
    ax1.set_xlabel(plotlib.x_label)
    ax3.set_xlabel(plotlib.x_label)

    ax1.set_xticks(plotlib.x_ticks)
    ax3.set_xticks(plotlib.x_ticks)

    # Save Figure
    if TEST_UNITS:
        directory = f"images_1/{DATASET}-{MODEL}-Epochs=%d-p=%d-U-CP.png" % (EPOCH, NOISE * 100)
    else:
        directory = f"images_1/{DATASET}-{MODEL}-Epochs=%d-p=%d-P-CP.png" % (EPOCH, NOISE * 100)

    plt.savefig(directory)

    print('Program Ends!!!')
