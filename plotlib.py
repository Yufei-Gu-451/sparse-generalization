import matplotlib.pyplot as plt


class PlotLib:
    def __init__(self, model, dataset, hidden_units, test_units):
        self.model = model
        self.dataset = dataset
        self.hidden_units = hidden_units
        self.test_units = test_units

        self.x_ticks = self.get_x_ticks()
        self.index = [index for index, value in enumerate(self.hidden_units) if value in self.x_ticks]

        self.scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

        self.x_label = self.get_x_labels()
        self.x_ticks = self.get_x_ticks()

    def get_x_labels(self):
        if self.test_units:
            if self.model in ['FCNN']:
                x_label = 'Number of Hidden Neurons (N)'
            elif self.model in ['CNN', 'ResNet18']:
                x_label = 'Convolutional Layer Width (K)'
            else:
                raise NotImplementedError
        else:
            x_label = 'Number of Model Parameters (P) (*10^3)'

        return x_label

    def get_x_ticks(self):
        if self.model in ['FCNN']:
            if self.test_units:
                xticks = [1, 5, 15, 40, 100, 250, 500, 1000]
            else:
                xticks = [1, 4, 20, 100, 500, 2000, 5000]
        elif self.model in ['CNN', 'ResNet18']:
            if self.test_units:
                xticks = [1, 8, 16, 24, 32, 40, 48, 56, 64]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return xticks


# Plot function of Experiment Results
def plot_test_result(args, hidden_units, test_result):
    # Plot the Diagram
    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.set_title(
        f'Experiment Results on {args.dataset} (N=%d, p=%d%%)' % (args.sample_size, args.noise_ratio * 100))

    # Use globally defined PlotLib for labels, ticks and scaling function
    plotlib = PlotLib(model=args.model,
                      dataset=args.dataset,
                      hidden_units=hidden_units,
                      test_units=args.test_units)

    # Set scale and limitation according to dataset usage
    if args.model == 'FCNN':
        ax1.set_xscale('function', functions=plotlib.scale_function)
        ax3.set_xscale('function', functions=plotlib.scale_function)

        if args.noise_ratio <= 0.2:
            ylim = 2.0
        elif args.noise_ratio <= 0.6:
            ylim = 4.0
    elif args.model == 'CNN':
        if args.noise_ratio <= 0.2:
            ylim = 2.5
        elif args.noise_ratio <= 0.6:
            ylim = 4.0
    elif args.model == 'ResNet18':
        if args.noise_ratio <= 0.2:
            ylim = 3.0
        elif args.noise_ratio <= 0.6:
            ylim = 4.0
    else:
        raise NotImplementedError

    ax3.set_ylim([0.0, ylim])

    # Configure the x-axis value on Model Units (k) or Model Parameters (P)
    if args.test_units:
        x_axis_value = hidden_units
    else:
        x_axis_value = test_result.get_parameters()[1:]

    # Set x_labels and x_scales
    ax1.set_xlabel(plotlib.x_label)
    ax3.set_xlabel(plotlib.x_label)

    ax1.set_xticks(plotlib.x_ticks)
    ax3.set_xticks(plotlib.x_ticks)

    # Subplot 1
    ln1 = ax1.plot(x_axis_value, test_result.get_train_accuracy(), label='Train Accuracy', color='red')
    ln2 = ax1.plot(x_axis_value, test_result.get_test_accuracy(), label='Test Accuracy', color='blue')

    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0.0, 1.05])

    if args.knn and args.noise_ratio > 0:
        ax2 = ax1.twinx()

        ln3 = ax2.plot(x_axis_value, test_result.get_knn_c_accuracy(), label='k-NN Accuracy (clean label)', color='cyan')
        ln4 = ax2.plot(x_axis_value, test_result.get_knn_n_accuracy(), label='k-NN Accuracy (noisy label)', color='purple')
        ax2.set_ylabel('KNN Label Accuracy (100%)')
        ax2.set_ylim([0.0, 1.05])

        lns = ln1 + ln2 + ln3 + ln4
    else:
        lns = ln1 + ln2

    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.grid()

    # Subplot 2
    ln6 = ax3.plot(x_axis_value, test_result.get_train_losses(), label='Train Losses', color='red')
    ln7 = ax3.plot(x_axis_value, test_result.get_test_losses(), label='Test Losses', color='blue')
    ax3.set_ylabel('Cross Entropy Loss')

    if args.rade:
        ax4 = ax3.twinx()

        ln8 = ax4.plot(x_axis_value, test_result.get_rade_complexity(model=args.model),
                       label='Complexity estimate', color='green')
        ax4.set_ylabel('Rademacher Complexity (estimate)')

        lns = ln6 + ln7 + ln8
    else:
        lns = ln6 + ln7

    labs = [line.get_label() for line in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid()

    # Save Figure
    directory = f"{args.dataset}-{args.model}-Epochs=%d-p=%d" % \
                (args.epochs, args.noise_ratio * 100)

    if args.knn:
        directory = 'images/k-NN/k-NN-' + directory
    elif args.rade:
        directory = 'images/Complexity/Rade-' + directory
    else:
        directory = 'images/Results/' + directory

    if args.test_units:
        directory += '-U.png'
    else:
        directory += '-P.png'

    plt.savefig(directory)


def plot_scaling(args, hidden_unit_list, parameter_list):
    fig = plt.figure(figsize=(6, 6))

    plt.plot(hidden_unit_list, parameter_list)
    plt.yscale('log')

    plt.xlabel('Layer Width (k)')
    plt.ylabel('Number of Parameters (P)')
    plt.grid()

    if args.model == 'FCNN':
        plt.title('Fully Connected Neural Network')
    elif args.model == 'CNN':
        plt.title('Five-layer CNN')
    elif args.model == 'ResNet18':
        plt.title('ResNet18')
    else:
        raise NotImplementedError

    plt.savefig(f'images/Others/{args.model}-Scaling')