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
                xticks = [1, 8, 20, 40, 64]
            else:
                xticks = [1, 8, 20, 40, 64]
        else:
            raise NotImplementedError

        return xticks
