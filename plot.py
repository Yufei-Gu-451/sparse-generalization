class Plot:
    def __init__(self, model, dataset, hidden_units):
        self.model = model
        self.dataset = dataset
        self.hidden_units = hidden_units

        self.x_ticks = self.get_x_ticks()
        self.index = [index for index, value in enumerate(self.hidden_units) if value in self.x_ticks]

        self.scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    def get_x_ticks(self):
        if self.model in ['SimpleFC', 'SimpleFC_2']:
            xticks = [1, 5, 12, 40, 100, 250, 500, 1000]
        elif self.model in ['CNN', 'ResNet18']:
            xticks = [1, 8, 20, 40, 64]
        else:
            raise NotImplementedError

        return xticks
