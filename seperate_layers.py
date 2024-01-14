import torch
import torch.nn as nn


class Representation_Layer(nn.Module):
    def __init__(self, n):
        super(Representation_Layer, self).__init__()

        self.n = n

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n),
            nn.ReLU()
        )

        self.features.requires_grad_(False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out

class One_Layer_Classifier(nn.Module):
    def __init__(self, n):
        super(One_Layer_Classifier, self).__init__()

        self.n = n

        self.classifier = nn.Linear(n, 10)

    def forward(self, x):
        out = self.classifier(x)

        return out

class Two_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2):
        super(Two_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n2, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.classifier(out)

        return out


class Three_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2, n3):
        super(Three_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(n2, n3),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n3, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.classifier(out)

        return out

class Four_Layer_Classifier(nn.Module):
    def __init__(self, n1, n2, n3, n4):
        super(Four_Layer_Classifier, self).__init__()

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

        self.layer_1 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(n2, n3),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(n3, n4),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n4, 10)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.classifier(out)

        return out