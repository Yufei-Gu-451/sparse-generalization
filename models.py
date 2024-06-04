import torch
import torch.nn as nn
import torch.nn.functional as func

import os
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Transformer -------------------------------------------------------------------------------
class ConvExtractor(nn.Module):
    def __init__(self, d_model, in_channels=3, conv_size=1):
        super(ConvExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(512 * conv_size * conv_size, d_model)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, model_width, in_channels=3, img_size=32, num_classes=10, num_heads=8, num_layers=3):
        super(ConvEncoder, self).__init__()
        self.n_hidden_units = model_width
        self.n_layers = 4

        conv_size = img_size // 2 // 2 // 2 // 2
        self.conv_layer = ConvExtractor(d_model=model_width, in_channels=in_channels, conv_size=conv_size)

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_width, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.mlp_head = nn.Linear(model_width, num_classes)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.transformer_encoder(x)
        x = self.mlp_head(x)
        return x

    def forward_full(self, act_0):
        act_1 = self.conv_layer(act_0)
        act_2 = self.transformer_encoder(act_1)
        act_3 = self.mlp_head(act_2)

        return act_0, act_1, act_2, act_3


class ImageEncoder(nn.Module):
    def __init__(self, model_width, in_channels=3, img_size=32, num_classes=10, num_heads=8, num_layers=3):
        super(ImageEncoder, self).__init__()
        self.n_hidden_units = model_width
        self.n_layers = 4

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, model_width),
            nn.ReLU()
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_width, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier = nn.Linear(model_width, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x

    def forward_full(self, act_0):
        act_1 = self.features(act_0)
        act_2 = self.transformer_encoder(act_1)
        act_3 = self.classifier(act_2)

        return act_0, act_1, act_2, act_3


# VisionTransformer ----------------------------------------------------------------------------------
# Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class VisionTransformer(nn.Module):
    def __init__(self, model_width, patch_size, in_channels=3, img_size=32, num_classes=10,
                 num_heads=8, num_layers=3, dropout=0.0):
        super(VisionTransformer, self).__init__()
        self.n_hidden_units = model_width

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_width),
            nn.LayerNorm(model_width),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model_width))
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_width))
        self.dropout = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_width, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(model_width, num_classes)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


# ResNet18 ----------------------------------------------------------------------------------
# Reference: [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun Deep Residual Learning for Image Recognition.
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.expansion = 1

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = func.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = func.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(ResNet, self).__init__()
        self.n_hidden_units = k
        self.in_planes = k
        self.n_layers = 3

        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.layer1 = self._make_layer(block, k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * k, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * k, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * k * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = func.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = func.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def forward_full(self, act_0):
        act_0 = func.relu(self.bn1(self.conv1(act_0)))
        act_1 = self.layer1(act_0)
        act_2 = self.layer2(act_1)
        act_3 = self.layer3(act_2)
        act_4 = self.layer4(act_3)
        act_4 = func.avg_pool2d(act_4, 4)

        act_0 = act_0.view(act_0.size(0), -1)
        act_4 = act_4.view(act_4.size(0), -1)
        act_5 = self.linear(act_4)

        # return act_0, act_1, act_2, act_3, act_4, act_5
        return act_0, act_4, act_5


def ResNet18(k):
    return ResNet(BasicBlock, [2, 2, 2, 2], k=k)


# Standard CNN ----------------------------------------------------------------------------------
# Reference: [2] https://gitlab.com/harvard-machine-learning/double-descent
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class CNN(nn.Module):
    def __init__(self, model_width, in_channels, img_size, num_classes):
        self.n_hidden_units = model_width
        self.n_layers = 6

        super(CNN, self).__init__()

        # Layer 0
        self.conv1 = nn.Conv2d(in_channels, model_width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(model_width)
        self.relu1 = nn.ReLU()

        # Layer 1
        self.conv2 = nn.Conv2d(model_width, 2 * model_width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(2 * model_width)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2
        self.conv3 = nn.Conv2d(2 * model_width, 4 * model_width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(4 * model_width)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Layer 3
        self.conv4 = nn.Conv2d(4 * model_width, 8 * model_width, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(8 * model_width)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # Layer 4
        self.pool4 = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(8 * model_width, num_classes, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_full(self, act_0):
        act_1 = self.pool1(self.relu1(self.bn1(self.conv1(act_0))))
        act_2 = self.pool2(self.relu2(self.bn2(self.conv2(act_1))))
        act_3 = self.pool3(self.relu3(self.bn3(self.conv3(act_2))))
        act_4 = self.pool4(self.relu4(self.bn4(self.conv4(act_3))))

        act_0 = act_0.view(act_0.size(0), -1)
        act_4 = act_4.view(act_4.size(0), -1)
        act_5 = self.fc(act_4)

        return act_0, act_1, act_2, act_3, act_4, act_5


# Simple FC ----------------------------------------------------------------------------------
class CustomNormalization(nn.Module):
    def forward(self, x):
        return x / torch.max(torch.abs(x))


class NormSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_sigmoid = nn.Sequential(
            nn.Sigmoid(),
            CustomNormalization()
        )

    def forward(self, x):
        return self.norm_sigmoid(x.t())


class FCNN(nn.Module):
    def __init__(self, model_width, in_channels, img_size, num_classes):
        self.n_hidden_units = model_width
        self.n_layers = 3

        super(FCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, model_width),
            nn.ReLU()
        )

        self.classifier = nn.Linear(model_width, num_classes)

        # self.features_act_mat = torch.zeros((archi[0], archi[1]))
        # self.classifier_act_mat = torch.zeros((archi[1], archi[2]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_full(self, act_0):
        act_1 = self.features(act_0.to(torch.float32))
        act_0 = act_0.view(act_0.size(0), -1)
        act_1 = act_1.view(act_1.size(0), -1)
        act_2 = self.classifier(act_1)

        # self.features_act_mat += torch.sum(act_0.unsqueeze(2) * act_1.unsqueeze(1), dim=0).detach().cpu()
        # self.classifier_act_mat += torch.sum(act_1.unsqueeze(2) * act_2.unsqueeze(1), dim=0).detach().cpu()

        return act_0, act_1, act_2


# ------------------------------------------------------------------------------------------


def save_model(model, checkpoint_path, hidden_unit):
    state = {
        'net': model.state_dict()
    }

    torch.save(state, os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    print("Torch saved successfully!\n")


def load_model(checkpoint_path, model_name, dataset_name, hidden_unit):
    model = get_model(model_name, dataset_name, hidden_unit)
    checkpoint = torch.load(os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model


# Set the neural network model to be used
def get_model(model_name, dataset_name, hidden_unit):
    if dataset_name == 'MNIST':
        in_channels = 1
        img_size = 28
        num_classes = 100
    elif dataset_name == 'CIFAR-10':
        in_channels = 3
        img_size = 32
        num_classes = 10
    elif dataset_name == 'CIFAR-100':
        in_channels = 3
        img_size = 32
        num_classes = 100
    else:
        raise NotImplementedError

    if model_name == 'FCNN':
        model = FCNN(model_width=hidden_unit, in_channels=in_channels, img_size=img_size, num_classes=num_classes)
    elif model_name == 'CNN':
        model = CNN(model_width=hidden_unit, in_channels=in_channels, img_size=img_size, num_classes=num_classes)
    elif model_name == 'ResNet18':
        model = ResNet18(hidden_unit)
    elif model_name == 'ImageEncoder':
        model = ImageEncoder(model_width=hidden_unit,
                             in_channels=in_channels,
                             img_size=img_size,
                             num_classes=num_classes)
    elif model_name == 'ConvEncoder':
        model = ConvEncoder(model_width=hidden_unit,
                            in_channels=in_channels,
                            img_size=img_size,
                            num_classes=num_classes)
    elif model_name == 'ViT':
        model = VisionTransformer(model_width=hidden_unit,
                                  in_channels=in_channels,
                                  img_size=img_size,
                                  patch_size=img_size // 4)
    else:
        raise NotImplementedError

    print(f"\n{model_name} Model with %d hidden neurons successfully generated;" % hidden_unit)
    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


def get_full_activation(model, dataloader):
    act_list = [[] for _ in range(model.n_layers)]
    labels_list = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            act_list_temp = model.forward_full(inputs)

            for i, act in enumerate(act_list_temp):
                act_list[i].append(act.detach().cpu().numpy())

            for label in labels:
                labels_list.append(label.detach().cpu().numpy())

    for i in range(model.n_layers):
        act_list[i] = np.vstack(act_list[i])

    return act_list, labels_list


def group_by_predicts(features, predicts):
    grouped_features = {}
    labels = np.unique(predicts)

    for label in labels:
        mask = (predicts == label)
        grouped_features[label] = features[mask]

    return grouped_features
