from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchvision import models
import warnings

import resnet_3d


class AbstractFactory(ABC):
    @abstractmethod
    def create_model(self):
        pass


class ResnetFactory(AbstractFactory):
    """
    This factory serves to create new instances of ResNet for the
    cross-validation.
    """

    def __init__(self,
                 n_channels_input: int,
                 n_classes_per_label: list,
                 pretrained: bool,
                 version: int,
                 dims: int):

        self.n_classes_per_label = n_classes_per_label
        self.n_channels_input = n_channels_input
        self.pretrained = pretrained
        self.version = version

        assert (dims == 2) | (
                dims == 3), "Value of dims must be either 2 or 3."
        self.dims = dims

    def create_model(self):
        if self.dims == 2:
            model = ResNet2D(n_channels_input=self.n_channels_input,
                             n_classes_per_label=self.n_classes_per_label,
                             pretrained=self.pretrained,
                             version=self.version)
        else:
            # Note: Pretraining on ImageNet is not supported for 3D ResNets!
            model = resnet_3d.ResNet3D.create_resnet3d(version=self.version,
                                                       n_channels_input=self.n_channels_input,
                                                       n_classes_per_label=self.n_classes_per_label)

        return model


class ResNet2D(nn.Module, resnet_3d.AbstractResNet):
    def __init__(self,
                 n_channels_input: int,
                 n_classes_per_label: list,
                 pretrained: bool,
                 version: int):

        super(ResNet2D, self).__init__()

        if version == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 \
                if pretrained else None
            self.resnet = models.resnet18(weights=weights)
        elif version == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 \
                if pretrained else None
            self.resnet = models.resnet34(weights=weights)
        elif version == 50:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 \
                if pretrained else None
            self.resnet = models.resnet50(weights=weights)
        elif version == 101:
            weights = models.ResNet101_Weights.IMAGENET1K_V1 \
                if pretrained else None
            self.resnet = models.resnet101(weights=weights)
        elif version == 152:
            weights = models.ResNet152_Weights.IMAGENET1K_V1 \
                if pretrained else None
            self.resnet = models.resnet152(weights=weights)
        else:
            raise ValueError("No valid version of ResNet provided!")

        # The init call of the AbstractResNet superclass can only happen
        # after attribute resnet is set!!
        resnet_3d.AbstractResNet.__init__(self,
                                          dims=2,
                                          version=version)

        self.n_channels_input = n_channels_input
        self.n_classes_per_label = n_classes_per_label
        self.n_labels = len(n_classes_per_label)

        new_conv1 = nn.Conv2d(in_channels=n_channels_input,
                              # Change to n channels
                              out_channels=64,  # Keep the same as original
                              kernel_size=7,  # Keep the same as original
                              stride=2,  # Keep the same as original
                              padding=3,  # Keep the same as original
                              bias=False)

        # Copy weights from the original 3-channel conv1 and replicate for
        # n_channels_input channels.
        if pretrained:
            with torch.no_grad():
                # Copy first 3 channels (RGB), or as many channels as can be
                # applied.
                pretrained_dims = min(n_channels_input, 3)
                new_conv1.weight[:, :pretrained_dims, :,
                :] = self.resnet.conv1.weight[:, :pretrained_dims, :, :]

        # Replace the original conv1 with the new one.
        self.resnet.conv1 = new_conv1
        # Replace the original fully connected layer with the new one.
        self.label_heads = None
        self.set_last_layer(n_classes_per_label=n_classes_per_label)

        return

    def set_device(self, device: str):
        # During initialization, resnet is not yet set.
        self.resnet = self.resnet.to(device)
        self.device = device
        return

    def forward(self, x):
        features = self.resnet.to(self.device)(x)
        # Collect predictions from each of the label-specific heads
        outputs = [head.to(self.device)(features) for head in self.label_heads]

        # Concatenate the outputs along the batch dimension
        # Outputs shape: (batch_size, num_labels, num_classes)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def set_last_layer(self,
                       n_classes_per_label: list):

        # Special handling for probabilistic regression: Make sure the output
        # of the last layer gets transformed into range [0, 1].
        if n_classes_per_label == [1]:
            self.label_heads = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(self.resnet.fc.in_features, 1),
                    nn.Sigmoid()
                )]  # bounds [0,1]
            )
        else:
            # Create separate fully connected layers for each label.
            self.label_heads = nn.ModuleList(
                [nn.Linear(self.resnet.fc.in_features, n_classes)
                 for n_classes in n_classes_per_label])

        # Remove the final fully connected layer of ResNet
        self.resnet.fc = nn.Identity()  # Keep only the feature extractor

        self.n_classes_per_label = n_classes_per_label
        self.n_labels = len(n_classes_per_label)

        return
