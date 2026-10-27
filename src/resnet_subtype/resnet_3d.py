import os

from abc import ABC, abstractmethod

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# Abstract class moved into this module to avoid circular imports.
class AbstractResNet(ABC):

    def __init__(self,
                 dims: int,
                 version: int):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.set_device(device=self.device)
        self.dims = dims
        self.version = version

        self.n_labels = None

        self.label_heads = None

        return

    def get_info(self):
        return F"ResNet{self.version}, {self.dims}D"

    def get_architecture(self):
        return F"ResNet{self.version}"

    def get_dims(self):
        return self.dims

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def set_last_layer(self,
                       n_classes_per_label: list):
        pass

    def get_n_labels(self):
        return self.n_labels

    def get_device(self):
        return self.device

    @abstractmethod
    def set_device(self, device: str):
        pass

    def load_weights(self,
                     file_weights: str,
                     load_heads: bool = True,
                     eval: bool = True,
                     solve_discrepancies: bool = True):
        """
        Loads weights from a PTH weights file while also optionally solving
        discrepancies in the naming of layers between the file and the model
        class.
        """
        # Load the weights of an already trained model.
        weights = torch.load(file_weights, weights_only=True)

        # In case the heads are not supposed to be loaded, i.e. for
        # finetuning, simply re-load the default weights the model has
        # assumed during initialization.
        if not load_heads:
            warnings.warn("The label heads are not being loaded! This only "
                          "makes sense in the case of pretraining!")

            weights_new = {}

            # Remove all label heads layers from the PTH file.
            for layer_loaded, val_loaded in weights.items():
                if "label_heads." not in layer_loaded:
                    weights_new[layer_loaded] = val_loaded

            # And insert all label heads layers of the model class.
            for layer_self, val_self in self.state_dict().items():
                if "label_heads." in layer_self:
                    weights_new[layer_self] = val_self

            weights = weights_new

        layers_self = set(self.state_dict().keys())
        layers_loaded = set(weights.keys())

        assert len(layers_self) == len(layers_loaded), \
            ("Different number of layers for the model class and the weights "
             "file. Match impossible.")

        discrepancy_layers = layers_loaded.symmetric_difference(layers_self)
        if len(discrepancy_layers) > 0:
            layers_loaded_disc = layers_loaded - layers_self
            layers_self_disc = layers_self - layers_loaded
            warn_text = (F"\nDiscrepancy between the layer names of the model "
                         F"class and the weights file detected!\n"
                         F"Following layers in the weights file have no match: "
                         F"{layers_loaded_disc}\n"
                         F"Following layers in the model class have no match: "
                         F"{layers_self_disc}")

            assert len(layers_self_disc) == len(layers_loaded_disc), \
                ("The number of mismatches between the weights file "
                 "and the model class is not identical. Match impossible.")

            if solve_discrepancies:
                warnings.warn(warn_text
                              + "\nTrying to solve the discrepancies.\n")

                dict_match = {}
                for layer_loaded in layers_loaded_disc:
                    max_match_len = 0
                    for layer_self in layers_self_disc:
                        prefix_match_len = len(os.path.commonprefix(
                            [layer_loaded, layer_self]))
                        suffix_match_len = len(os.path.commonprefix(
                            [layer_loaded[::-1], layer_self[::-1]]))
                        match_len = prefix_match_len + suffix_match_len
                        if match_len > max_match_len:
                            max_match_len = match_len
                            dict_match[layer_loaded] = layer_self

                assert (len(set(dict_match.values()))
                        == len(dict_match.values())), \
                    ("Some layers from the weights file were matched to the "
                     "same layer of the model class! Match impossible.")

                match_str = ""
                for layer_loaded, layer_self in dict_match.items():
                    match_str += F"\n{layer_loaded} --> {layer_self}"
                warnings.warn(
                    "\nMatches between the weights file and the model class:"
                    + match_str + "\n")

                # Replacing the weights file layer names with the model class
                # layer names.
                weights_new = {}
                for layer_old, val_loaded in weights.items():
                    if layer_old in dict_match.keys():
                        layer_new = dict_match[layer_old]
                    else:
                        layer_new = layer_old
                    weights_new[layer_new] = val_loaded
                weights = weights_new
            else:
                raise ValueError(warn_text + "\nQuitting.")

        self.load_state_dict(weights, strict=True)

        if eval:
            warnings.warn("IMPORTANT: The model from weight files will be "
                          "run in eval mode! If this is unwanted, change the "
                          "code!")
            self.eval()

        return


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet3D(nn.Module, AbstractResNet):
    @staticmethod
    def create_resnet3d(version: int,
                        n_channels_input: int,
                        n_classes_per_label: list):
        if version == 18:
            resnet = ResNet3D(block=BasicBlock3D,
                              layers=[2, 2, 2, 2],
                              n_channels_input=n_channels_input,
                              n_classes_per_label=n_classes_per_label,
                              version=version)
        elif version == 34:
            resnet = ResNet3D(block=BasicBlock3D,
                              layers=[3, 4, 6, 3],
                              n_channels_input=n_channels_input,
                              n_classes_per_label=n_classes_per_label,
                              version=version)
        elif version == 50:
            resnet = ResNet3D(block=Bottleneck3D,
                              layers=[3, 4, 6, 3],
                              n_channels_input=n_channels_input,
                              n_classes_per_label=n_classes_per_label,
                              version=version)
        elif version == 101:
            resnet = ResNet3D(block=Bottleneck3D,
                              layers=[3, 4, 23, 3],
                              n_channels_input=n_channels_input,
                              n_classes_per_label=n_classes_per_label,
                              version=version)
        elif version == 152:
            resnet = ResNet3D(block=Bottleneck3D,
                              layers=[3, 8, 36, 3],
                              n_channels_input=n_channels_input,
                              n_classes_per_label=n_classes_per_label,
                              version=version)
        else:
            raise ValueError("Specified version of ResNet3D is not supported.")

        return resnet

    def __init__(self,
                 block,
                 layers,
                 n_channels_input: int,
                 n_classes_per_label: list,
                 version: int):

        super(ResNet3D, self).__init__()
        AbstractResNet.__init__(self,
                                dims=3,
                                version=version)

        self.n_classes_per_label = n_classes_per_label
        self.n_labels = len(self.n_classes_per_label)

        self.expansion = block.expansion

        self.in_planes = 64

        self.conv1 = nn.Conv3d(n_channels_input, 64, kernel_size=7, stride=2,
                               padding=3,
                               bias=False).to(self.device)
        self.bn1 = nn.BatchNorm3d(64).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1).to(
            self.device)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)).to(self.device)

        # Set the fully connected layer.
        self.label_heads = None
        self.set_last_layer(n_classes_per_label=self.n_classes_per_label)

        return

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        # Collect predictions from each of the label-specific heads
        outputs = [head(features) for head in self.label_heads]

        # Concatenate the outputs along the batch dimension
        # Outputs shape: (batch_size, num_labels, num_classes)
        return torch.stack(outputs, dim=1)

        return x

    def set_last_layer(self,
                       n_classes_per_label: list):

        # Create separate fully connected layers for each label
        self.label_heads = nn.ModuleList(
            [nn.Linear(512 * self.expansion, n_classes).to(self.device)
             for n_classes in n_classes_per_label])

        self.n_classes_per_label = n_classes_per_label
        self.n_labels = len(self.n_classes_per_label)

        return

    def set_device(self, device: str):
        self.to(device)
        self.device = device

        return


"""
    def load_weights(self,
                     file_weights: str,
                     load_heads: bool = True,
                     eval: bool = True):
        # Load the weights of an already trained model.
        weights = torch.load(file_weights, weights_only=True)

        # For some reason, the names of the layers are given the prefix
        # resnet when saving. Therefore, remove the prefix when loading the
        # weights-dictionary.
        new_weights = {}
        for key in weights.keys():
            new_key = key.replace('resnet.', '')  # Remove the prefix
            new_weights[new_key] = weights[key]
        self.load_state_dict(new_weights, strict=True)
        
        if eval:
            warnings.warn("IMPORTANT: The model from weight files will be "
                          "run in eval mode! If this is unwanted, change the "
                          "code!")
            self.eval()

        return
"""
