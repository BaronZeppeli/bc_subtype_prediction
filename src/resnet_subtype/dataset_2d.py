import sys

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

import json

import data_processing.voxelizer as voxelizer


class AbstractDataset(ABC):
    VALID_IMAGE_FORMATS=["images", "masks", "whole_body"]

    @abstractmethod
    def __init__(self,
                 file_task: str,
                 image_format: str,
                 tvt: str,
                 triplets_name_file_labels: list,
                 ablation_channel_idxs: list,
                 include_channel_idxs: list,
                 transform):
        assert image_format in AbstractDataset.VALID_IMAGE_FORMATS,\
            (F"Invalid image format {image_format}, has to be one of "
             F"{AbstractDataset.VALID_IMAGE_FORMATS}.")

        self.transform = transform

        with open(file_task, 'r') as f:
            self.dict_config = json.load(f)

        self.task = self.dict_config["name"]
        dir_image_format = os.path.join(self.dict_config["dirPreprocessed"],
                                        self.task, image_format)
        if tvt is not None:
            assert tvt in os.listdir(dir_image_format), \
                (F"No subdirectory {tvt} in directory of the task"
                 F" {dir_image_format}!")
            assert tvt in self.dict_config.keys(), (F"No key {tvt} in "
                                                    F"config file!")
            tvt = [tvt]
        else:
            # Enables the use of images of training and validation at once if
            # None is specified as tvt.
            tvt = ["train", "val"]

        list_dict_train_val = []
        files_images = []
        for subset in tvt:
            dir_images = os.path.join(dir_image_format, subset)
            files_images += [os.path.join(dir_images, file) for file
                             in os.listdir(dir_images)]
            list_dict_train_val += self.dict_config[subset]
        files_images.sort()

        self.labels = [self.dict_config["labels"][key_labels]["name"] for
                       key_labels in
                       self.dict_config["labels"].keys()]
        self.classes_per_label = [sorted(self.dict_config["labels"][key_labels][
                                             "classes"].keys()) for
                                  key_labels in
                                  self.dict_config["labels"].keys()]

        # Sets the specified channels to constant 0, for ablation studies.
        self.ablation_channel_idxs = ablation_channel_idxs

        # Limits the channels to the specified indices.
        self.include_channel_idxs = include_channel_idxs

        if triplets_name_file_labels is None:
            self.triplets_name_file_labels = []
            for file_images in files_images:
                task = "_".join(os.path.basename(file_images).split("_")[:-1])
                matching_entries = [dict_img["labels"] for dict_img in
                                    list_dict_train_val if task in dict_img[
                                        "image"]]
                assert len(matching_entries) > 0, (
                    F"No matching entry for name "
                    F"{task} found in config file!")
                assert len(matching_entries) == 1, (
                    F"Ambiguous entry match for name"
                    F" {task} in config file!")
                labels = matching_entries[0]

                self.triplets_name_file_labels += [[task, file_images, labels]]
            # Sort the elements of the list in the order of the patient IDs.
            self.triplets_name_file_labels = sorted(
                self.triplets_name_file_labels,
                key=lambda x: os.path.basename(
                    x[0]))
        else:
            # Optionally load an already created triplet list, for use in k-fold
            # cross-validation.
            self.triplets_name_file_labels = triplets_name_file_labels

        return


class ImageDataset(AbstractDataset, Dataset):
    """
    OUTDATED, DO NOT USE.
    """

    def __init__(self,
                 dir_img: str,
                 transform,
                 ):
        AbstractDataset.__init__(self,
                                 dir_img=dir_img,
                                 transform=transform)

    def __len__(self):
        return len(self.list_file_class_name)

    def __getitem__(self, idx):
        # Load your 10-channel image (Assume it's stored as a .npy file for simplicity)
        file, clss, name = self.list_file_class_name[idx]
        img = np.load(file)  # Assuming images are stored as NumPy arrays

        # Convert to a torch tensor.
        img = torch.tensor(np.einsum("hwc->chw", img[0, ...]),
                           dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, clss, name


class VolumeDataset2D(AbstractDataset, Dataset):
    """
    Current 2D dataset class.
    """

    def __init__(self,
                 file_task: str,
                 image_format: str,
                 tvt: str,
                 transform,
                 axial_only: bool,
                 triplets_name_file_labels: list,
                 ablation_channel_idxs: list,
                 include_channel_idxs: list):
        AbstractDataset.__init__(self=self,
                                 file_task=file_task,
                                 image_format=image_format,
                                 tvt=tvt,
                                 triplets_name_file_labels=triplets_name_file_labels,
                                 ablation_channel_idxs=ablation_channel_idxs,
                                 include_channel_idxs=include_channel_idxs,
                                 transform=transform)

        # This option allows elegantly to only sample the images in the axial
        # direction.
        self.axial_only = axial_only

        # Create the list specifying which file an index belongs to.
        self.max_idx_per_file = self.create_max_idx_per_file()

        return

    def __len__(self):
        return self.max_idx_per_file[-1] + 1

    def __getitem__(self, idx):
        file_idx = np.min(np.where(self.max_idx_per_file >= idx)[0])
        name, file, labels = self.triplets_name_file_labels[file_idx]
        volume = voxelizer.VoxelArray(file).array
        if file_idx == 0:
            slice_idx = idx
        else:
            # Subtract 1 to account for the subtraction of 1 during computing
            # max_idx_per_file.
            slice_idx = idx - self.max_idx_per_file[file_idx - 1] - 1
        assert slice_idx < np.sum(volume.shape[:3]), ("Something went wrong in "
                                                      "determining the slice "
                                                      "index of the image.")
        # Axial slicing. In case of axial_only=True, only this will be applied.
        if slice_idx < volume.shape[0]:
            img = volume[slice_idx, ...]
        # Coronal slicing.
        elif slice_idx < np.sum(volume.shape[:2]):
            slice_idx -= volume.shape[0]
            img = volume[:, slice_idx, ...]
        # Sagittal slicing.
        else:
            slice_idx -= np.sum(volume.shape[:2])
            img = volume[:, :, slice_idx, ...]

        img = torch.tensor(np.copy(np.einsum("hwc->chw", img)),
                           dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.ablation_channel_idxs is not None:
            img[self.ablation_channel_idxs, ...] = 0

        # For training on only certain imaging modalities.
        if self.include_channel_idxs is not None:
            img = img[self.include_channel_idxs, ...]

        labels = torch.tensor(labels)

        return img, labels, name

    def create_max_idx_per_file(self, ):
        n_directions = 1 if self.axial_only else 3
        n_images_per_file = [np.sum(voxelizer.VoxelArray(
            source_voxel=file).array.shape[:n_directions]) for (_, file, __) in
                             self.triplets_name_file_labels]
        # Decrement by 1 to turn sums into the max index.
        max_idx_per_file = np.array([np.sum(n_images_per_file[:i + 1]) - 1
                                     for i in
                                     range(len(n_images_per_file))])

        return max_idx_per_file


class DataLoaderTest(DataLoader):
    """
    Workaround to AVOID having to transform all test images into a shared
    resolution.
    """

    def __init__(self,
                 dataset,
                 device=None):
        self.dataset = dataset
        self.idx = 0

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "cpu")
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.dataset):
            img, clss, name = self.dataset[self.idx]
            # We need to add a dimension to the tensor to turn
            # the output into batch with size n=1!
            img = img.unsqueeze(0)
            clss = torch.tensor([clss])
            name = [name]
            self.idx += 1
            return img.to(self.device), clss.to(self.device), name
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return len(self.dataset)
