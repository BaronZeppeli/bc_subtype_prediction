import sys
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from dataset_2d import AbstractDataset

import data_processing.voxelizer as voxelizer


class VolumeDataset3D(AbstractDataset, Dataset):

    def __init__(self,
                 file_task: str,
                 image_format:str,
                 tvt: str,
                 transform,
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

        return

    def __len__(self):
        return len(self.triplets_name_file_labels)

    def __getitem__(self, idx):
        # Load your 10-channel volume.
        name, file, labels = self.triplets_name_file_labels[idx]
        volume = torch.tensor(np.copy(voxelizer.VoxelArray(
            source_voxel=file).array))

        # Convert to a torch tensor.
        volume = torch.einsum("hdwc->chdw", volume).to(torch.float32)

        if self.transform:
            volume = self.transform(volume)

        if self.ablation_channel_idxs is not None:
            volume[self.ablation_channel_idxs, ...] = 0

        if self.include_channel_idxs is not None:
            volume = volume[self.include_channel_idxs, ...]

        labels = torch.tensor(labels)

        return volume, labels, name
