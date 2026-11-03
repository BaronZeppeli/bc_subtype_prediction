import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import json

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, \
    AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

from resnet_subtype.task import Task
import resnet_subtype.ensemble as ensemble
import resnet_subtype.dataset_2d as dataset_2d
import resnet_subtype.dataset_3d as dataset_3d
import data_processing.voxelizer as voxelizer


class GradCAMBase:
    """
    The base class handling all Grad-CAM calls.
    """

    def __init__(self,
                 model,
                 target_layers,
                 class_idx: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.target_layers = target_layers
        self.class_idx = class_idx

        return

    def forward(self,
                input_tensor: torch.Tensor):
        """
        Note: input_tensor can be a batch tensor with several images!
        """
        # Construct the CAM object once, and then re-use it on many images.
        cam = GradCAM(model=self.model,
                      target_layers=self.target_layers)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers,
        # use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own
        # custom targets that are, for example, combinations of categories, or
        # specific outputs in a non-standard model.

        targets = [ClassifierOutputTarget(self.class_idx)]

        # Needs to be turned into a 5D batch tensor with batch_size=1.
        input_tensor_model = input_tensor[None, ...].to(self.device)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        cam_grayscale = cam(input_tensor=input_tensor_model, targets=targets)
        assert cam_grayscale.shape[
                   0] == 1, "Somehow, more than 1 image was inferred!"
        # In this example grayscale_cam has only one image in the batch:
        cam_grayscale = cam_grayscale[0, ...]

        img_grayscale = input_tensor.cpu().detach().numpy()
        # Returning the input image to its normal dimension order of putting
        # the image channels at the end.
        img_grayscale = np.einsum("chdw->hdwc", img_grayscale)

        return img_grayscale, cam_grayscale

    @staticmethod
    def display(img_grayscale: np.ndarray,
                cam_grayscale: np.ndarray,
                slide_idx: int = None,
                channel_idx: int = -1):
        """
        Displaying the Grad-CAM results for a single slide and a single channel.
        """
        assert img_grayscale.ndim == 4, "The input image needs to be 4D!"
        if img_grayscale.shape[-1] > 1:
            warnings.warn(F"More than one image channel detected! Only "
                          F"displaying the channel with index {channel_idx}.")
            img_grayscale = cam_grayscale[..., channel_idx]

        # Handling for 3D models.
        # 3D image tensor shape: chdw.
        if cam_grayscale.ndim == 3:
            assert slide_idx is not None, ("In case of a 3D model, a slide "
                                           "index needs to be provided!")
            img_grayscale = img_grayscale[slide_idx, ...]
            cam_grayscale = cam_grayscale[slide_idx, ...]

        fig, axes = plt.subplots(1)
        axes.imshow(img_grayscale, cmap="Greys_r")
        axes.imshow(cam_grayscale, cmap="jet", alpha=0.5)
        plt.show()

        return

    @staticmethod
    def export(arr,
               dir_export: str,
               sample_id: str,
               suffix: str):
        """
        Simple function to import any array within a subdirectory structure.
        """
        file_export = os.path.join(dir_export, sample_id,
                                   sample_id + "_" + suffix + ".nii")
        # For compatibility.
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().detach().numpy()
        assert 3 <= arr.ndim <= 4, ("The input image needs to be 3D or "
                                    "4D!")
        # Removing an unnecessary dimension in case there is only 1 image
        # channel.
        if (arr.ndim == 4) & (arr.shape[-1] == 1):
            arr = arr[..., 0]
        # Always export 3D NIFTIs if possible.
        enforce_3D = arr.ndim == 3

        va = voxelizer.VoxelArray(source_voxel=arr)
        va.to_nifti(file_export,
                    enforce_3D=enforce_3D)

        return


class GradCAMResNet(GradCAMBase):
    """
    Performs Grad-CAM on a ResNet model.
    """

    def __init__(self,
                 model_id: str,
                 file_task: str,
                 channel_couple: list,
                 transform_2D,
                 transform_3D,
                 file_weights: str = None,
                 file_hyperparams: str = None
                 ):
        self.file_task = file_task
        self.channel_couple = channel_couple
        self.transform_2D = transform_2D
        self.transform_3D = transform_3D

        self.task = Task(file_task=file_task)
        model_obj, dict_hyperparams = self._get_model_and_dict_hyperparams(
            model_id=model_id,
            channel_couple=channel_couple,
            file_weights=file_weights,
            file_hyperparams=file_hyperparams)
        self.model_id = model_id
        class_idx = self._get_class_idx()

        # Shape: ResnetXX, YD.
        self.dims = model_obj.dims
        assert self.dims in [2, 3], F"Invalid dims provided! ({self.dims})"

        # Compatibility with 2D and 3D ResNet. This needs to be done since
        # pytorch_grad_cam is explicitly compatible with the pre-implemented
        # ResNet18 class.
        if self.dims == 2:
            m = model_obj.resnet
        else:
            m = model_obj
        target_layers = [m.layer4[-1]]

        # The actual PyTorch model is always stored in an attribute!!
        super(GradCAMResNet, self).__init__(model=m,
                                            target_layers=target_layers,
                                            class_idx=class_idx)

        return

    def get_cam(self,
                tvt: str,
                patient_id: str,
                lesion_idx: int,
                slide_idx=None):
        """
        Performing Grad-CAM on a single sample. This is either a 2D slide or
        the entire volume of a tumor lesion.
        """
        input_tensor = self._get_input_tensor(patient_id=patient_id,
                                              image_format="images",
                                              tvt=tvt,
                                              lesion_idx=lesion_idx,
                                              slide_idx=slide_idx)
        return self.forward(input_tensor=input_tensor)

    def export_cam(self):
        """
        Exporting the Grad-CAM arrays for all patients in the dataset.
        """
        suffix_img = "img"
        suffix_cam = "grad_cam"

        dict_patient_ids = self.task.get_patient_ids()
        n_total = sum(
            [len(patient_ids) for patient_ids in dict_patient_ids.values()])
        pbar = tqdm(total=n_total, desc="Extracting Grad-CAM")
        for tvt in dict_patient_ids.keys():
            dir_export_base = self._get_dir_export_base(tvt=tvt)
            patient_ids = dict_patient_ids[tvt]
            for patient_id in patient_ids:
                dataset = self._get_dataset(image_format="images",
                                            tvt=tvt)
                sample_ids = [os.path.splitext(os.path.basename(triplet[1]))[0]
                              for
                              triplet in dataset.triplets_name_file_labels
                              if triplet[0] == patient_id]
                for lesion_idx, sample_id in enumerate(sample_ids):
                    img_grayscale, cam_grayscale = self.get_cam(tvt=tvt,
                                                                patient_id=patient_id,
                                                                lesion_idx=lesion_idx,
                                                                slide_idx=None)
                    # Exporting the input image.
                    self.export(arr=img_grayscale,
                                dir_export=dir_export_base,
                                sample_id=sample_id,
                                suffix=suffix_img)
                    # Exporting the Grad-CAM map.
                    self.export(arr=cam_grayscale,
                                dir_export=dir_export_base,
                                sample_id=sample_id,
                                suffix=suffix_cam)
                # Don't forget to update the progress bar!
                pbar.update(1)
        return

    def export_mask_arrs(self):
        """
        Exporting the mask arrays for all patients in the dataset.
        """
        dict_patient_ids = self.task.get_patient_ids()
        n_total = sum(
            [len(patient_ids) for patient_ids in dict_patient_ids.values()])
        pbar = tqdm(total=n_total, desc="Extracting transformed binary mask "
                                        "arrays")
        for tvt in dict_patient_ids.keys():
            patient_ids = dict_patient_ids[tvt]
            for patient_id in patient_ids:
                self._export_mask_arr_single_patient(patient_id=patient_id,
                                                     tvt=tvt)
                # Don't forget to update the progress bar!
                pbar.update(1)

        return

    def _export_mask_arr_single_patient(self,
                                        tvt: str,
                                        patient_id: str):
        """
        Helper function to export the transformed and mask array for a single
        patient.
        """
        dir_export_base = self._get_dir_export_base(tvt=tvt)
        suffix = "mask_anisotropic"

        dataset = self._get_dataset(image_format="masks",
                                    tvt=tvt)
        patient_file_idxs = [i for i, triplet in
                             enumerate(dataset.triplets_name_file_labels)
                             if triplet[0] == patient_id]
        sample_ids = [os.path.splitext(os.path.basename(triplet[1]))[0] for
                      triplet in dataset.triplets_name_file_labels
                      if triplet[0] == patient_id]
        for patient_file_idx, sample_id in zip(patient_file_idxs,
                                               sample_ids):
            if self.dims == 2:
                min_idx, max_idx = self._get_min_and_max_idx_per_file(
                    dataset=dataset,
                    patient_file_idx=patient_file_idx)
                # n_samples x height x width
                # The first index removes the triplet, the second index
                # removes the batch dimension.
                mask_tensor = torch.stack([dataset[idx][0][0, ...] for idx in
                                           range(min_idx, max_idx + 1)], axis=0)
            else:
                # height x depth x width
                mask_tensor, _a, _b = dataset[patient_file_idx]
                mask_tensor = mask_tensor[0, ...]

            # Making sure the mask array only contains 0 and 1.
            mask_arr_binarized = self._binarize_mask(mask_arr=mask_tensor)

            # Exporting the arrays.
            self.export(arr=mask_arr_binarized,
                        dir_export=dir_export_base,
                        sample_id=sample_id,
                        suffix=suffix)

        return

    def _get_dir_export_base(self,
                             tvt: str):
        """
        Unified approach to saving the exported volumes.
        """
        return os.path.join(self.task.dir_explainability,
                            self.model_id,
                            tvt)

    @staticmethod
    def _binarize_mask(mask_arr: np.ndarray,
                       threshold: float = 0.5):
        """
        Since the mask array will have continuous values after upscaling,
        re-binarize them for downstream analysis.
        """
        mask_arr_binarized = np.zeros(mask_arr.shape)
        mask_arr_binarized[mask_arr >= threshold] = 1

        return mask_arr_binarized

    def _get_input_tensor(self,
                          tvt: str,
                          patient_id: str,
                          image_format: str,
                          lesion_idx: int,
                          slide_idx: int):
        """
        Perform the actual Grad-CAM analysis based on the patient name and
        slide index (if dims==2).
        """
        assert (self.dims == 3) | (slide_idx is not None), (
            "If dims==2, a slide index needs to be provided!")

        dataset = self._get_dataset(image_format=image_format,
                                    tvt=tvt)
        # Get all the indices of the triplets related to the patient.
        # This may be more than 1, since a patient might have more than 1
        # tumor lesion, and lesions are stored in separate files.
        patient_file_idxs = [i for i, triplet in
                             enumerate(dataset.triplets_name_file_labels)
                             if triplet[0] == patient_id]
        assert lesion_idx < len(
            patient_file_idxs), (F"Index {lesion_idx} out of bounds for "
                                 F"patient with {len(patient_file_idxs)} "
                                 F"files!")
        relevant_patient_file_idx = patient_file_idxs[lesion_idx]
        if self.dims == 2:
            min_idx, max_idx = self._get_min_and_max_idx_per_file(
                dataset=dataset,
                patient_file_idx=
                relevant_patient_file_idx)
            n_slides = max_idx - min_idx + 1
            assert slide_idx < n_slides, (F"Slide index {slide_idx} out of "
                                          F"bounds for patient {patient_id} "
                                          F"with {n_slides} axial slides "
                                          F"across all tumor lesions!")
            dataset_idx = min_idx + slide_idx
        else:
            dataset_idx = relevant_patient_file_idx

        input_tensor, _a, _b = dataset[dataset_idx]

        return input_tensor

    @staticmethod
    def _get_min_and_max_idx_per_file(dataset,
                                      patient_file_idx: int):
        # Avoids errors when the relevant file index is zero.
        if patient_file_idx == 0:
            min_idx = -1
        else:
            min_idx = dataset.max_idx_per_file[patient_file_idx - 1]
        max_idx = dataset.max_idx_per_file[patient_file_idx]

        return min_idx + 1, max_idx

    def _get_dataset(self,
                     image_format: str,
                     tvt: str):
        """
        Return either a 2D or 3D dataset, from either train, val, or test.
        """
        include_channel_idxs = self.channel_couple if (image_format == "images") \
            else None
        if self.dims == 2:
            dataset = dataset_2d.VolumeDataset2D(
                file_task=self.file_task,
                image_format=image_format,
                tvt=tvt,
                transform=self.transform_2D,
                axial_only=True,
                triplets_name_file_labels=None,
                ablation_channel_idxs=None,
                include_channel_idxs=include_channel_idxs)
        else:
            dataset = dataset_3d.VolumeDataset3D(
                file_task=self.file_task,
                image_format=image_format,
                tvt=tvt,
                transform=self.transform_3D,
                triplets_name_file_labels=None,
                ablation_channel_idxs=None,
                include_channel_idxs=include_channel_idxs)

        return dataset

    def _get_model_and_dict_hyperparams(self,
                                        model_id: str,
                                        channel_couple: list,
                                        file_weights: str,
                                        file_hyperparams: str
                                        ):
        """
        Get the loaded model object as well as the dictionary containing the
        hyperparameters of the model.
        """
        if file_hyperparams is None:
            str_channel_couple = self.task.get_str_channel_couple(
                channel_couple=channel_couple)
            dir_model = os.path.join(self.task.dir_models, str_channel_couple,
                                     model_id)
            file_hyperparams = os.path.join(dir_model, "hyperparams.json")
        with open(file_hyperparams, 'r') as f:
            dict_hyperparams = json.load(f)

        if file_weights is None:
            file_weights = os.path.join(dir_model, "models", model_id + ".pth")
        model, n_channels_input, n_classes_per_label, dims = (
            ensemble.EnsembleModel.load_model(
                file_weights=file_weights,
                dict_hyperparams=dict_hyperparams))

        return model, dict_hyperparams

    def _get_class_idx(self):
        """
        Gets the highest index of a class for a single-label task.
        """
        n_classes_per_label = self.task.n_classes_per_label
        assert len(n_classes_per_label) == 1, ("For now, only single-label "
                                               "tasks are supported!")
        # Returns an index, therefore minus 1.
        return n_classes_per_label[0] - 1
