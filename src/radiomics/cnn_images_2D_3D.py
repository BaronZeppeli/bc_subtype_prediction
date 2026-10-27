import sys
import os
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy.ndimage as scnd
# import tdigest
from tqdm import tqdm

from data_processing.voxelizer import VoxelArray, VoxelArrayProcessor

class ImageToBBox:

    N_TIMEPOINTS_DCE_MRI = 7

    def __init__(self,
                 dir_image: str,
                 modalities: list,
                 modalities_dynamic: list,
                 modalities_z_score: list,
                 file_mask_roi: str,
                 file_mask_foreground: str,
                 file_normalization: str,
                 rthreshold_component: float):

        self.name = os.path.basename(dir_image)
        self.modalities = modalities
        self.files_image = [os.path.join(dir_image,
                                         self.name + "_" + modality + ".nii")
                            for
                            modality in modalities]
        self.file_mask_roi = file_mask_roi
        self.file_mask_foreground = file_mask_foreground

        self.rthreshold_component = rthreshold_component if rthreshold_component is not None else 0

        self.list_bbox_component_img = None
        self.list_bbox_component_mask = None
        self.voxel_size = None

        # NOTE: There is no need to keep track of the global indices,
        # since the GlobalNormalization class already identifies the global
        # modalities!
        modalities_dynamic = modalities_dynamic if modalities_dynamic else []
        modalities_z_score = modalities_z_score if modalities_z_score else []
        self.dynamic_idxs = [i for i, modality in enumerate(modalities) if
                             modality in modalities_dynamic]
        self.z_score_idxs = [i for i, modality in enumerate(modalities) if
                             modality in modalities_z_score]

        print("DYNAMIC IDXS", self.dynamic_idxs)
        print("Z-SCORE IDXS", self.z_score_idxs)

        # Initialize object for global normalization.
        if file_normalization is not None:
            self.gn = GlobalNormalizer(file_normalization=file_normalization)
        else:
            self.gn = None

        return

    @classmethod
    def from_QIN(cls,
                 sample_id: str,
                 trial_timepoint: str,
                 closest_timepoint_files: list,
                 file_adc: str,
                 file_mask_roi: str,
                 file_mask_foreground: str,
                 file_normalization: str,
                 rthreshold_component: float):
        """
        Load files from the QIN dataset.
        """
        dir_image = os.path.dirname(file_adc)
        if len(closest_timepoint_files) !=ImageToBBox.N_TIMEPOINTS_DCE_MRI:
            warnings.warn(F"Invalid number of DCE-MRI files detected! "
                          F"Expected {ImageToBBox.N_TIMEPOINTS_DCE_MRI}, "
                          F"got {len(closest_timepoint_files)}.")

        modalities_dynamic = [F"T1_POST_DYNAMIC_{i}" for i in
                              range(len(closest_timepoint_files))]
        modalities = modalities_dynamic + ["ADC"]
        modalities_z_score = []
        obj = cls(dir_image=dir_image,
                  modalities=modalities,
                  modalities_dynamic=modalities_dynamic,
                  modalities_z_score=modalities_z_score,
                  file_mask_roi=file_mask_roi,
                  file_mask_foreground=file_mask_foreground,
                  file_normalization=file_normalization,
                  rthreshold_component=rthreshold_component)

        obj.name = F"{sample_id}_{trial_timepoint}"
        # Important step to account for the excess of timepoints in QIN
        # compared to TUE018.
        obj.files_image = closest_timepoint_files + [file_adc]

        return obj

    def perform_normalization_resnet(self,
                                     dir_export: str,
                                     separate_modalities:bool=False):
        """
        Sums up all functionality to normalize bboxes for ResNet.
        """
        # Of course always include the voxels inside the bbox but outside the
        # tumor ROI!
        self.create_connected_components(include_voxels_outside_mask=True)
        # Clipping since it was done in TUE018.
        self.normalize_global(clip=True)
        self.normalize_dynamic()
        self.normalize_z_score()

        if separate_modalities:
            self._export_volumes_separate_modalities(dir_export=dir_export)
        else:
            self.export_volumes(dir_export=dir_export,
                                format="nii",
                                exporting_images=True)

        return

    def create_connected_components(self,
                                    include_voxels_outside_mask: bool):
        vap = VoxelArrayProcessor(volume_files=self.files_image,
                                  ROI_file=self.file_mask_roi)
        print("FILES IN THEIR ORDER", self.files_image)
        print("MASK FILE ROI", self.file_mask_roi)
        print("MASK FILE FOREGROUND", self.file_mask_foreground)
        # This is needed for the export of connected components.
        self.voxel_size = vap.voxel_array.voxel_size
        # Identifying the connected components of the mask array.
        if self.file_mask_roi is not None:
            mask_arr = vap.ROI_array[..., 0]
        else:
            # In case no mask array for the tumor was provided, just use the
            # entire volume as one connected component.
            warnings.warn("No ROI file provided, exporting the entire volume.")
            mask_arr = np.ones(vap.voxel_array.array.shape[:-1])
        # TODO: Change the handling of ausläufer, satellites and lymph node
        #  metastases once they are included in the analysis.
        primarius_id = 1
        auslaeufer_id = 2
        lymphnode_id = 3
        mask_arr[mask_arr > primarius_id] = 0
        # THE ACTUAL COMPONENTS COMPUTATION.
        components, n_components = scnd.label(mask_arr)
        n_voxels_foreground = np.sum(components != 0)
        print(F"Number of connected components in {self.name}: ", n_components)
        self.list_bbox_component_img = []
        self.list_bbox_component_mask = []
        n_components_excluded = 0
        # Has to start at 1 since 0 signifies the background component.
        for component_idx in range(1, n_components + 1):
            valid_idxs = np.where(components == component_idx)
            n_voxels_component = valid_idxs[0].shape[0]
            # Kicks out all components whose fraction of all
            # foreground voxels is smaller than a specified threshold.
            if n_voxels_component / n_voxels_foreground < self.rthreshold_component:
                n_components_excluded += 1
                continue
            limits_bounding_box = np.array([np.min(valid_idxs, axis=1),
                                            np.max(valid_idxs, axis=1) + 1]).T
            bbox_component_img = np.copy(vap.voxel_array.array)
            bbox_component_mask = np.copy(mask_arr)
            if not include_voxels_outside_mask:
                # Mask the voxels INSIDE the bbox BUT outside the ROI,
                # for computing a more concise mean and std for clustering.
                bbox_component_img[components != component_idx] = np.nan
            bbox_component_img = bbox_component_img[
                limits_bounding_box[0, 0]:limits_bounding_box[
                    0, 1],
                limits_bounding_box[1, 0]:limits_bounding_box[
                    1, 1],
                limits_bounding_box[2, 0]:limits_bounding_box[
                    2, 1]]
            # Apply the same cropping to the mask, for later use in
            # explainability analysis.
            bbox_component_mask = bbox_component_mask[
                limits_bounding_box[0, 0]:limits_bounding_box[
                    0, 1],
                limits_bounding_box[1, 0]:limits_bounding_box[
                    1, 1],
                limits_bounding_box[2, 0]:limits_bounding_box[
                    2, 1]]

            self.list_bbox_component_img += [bbox_component_img]
            self.list_bbox_component_mask += [bbox_component_mask]

        print(F"Number of excluded components: {n_components_excluded}")

        return

    def export_images(self,
                      dir_export: str,
                      format: str):
        if not os.path.isdir(dir_export):
            os.makedirs(dir_export)

        img_idx = 0
        for bbox_component in self.list_bbox_component_img:
            for slice_idx in range(bbox_component.shape[0]):
                # Make the images 4-dimensional in order to be compatible
                # with the SliceAnalyzer.
                img = bbox_component[slice_idx][None, ...]
                suffix = ImageToBBox._get_suffix(idx=img_idx)

                file_export = os.path.join(dir_export,
                                           self.name
                                           + "_" + suffix
                                           + "." + format)
                if format.lower() == "npy":
                    with open(file_export, 'wb') as f:
                        np.save(f, img)
                else:
                    raise ValueError("Currently only npy as export format "
                                     "supported.")

                img_idx += 1
                assert img_idx <= 9999, ("Currently only up to 10,000 images "
                                         "per patient are supported!")

        return

    def export_volumes(self,
                       dir_export: str,
                       format: str,
                       exporting_images: bool):
        if not os.path.isdir(dir_export):
            os.makedirs(dir_export)

        list_bbox = self.list_bbox_component_img if exporting_images \
            else self.list_bbox_component_mask

        for component_idx, bbox_component in enumerate(list_bbox):
            suffix = ImageToBBox._get_suffix(idx=component_idx)
            file_export = os.path.join(dir_export,
                                       self.name
                                       + "_" + suffix
                                       + "." + format)
            va_export = VoxelArray(source_voxel=bbox_component,
                                   voxel_size=self.voxel_size)
            if format.lower() == "npy":
                va_export.to_npy(output_path=file_export)
            elif format.lower() == "nii":
                # Export the connected component as a 4D array.
                # Masks should be exported as 3D.
                va_export.to_nifti(output_path=file_export,
                                   enforce_3D=not exporting_images)
            else:
                raise ValueError(
                    "Currently only npy and nii as export format supported.")
        return

    def _export_volumes_separate_modalities(self,
                                            dir_export: str, ):
        """
        Export modalities into separate 3D BBOX NIFTI files.
        """
        for component_idx, bbox_component in enumerate(self.list_bbox_component_img):
            suffix = ImageToBBox._get_suffix(idx=component_idx)

            for modality_idx, modality in enumerate(self.modalities):
                file_export = os.path.join(dir_export,
                                           self.name+ "_" + suffix,
                                           self.name
                                           + "_" + suffix
                                           + "_" + modality
                                           + ".nii")
                os.makedirs(os.path.dirname(file_export), exist_ok=True)

                va_export = VoxelArray(source_voxel=bbox_component[...,modality_idx],
                                       voxel_size=self.voxel_size)
                va_export.to_nifti(output_path=file_export,
                                   enforce_3D=True)

        return



    def normalize_global(self,
                         clip: bool):
        assert self.gn is not None, ("A normalization file needs to be "
                                     "provided for the global normalization.")

        list_bbox_component_new = []

        for connected_component in self.list_bbox_component_img:
            connected_component_new = self.gn.apply_normalization(
                arr=connected_component,
                modalities=self.modalities,
                clip=clip)
            list_bbox_component_new += [connected_component_new]

        self.list_bbox_component_img = list_bbox_component_new

        return

    def normalize_dynamic(self):

        # IMPORTANT: For the normalization of all MRI parameters except ADC,
        # consider all voxels in the foreground, i.e. all breast voxels,
        # across all timepoints.
        vap = VoxelArrayProcessor(volume_files=self.files_image,
                                  ROI_file=self.file_mask_foreground)
        dynamic_vals = vap.voxel_array.array[vap.ROI_array[..., 0] != 0][
            ..., self.dynamic_idxs]

        mean = np.mean(dynamic_vals)
        std = np.std(dynamic_vals)

        list_bbox_component_new = []

        for connected_component in self.list_bbox_component_img:
            connected_component_new = connected_component.copy()
            connected_component_new[...,
            self.dynamic_idxs] = ((connected_component_new[
                                       ..., self.dynamic_idxs] - mean) /
                                  std)
            list_bbox_component_new += [connected_component_new]

        self.list_bbox_component_img = list_bbox_component_new

        return

    def normalize_z_score(self):
        # Similar to DCE-MRI, normalize across all foreground voxels, however
        # in this case separately for each parameter.
        vap = VoxelArrayProcessor(volume_files=self.files_image,
                                  ROI_file=self.file_mask_foreground)
        z_score_vals = vap.voxel_array.array[vap.ROI_array[..., 0] != 0][
            ..., self.z_score_idxs]
        mean = np.mean(z_score_vals, axis=0)
        std = np.std(z_score_vals, axis=0)

        list_bbox_component_new = []

        for connected_component in self.list_bbox_component_img:
            connected_component_new = connected_component.copy()
            connected_component_new[..., self.z_score_idxs] = (
                    (connected_component_new[
                         ..., self.z_score_idxs] - mean) / std)
            list_bbox_component_new += [connected_component_new]

        self.list_bbox_component_img = list_bbox_component_new

        return

    def get_list_bbox_component(self):
        return self.list_bbox_component_img

    @staticmethod
    def _get_suffix(idx):
        # Currently support for up to 10,000 files per patient!
        suffix = "000" + str(
            idx) if idx < 10 else "00" + str(
            idx) if idx < 100 else "0" + str(
            idx) if idx < 1000 else str(idx)
        return suffix


class GlobalNormalizer:
    """
    VERY IMPORTANT: For normalization, consider all the foreground voxels,
    i.e. pass a mask file which masks the entire breast!
    """

    @staticmethod
    def compute_mean_std_bounds(modalities_global: list,
                                percentile_lower_clip: int,
                                percentile_upper_clip: int,
                                dirs_image: list,
                                files_mask_foreground: list,
                                file_export: str):
        percentile_lower_clip = percentile_lower_clip if (
                percentile_lower_clip is not
                None) else 0
        percentile_upper_clip = percentile_upper_clip if (
                percentile_lower_clip is not
                None) else 100
        assert percentile_lower_clip < percentile_upper_clip, (
            "The lower bound "
            "must be smaller than the upper bound.")

        vals_per_modality_global = [[] for _ in modalities_global]

        for dir_image, file_mask_foreground in zip(dirs_image,
                                                   files_mask_foreground):

            name = os.path.basename(dir_image)
            files_image = [os.path.join(dir_image,
                                        name + "_" + modality + ".nii")
                           for
                           modality in modalities_global]
            vap = VoxelArrayProcessor(volume_files=files_image,
                                      ROI_file=file_mask_foreground)
            arr_img = vap.voxel_array.array
            if np.sum(arr_img < 0) > 0:
                warnings.warn("WARNING: Negative voxel values detected! They "
                              "will be replaced with 0! If this behavior is "
                              "unwanted, change the code!!")
                arr_img[arr_img < 0] = 0
            arr_mask = vap.ROI_array[..., 0]
            vals_per_modality = [arr_img[..., i][arr_mask > 0]
                                 .flatten().tolist() for i
                                 in range(arr_img.shape[-1])]

            vals_per_modality_global = [vals_per_modality_global[i]
                                        + vals_per_modality[i] for i in
                                        range(len(modalities_global))]

        mean_per_modality = []
        std_per_modality = []
        lower_bound_clip_per_modality = []
        upper_bound_clip_per_modality = []

        for vals_single_modality in vals_per_modality_global:
            mean_per_modality += [np.mean(vals_single_modality)]
            std_per_modality += [np.std(vals_single_modality)]
            lower_bound_clip_per_modality += [
                np.percentile(vals_single_modality,
                              percentile_lower_clip)]
            upper_bound_clip_per_modality += [
                np.percentile(vals_single_modality,
                              percentile_upper_clip)]

        print("OG METHOD", mean_per_modality, std_per_modality,
              lower_bound_clip_per_modality, upper_bound_clip_per_modality)

        # Export the statistics.
        dict_export = {"Modality": modalities_global,
                       "Mean": mean_per_modality,
                       "Std": std_per_modality,
                       "Lower_Bound_Clip": lower_bound_clip_per_modality,
                       "Upper_Bound_Clip": upper_bound_clip_per_modality,
                       "Percentile_Lower_Clip": [percentile_lower_clip] * len(
                           modalities_global),
                       "Percentile_Upper_Clip": [percentile_upper_clip] * len(
                           modalities_global)
                       }
        df_export = pd.DataFrame(dict_export)

        if not os.path.isdir(os.path.dirname(file_export)):
            os.makedirs(os.path.dirname(file_export))
        df_export.to_csv(file_export, index=False)

        return

    def __init__(self,
                 file_normalization: str, ):

        self.df_normalization = pd.read_csv(file_normalization)

        return

    def apply_normalization(self,
                            arr: np.ndarray,
                            modalities: list,
                            clip: bool):
        assert arr.shape[-1] == len(
            modalities), "Incorrect number of modalities provided."
        global_idxs = [i for i, modality in enumerate(modalities) if
                       modality in self.df_normalization["Modality"].to_list()]

        print("GLOBAL IDXS", global_idxs)

        arr_normalized = np.copy(arr)
        for global_idx in global_idxs:
            modality_global = modalities[global_idx]
            row_modality_global = self.df_normalization[
                self.df_normalization["Modality"] == modality_global]
            mean = row_modality_global["Mean"].iloc[0]
            std = row_modality_global["Std"].iloc[0]
            lower_bound_clip = row_modality_global["Lower_Bound_Clip"].iloc[0]
            upper_bound_clip = row_modality_global["Upper_Bound_Clip"].iloc[0]

            print(F"For modality {modality_global}, applying mean: {mean}, "
                  F"std: {std}, lower_bound: {lower_bound_clip}, "
                  F"upper_bound: {upper_bound_clip}.")

            # First clip, then normalize, in accordance with nnU-Net!
            if clip:
                arr_normalized[..., global_idx] = np.clip(arr[..., global_idx],
                                                          lower_bound_clip,
                                                          upper_bound_clip)
                assert (np.all(
                    arr_normalized[..., global_idx] >= lower_bound_clip) &
                        np.all(arr_normalized[
                                   ..., global_idx] <= upper_bound_clip)), (
                    "Clipping somehow failed.")
            arr_normalized[..., global_idx] = (
                    (arr_normalized[..., global_idx] - mean)
                    / std)

        return arr_normalized


class RunningStats:
    """
    Class to compute the mean, std and percentiles in an online fashion.
    """

    def __init__(self,
                 percentile_lower: float,
                 percentile_upper: float, ):
        self.n = 0  # Total number of elements processed
        self.mean = 0  # Running mean
        self.M2 = 0  # Sum of squared differences for variance

        self.digest = tdigest.TDigest()
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper

    def update(self, chunk):
        """
        Update the running mean and M2 with a new chunk of data.
        """
        chunk = chunk.flatten()
        chunk = chunk[~np.isnan(chunk)]
        chunk_size = chunk.shape[0]  # Number of elements in the chunk
        chunk_mean = np.mean(chunk)  # Mean of the current chunk
        chunk_var = np.var(chunk, ddof=0)  # Variance of the current chunk

        # Incremental update of the mean
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_size / (self.n + chunk_size)

        # Update sum of squared differences (M2)
        self.M2 += chunk_size * chunk_var + delta ** 2 * (
                self.n * chunk_size) / (self.n + chunk_size)

        # Update count of total elements
        self.n += chunk_size

        # Update the running percentile measurement.
        for elem in tqdm(chunk.flatten(), desc="Updating the running "
                                               "percentiles"):
            self.digest.update(elem)

    def finalize(self):
        """
        Finalize the statistics. Calculate variance and standard deviation.
        """
        variance = self.M2 / (self.n - 1) if self.n > 1 else np.nan
        std_dev = np.sqrt(variance)
        lower_bound = self.digest.percentile(self.percentile_lower)
        upper_bound = self.digest.percentile(self.percentile_upper)

        return self.mean, std_dev, lower_bound, upper_bound
