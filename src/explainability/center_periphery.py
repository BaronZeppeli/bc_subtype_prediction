import json
import sys
import os
import warnings

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from resnet_subtype.task import Task
import data_processing.voxelizer as voxelizer
import data_processing.handler_nifti as handler_nifti


class PhysicalDimensions:

    def __init__(self,
                 file_task: str,
                 model_id: str):
        self.task = Task(file_task=file_task)
        self.model_id = model_id

        return

    def get_voxel_size_transformed(self,
                                   sample_id: str,
                                   tvt: str,
                                   voxel_size_original: float = 1):
        """
        Note: The unit of original_size is mm!
        """
        arr_mask_original, arr_mask_transformed = (
            self.get_original_and_transformed_mask_arr(sample_id=sample_id,
                                                       tvt=tvt))

        print("Original mask shape:", arr_mask_original.shape)
        print("Transformed mask shape:", arr_mask_transformed.shape)

        ratio_axial_original = arr_mask_original.shape[1] / \
                               arr_mask_original.shape[2]
        ratio_axial_transformed = arr_mask_transformed.shape[1] / \
                                  arr_mask_transformed.shape[2]

        assert np.isclose(ratio_axial_original, ratio_axial_transformed,
                          atol=0.01), \
            (F"The ratio of height vs width is not matching! (Original: "
             F"{ratio_axial_original}, Transformed: {ratio_axial_transformed})")

        # The physical length of an array is the dimensions of the original
        # shape multiplied by the KNOWN voxel size of that original array.
        physical_dims = np.array(arr_mask_original.shape) * voxel_size_original
        print("The physical dimensions of the array:", physical_dims)

        # Same procedure as above but reversed to get the transformed
        # voxel size.
        voxel_size_transformed = physical_dims / np.array(
            arr_mask_transformed.shape)

        return voxel_size_transformed

    def get_original_and_transformed_mask_arr(self,
                                              sample_id: str,
                                              tvt: str):
        """
        Loads the mask arrays.
        """
        file_mask_original = os.path.join(self.task.dir_masks,
                                          tvt,
                                          sample_id + ".nii")
        file_mask_transformed = os.path.join(self.task.dir_explainability,
                                             self.model_id,
                                             tvt,
                                             sample_id,
                                             sample_id +
                                             "_mask_anisotropic.nii")

        arr_mask_original = voxelizer.VoxelArray(
            source_voxel=file_mask_original).array[..., 0].astype(int)
        arr_mask_transformed = voxelizer.VoxelArray(
            source_voxel=file_mask_transformed).array[..., 0].astype(int)

        return arr_mask_original, arr_mask_transformed

    def export_array_with_voxel_size(self,
                                     arr: np.ndarray,
                                     voxel_size: list,
                                     sample_id: str,
                                     tvt: str,
                                     suffix: str):
        """
        Simple function to save a function along its voxel size as stored in
        the affine of a NIFTI file.
        """
        affine = handler_nifti.update_affine_voxel_size(affine=None,
                                                        voxel_size=voxel_size,
                                                        encoded=False)
        # NIFTI only works with floats!!
        va = voxelizer.VoxelArray(source_voxel=arr.astype(np.float32),
                                  affine=affine)
        enforce_3D = (arr.ndim == 3) | ((arr.ndim == 4) & (arr.shape[-1] == 1))
        dir_export = os.path.join(self.task.dir_explainability,
                                  self.model_id,
                                  tvt,
                                  sample_id)
        file_export = os.path.join(dir_export,
                                   sample_id + suffix + ".nii")

        va.to_nifti(file_export,
                    enforce_3D=enforce_3D)
        print(F"Exported array to {file_export}")

        return


class TumorCenter:

    def __init__(self,
                 file_task: str,
                 model_id: str,
                 voxel_size_original: int
                 ):
        self.task = Task(file_task=file_task)
        self.model_id = model_id
        self.voxel_size_original = voxel_size_original
        self.physical_dimensions = PhysicalDimensions(file_task=file_task,
                                                      model_id=model_id)

        return

    def get_tumor_centers(self,
                          sample_id: str,
                          tvt: str,
                          min_distance_peaks_physical: float
                          ):
        """
        Performs the center analysis and stores the results in a NIFTI file
        along an affine with the extracted voxel size.
        """
        segmentation, centers, voxel_size_transformed = (
            self._get_center_segmentations(
                sample_id=sample_id,
                tvt=tvt,
                min_distance_peaks_physical=min_distance_peaks_physical))
        self.physical_dimensions.export_array_with_voxel_size(arr=segmentation,
                                                              voxel_size=voxel_size_transformed,
                                                              sample_id=sample_id,
                                                              tvt=tvt,
                                                              suffix="_segmentation")
        self.physical_dimensions.export_array_with_voxel_size(arr=centers,
                                                              voxel_size=voxel_size_transformed,
                                                              sample_id=sample_id,
                                                              tvt=tvt,
                                                              suffix="_centers")

        return

    def get_center_periphery_shells(self,
                                    ps: list):
        """
        Split a mask into shells based on the distance to the nearest
        background voxel.
        """
        for tvt in ["train", "val", "test"]:
            dir_samples = os.path.join(self.task.dir_explainability,
                                       self.model_id,
                                       tvt)
            sample_ids = sorted(os.listdir(dir_samples))
            for sample_id in sample_ids:
                arr_mask_percentiles = self._center_periphery_cdf(
                    sample_id=sample_id,
                    tvt=tvt,
                    ps=ps)

                # In mm.
                voxel_size_transformed = self.physical_dimensions.get_voxel_size_transformed(
                    sample_id=sample_id,
                    tvt=tvt,
                    voxel_size_original=self.voxel_size_original)
                self.physical_dimensions.export_array_with_voxel_size(
                    arr=arr_mask_percentiles,
                    voxel_size=voxel_size_transformed,
                    sample_id=sample_id,
                    tvt=tvt,
                    suffix="_shells")

        return

    def _get_center_segmentations(self,
                                  sample_id: str,
                                  tvt: str,
                                  min_distance_peaks_physical: float):
        """
        Helper function to determine the centers and their associated voxels.
        """
        # In mm.
        voxel_size_transformed = self.physical_dimensions.get_voxel_size_transformed(
            sample_id=sample_id,
            tvt=tvt,
            voxel_size_original=self.voxel_size_original)
        assert np.allclose(voxel_size_transformed,
                           voxel_size_transformed[0],
                           atol=0.01), "The mask array is not isotropic!"
        # Round up to make sure the minimal distance is always met.
        min_distance_peaks_voxel = int(np.ceil(
            min_distance_peaks_physical / voxel_size_transformed[0]))
        assert min_distance_peaks_voxel > 0, (
            "The physical center distance must "
            "be selected such that the voxel "
            "center distance is at least 1!")

        _, arr_mask_transformed = (
            self.physical_dimensions.get_original_and_transformed_mask_arr(
                sample_id=sample_id))

        print("The array dimensions: ", arr_mask_transformed.shape)
        print("The minimal voxel size between 2 peaks: ",
              min_distance_peaks_voxel)

        # Distance transform.
        dist = ndimage.distance_transform_edt(arr_mask_transformed)

        # Find local maxima in distance map within mask
        # peak_local_max returns coords in (z,y,x) order by default for 3D.
        peaks = peak_local_max(image=dist,
                               min_distance=min_distance_peaks_voxel)
        print("Peaks:\n", peaks)

        # Create markers image (integer labels).
        centers = np.zeros(dist.shape).astype(int)
        for i, p in enumerate(peaks, start=1):
            centers[tuple(p)] = i

        # Watershed on negative distance (so basins are deep regions).
        segmentation = watershed(-dist, centers, mask=arr_mask_transformed)

        return segmentation, centers, voxel_size_transformed

    def _center_periphery_cdf(self,
                              sample_id: str,
                              tvt: str,
                              ps: list):
        """
        Separate voxels into shells of equal volume by sorting the distances
        to the nearest background voxel.
        """
        _, arr_mask_transformed = (
            self.physical_dimensions.get_original_and_transformed_mask_arr(
                sample_id=sample_id,
                tvt=tvt))
        distances = ndimage.distance_transform_edt(arr_mask_transformed)
        sorted_dists = np.sort(distances[distances > 0]).flatten()

        # Basic logic: The tumor center is all voxels in the greatest
        # provided percentile, i.e. the 75th percentile represents the
        # innermost 25% of voxels.
        assert 0 in ps, "The 0 percentile needs to be in ps!"
        # Sanity check.
        ps = sorted(ps)
        threshold_dists = [
            self._get_percentile_dist(sorted_dists=sorted_dists, p=p) for p in
            ps]

        arr_mask_percentiles = np.zeros(arr_mask_transformed.shape)
        for threshold_idx, threshold_dist in enumerate(threshold_dists):
            # Essentially overwrites the voxels multiple times as we get
            # closer to the center.
            arr_mask_percentiles[distances >= threshold_dist] = len(
                threshold_dists) - threshold_idx

        # Sanity checks.
        assert np.sum(arr_mask_percentiles > 0) == np.sum(
            arr_mask_transformed > 0), "Something went wrong."

        for p_idx, p in enumerate(ps):
            if p_idx == len(ps) - 1:
                ratio_required = 1 - p
            else:
                ratio_required = ps[p_idx + 1] - p
            p_label = len(ps) - p_idx
            ratio_p = np.sum(arr_mask_percentiles == p_label) / np.sum(
                arr_mask_percentiles > 0)
            assert np.isclose(ratio_p, ratio_required, atol=0.03), \
                (F"The {p} percentile constitutes {ratio_p} voxels instead "
                 F"of the required {ratio_required}!")

        return arr_mask_percentiles

    @staticmethod
    def _get_percentile_dist(sorted_dists: np.ndarray,
                             p: float):
        """
        Get the distance thresholds for the different percentiles.
        """
        assert (0 <= p <= 1), "p must be between 0 and 1!"

        threshold_dist = None

        # Makes iteration more efficient, since there can be many repeat values.
        unique_dists = np.sort(np.unique(sorted_dists))

        for dist in unique_dists:
            # Functions as the cumulative distribution function.
            ratio = np.sum(sorted_dists < dist) / sorted_dists.shape[0]
            if ratio >= p:
                threshold_dist = dist
                break

        return threshold_dist


class EvaluateGradCAM:
    TVT = ["train", "val", "test"]
    # From the Seaborn deep palette.
    ORDERED_COLORS = [
        # "#4C72B0",  # blue, commented out since it represents "all" in
        # the pretraining plot.
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#937860",  # brown
        "#DA8BC3",  # pink
        "#8C8C8C",  # gray
        "#CCB974",  # yellow
        "#64B5CD",  # cyan
    ]

    def __init__(self,
                 file_task: str,
                 dict_models: dict):
        self.task = Task(file_task=file_task)
        self.dict_models = dict_models
        self.version_names = list(self.dict_models.keys())

        return

    def generate_grad_cam_plots_spherical(self,
                                          model_id: str,
                                          sample_id: str,
                                          tvt: str,
                                          n_bins: int,
                                          modality_name: str,
                                          export: bool):
        dir_sample = os.path.join(self.task.dir_explainability,
                                  model_id,
                                  tvt,
                                  sample_id)
        file_grad_cam = os.path.join(dir_sample, sample_id + "_grad_cam.nii")
        file_centers = os.path.join(dir_sample, sample_id + "_centers.nii")
        file_segmentation = os.path.join(dir_sample,
                                         sample_id + "_segmentation.nii")

        arr_grad_cam = voxelizer.VoxelArray(source_voxel=file_grad_cam).array[
            ..., 0]
        va_centers = voxelizer.VoxelArray(source_voxel=file_centers)
        arr_center = va_centers.array[..., 0]
        voxel_size = va_centers.voxel_size
        arr_segmentation = \
            voxelizer.VoxelArray(source_voxel=file_segmentation).array[..., 0]

        print(F"Voxel size of sample {sample_id}: {voxel_size}")

        assert (arr_grad_cam.shape == arr_center.shape) & (
                arr_grad_cam.shape == arr_segmentation.shape), \
            ("The Grad-CAM, centers and segmentation arrays do not have the "
             "same dimensions!")
        assert (np.unique(arr_center).shape[0] - 1) == np.sum(
            arr_center != 0), ("The centers array does not consist of "
                               "single voxels denoting the center (>0) "
                               "and background (=0) for the rest!")
        assert np.all(np.sort(np.unique(arr_center)) == np.sort(np.unique(
            arr_segmentation))), ("The centers and segmentation arrays do not "
                                  "have the same labels!")

        # To figure out how much activation was outside the tumor.
        sum_grad_total = np.sum(arr_grad_cam)
        print("The total sum of Grad-CAM values is: ", sum_grad_total)

        center_labels = np.sort(np.unique(arr_center[arr_center != 0]))
        for center_label in center_labels:
            # To set the activation of a distance bin into relation.
            sum_grad_center = np.sum(arr_grad_cam[arr_segmentation ==
                                                  center_label])
            print("The sum of Grad-CAM values within the current tumor center "
                  "is: ", sum_grad_center)
            # Avoids errors when displaying bar plots for volumes without any
            # activation.
            if sum_grad_center == 0:
                warnings.warn("Total activation of 0 detected!")
                sum_grad_center = 1e-10

            # This is just one voxel, therefore remove the surrounding array.
            center_coord = np.array(np.where(arr_center == center_label)).T[0]
            associated_coords = np.array(np.where(arr_segmentation ==
                                                  center_label)).T
            associated_dists = np.linalg.norm(associated_coords -
                                              center_coord, axis=1)
            max_dist = np.max(associated_dists)
            # Add -1 at the beginning to ensure the first bin works properly.
            max_dist_per_bin = [-1] + [i_bin * max_dist / n_bins for i_bin in
                                       range(1, n_bins + 1)]
            rel_grad_per_bin = []
            rel_vol_per_bin = []
            # max_dist_per_bin has length n_bins + 1.
            for i_bin in range(1, n_bins + 1):
                # A voxel is within a bin if it is greater than the max
                # distance of the previous bin and less then or equal to the
                # max distance of the current bin.
                bin_idxs = ((max_dist_per_bin[i_bin - 1] <
                             associated_dists) & (associated_dists <=
                                                  max_dist_per_bin[i_bin]))
                bin_coords = associated_coords[bin_idxs]
                sum_grad_bin = np.sum(arr_grad_cam[tuple(bin_coords.T)])
                # Will sum up to 1 this way.
                rel_grad_per_bin += [sum_grad_bin / sum_grad_center]
                # Will also sum up to 1.
                rel_vol_per_bin += [
                    bin_coords.shape[0] / associated_coords.shape[0]]

            rel_grad_per_bin = np.array(rel_grad_per_bin)
            rel_vol_per_bin = np.array(rel_vol_per_bin)
            adjusted_grad_per_bin = rel_grad_per_bin / rel_vol_per_bin
            # Ensure it sums up to 1.
            sum_adjusted = np.sum(adjusted_grad_per_bin)
            # Avoids divide-by-zero errors.
            sum_adjusted = sum_adjusted if sum_adjusted > 0 else 1e-10
            adjusted_grad_per_bin /= sum_adjusted
            # The actual plotting.
            if export:
                file_export = (F"{sample_id}_center_distance_idx="
                               F"{int(center_label)}.png")
                file_export = os.path.join(dir_sample, file_export)
            else:
                file_export = None
            self._bar_plot(y_series=[rel_grad_per_bin,
                                     rel_vol_per_bin,
                                     adjusted_grad_per_bin],
                           names=["Grad-CAM activation",
                                  "Volume",
                                  "Grad-CAM adjusted by volume"],
                           colors=self.ORDERED_COLORS[:3],
                           modality_name=modality_name,
                           file_export=file_export)

        return

    def shell_plot_single(self,
                          sample_id: str,
                          tvt:str):
        shells_per_version={}
        for version in self.dict_models.keys():
            dir_samples = os.path.join(self.task.dir_explainability,
                                       version,
                                       tvt)

            file_grad_cam = os.path.join(dir_samples,
                                         sample_id,
                                         sample_id + "_grad_cam.nii")
            file_shells = os.path.join(dir_samples,
                                       sample_id,
                                       sample_id + "_shells.nii")

            arr_grad_cam = \
                voxelizer.VoxelArray(source_voxel=file_grad_cam).array[
                    ..., 0]
            arr_shells = \
                voxelizer.VoxelArray(source_voxel=file_shells).array[
                    ..., 0]

            assert arr_grad_cam.shape == arr_shells.shape, \
                ("The Grad-CAM, centers and segmentation arrays do not "
                 "have the same dimensions!")

            # To figure out how much activation was inside the tumor.
            sum_grad_total = np.sum(arr_grad_cam[arr_shells > 0])
            # Avoids errors when displaying bar plots for volumes
            # without any TUMOR activation.
            if sum_grad_total == 0:
                warnings.warn("Total tumor activation of 0 detected!")
                sum_grad_total = 1e-10

            shell_labels = np.sort(
                np.unique(arr_shells[arr_shells > 0]))
            rel_activation_per_shell = []
            for shell_label in shell_labels:
                # To set the activation of a distance percentile into
                # relation.
                sum_grad_shell = np.sum(
                    arr_grad_cam[arr_shells == shell_label])
                # Always needs to be relative!
                rel_activation_per_shell += [
                    sum_grad_shell / sum_grad_total]

            shells_per_version[version] = rel_activation_per_shell

        y_series = [shells_per_version[version] for version in
                    self.version_names]
        names = self.dict_models.keys()
        colors = self.ORDERED_COLORS[:len(names)]

        sample_ids = sorted(os.listdir(dir_samples))
        n_lesions = len([si for si in sample_ids if
                         ("_".join(
                             sample_id.split("_")[:-1]) in si)])

        self._bar_plot(y_series=y_series,
                       names=names,
                       colors=colors,
                       sample_id=sample_id,
                       n_lesions=n_lesions,
                       file_export=None)

        return

    def generate_shell_plots(self):
        """
        Create and save the plots based on the shell distribution.
        """
        # Collect all important information along the road, for later
        # reproducibility.
        dict_shell_plots = {}
        for tvt in self.TVT:
            print("==========")
            print(F"Processing {tvt} dataset...")
            dict_shell_plots[tvt] = {}
            for version in self.dict_models.keys():
                print(F"Processing version {version}...")
                model_id = self.dict_models[version]["model_id"]
                dir_samples = os.path.join(self.task.dir_explainability,
                                           model_id,
                                           tvt)
                sample_ids = sorted(os.listdir(dir_samples))
                for sample_id in sample_ids:
                    # Needs to be checked since it will be looped over
                    # multiple times, once per version.
                    if dict_shell_plots[tvt].get(sample_id) is None:
                        dict_shell_plots[tvt][sample_id] = {}
                        # Figures out how many lesions there are of the current
                        # patient.
                        n_lesions = len([si for si in sample_ids if
                                         ("_".join(
                                             sample_id.split("_")[:-1]) in si)])
                        dict_shell_plots[tvt][sample_id][
                            "n_lesions"] = n_lesions

                    file_grad_cam = os.path.join(dir_samples,
                                                 sample_id,
                                                 sample_id + "_grad_cam.nii")
                    file_shells = os.path.join(dir_samples,
                                               sample_id,
                                               sample_id + "_shells.nii")

                    arr_grad_cam = \
                        voxelizer.VoxelArray(source_voxel=file_grad_cam).array[
                            ..., 0]
                    arr_shells = \
                        voxelizer.VoxelArray(source_voxel=file_shells).array[
                            ..., 0]

                    assert arr_grad_cam.shape == arr_shells.shape, \
                        ("The Grad-CAM, centers and segmentation arrays do not "
                         "have the same dimensions!")

                    # To figure out how much activation was inside the tumor.
                    sum_grad_total = np.sum(arr_grad_cam[arr_shells > 0])
                    # Avoids errors when displaying bar plots for volumes
                    # without any TUMOR activation.
                    if sum_grad_total == 0:
                        warnings.warn("Total tumor activation of 0 detected!")
                        sum_grad_total = 1e-10

                    shell_labels = np.sort(
                        np.unique(arr_shells[arr_shells > 0]))
                    rel_activation_per_shell = []
                    for shell_label in shell_labels:
                        # To set the activation of a distance percentile into
                        # relation.
                        sum_grad_shell = np.sum(
                            arr_grad_cam[arr_shells == shell_label])
                        # Always needs to be relative!
                        rel_activation_per_shell += [
                            sum_grad_shell / sum_grad_total]

                    dict_shell_plots[tvt][sample_id][
                        version] = rel_activation_per_shell

        # Plot creation and exporting.
        dir_export = os.path.join(self.task.dir_explainability,
                                  "grad_cam_shell_plots")
        os.makedirs(dir_export, exist_ok=True)

        for tvt in dict_shell_plots.keys():
            for sample_id in dict_shell_plots[tvt].keys():
                n_lesions = dict_shell_plots[tvt][sample_id]["n_lesions"]
                y_series = [dict_shell_plots[tvt][sample_id][version] for
                            version in self.version_names]
                names = self.version_names
                colors = self.ORDERED_COLORS[:len(names)]

                # Split by Luminal A or B.
                label = str(self.task.get_patient_label(
                    patient_id="_".join(sample_id.split("_")[:-1]),
                    tvt=tvt)[0])
                file_export = os.path.join(dir_export,
                                           tvt,
                                           label,
                                           sample_id + ".png")
                os.makedirs(os.path.dirname(file_export), exist_ok=True)

                self._bar_plot(y_series=y_series,
                               names=names,
                               colors=colors,
                               sample_id=sample_id,
                               n_lesions=n_lesions,
                               file_export=file_export)

        # Also export the plots dictionary!
        file_export_dict = os.path.join(dir_export,
                                        "shell_plots.json")
        with open(file_export_dict, "w") as f:
            json.dump(dict_shell_plots, f, indent=4)

        print(F"Exported files to {dir_export}.")

        return

    @staticmethod
    def _bar_plot(y_series: list,
                  names: list,
                  colors: list,
                  sample_id: str,
                  n_lesions: int,
                  file_export: str):
        """
        Helper function for plotting bar charts using matplotlib.
        """
        # Ensures that bins always have the same distance from one another.
        pad_bins = 0.2

        assert len(y_series) == len(names), ("The same number of y-series and "
                                             "names must be identical!")

        n_bins = len(y_series[0])
        n_bars = len(y_series)
        x_vals = np.arange(n_bins)
        x_ticklabels = np.round(1 / n_bins * (x_vals + 1), 2)
        bar_width = (1 - pad_bins) / n_bars

        # Figuring out the starting positions for the bars such that bars of
        # the same bin are immediately neighboring.
        n_bars_per_bin = len(y_series)
        start_pos_bar = -(n_bars_per_bin - 1) * bar_width / 2
        bar_idx = 0

        fig, ax = plt.subplots(1, figsize=(10, 5))
        for y_vals, name, color in zip(y_series, names, colors):
            x_vals_bar = x_vals + start_pos_bar + bar_idx * bar_width
            ax.bar(x_vals_bar, y_vals,
                   width=bar_width,
                   align='center',
                   color=color,
                   label=name)

            bar_idx += 1

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_ticklabels)
        ax.set_xlabel("Tumor shells, from center to periphery")
        ax.set_ylabel("Activation")
        # Important for conformity.
        ax.set_ylim([0, 1])

        patient_id = sample_id.split("TUE018_0")[1].split("_")[0]
        # Remove leading 0.
        patient_id = patient_id if patient_id[0] != "0" else patient_id[1:]
        # In case there are multiple lesions per patient, change the title
        # accordingly
        if n_lesions > 1:
            lesion_id = int(sample_id[-1]) + 1
            ax.set_title(F"Patient-ID {patient_id} - Lesion no. {lesion_id}")
        else:
            ax.set_title(F"Patient-ID {patient_id}")
        ax.legend()

        # Either export or show a plot.
        if file_export is not None:
            fig.savefig(file_export, bbox_inches="tight", pad_inches=0.1,
                        dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return
