import sys
import os

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import cnn_images_2D_3D
import omics

import data_processing.voxelizer as voxelizer

from resnet_subtype.task import Task


class TaskPreprocess:
    """
    Class that acts as a wrapper by which to access all functionality for
    preprocessing and training for a specified train/test data split,
    modalities, and labels.
    """
    TVT = ["train", "val", "test"]
    KEYS_CONFIG = ["name", "modality", "train", "val", "test", "labels",
                   "dirBase", "dirPreprocessed",
                   "rthreshold_component", "percentile_lower_clip",
                   "percentile_upper_clip", ]
    KEYS_TRAIN_VAL_TEST = ["image", "maskROI", "maskForeground", "labels"]

    ORDERED_MODALITIES = ["T1_POST_DYNAMIC_0", "T1_POST_DYNAMIC_1",
                          "T1_POST_DYNAMIC_2", "T1_POST_DYNAMIC_3",
                          "T1_POST_DYNAMIC_4", "T1_POST_DYNAMIC_5",
                          "T1_POST_DYNAMIC_6",
                          "STIR",
                          "DWI_0", "DWI_1",
                          "ADC",
                          "PET"]

    MODALITIES_GLOBAL = ["ADC", "PET"]
    MODALITIES_DYNAMIC = ["T1_POST_DYNAMIC_0", "T1_POST_DYNAMIC_1",
                          "T1_POST_DYNAMIC_2", "T1_POST_DYNAMIC_3",
                          "T1_POST_DYNAMIC_4", "T1_POST_DYNAMIC_5",
                          "T1_POST_DYNAMIC_6"]
    MODALITIES_Z_SCORE = ["STIR", "DWI_0", "DWI_1"]

    @staticmethod
    def _verify_config(dict_config: dict):
        for key in TaskPreprocess.KEYS_CONFIG:
            assert key in dict_config.keys(), (F"{key} is not a key in the "
                                               F"config file!")
        assert len(dict_config[
                       "train"]) > 0, "At least 1 train image must be provided!"

        n_labels = len(dict_config["labels"].keys())
        assert n_labels > 0, "At least 1 label must be provided!"

        for t in TaskPreprocess.TVT:
            for dict_img_label in dict_config[t]:
                for key in TaskPreprocess.KEYS_TRAIN_VAL_TEST:
                    assert key in dict_img_label.keys(), (
                        F"{key} is not a key for all the {t} files!")
                assert len(dict_img_label["labels"]) == n_labels, \
                    F"Not all {t} files have {n_labels} labels!"

        return

    def __init__(self,
                 file_task: str,
                 tumor_roi_id: int = 1):
        with open(file_task, 'r') as f:
            self.dict_task = json.load(f)

        # Ensure the correctness of the config file.
        TaskPreprocess._verify_config(self.dict_task)

        self.name = self.dict_task["name"]
        self.modalities = [self.dict_task["modality"][key] for key in
                           sorted(self.dict_task["modality"].keys())]

        self.dir_base = os.path.expanduser(self.dict_task["dirBase"])
        self.dir_preprocessed = os.path.expanduser(self.dict_task[
                                                       "dirPreprocessed"])

        self.dir_radiomics = os.path.join(self.dir_preprocessed,
                                          self.name,
                                          "radiomics")

        # IMPORTANT: 1 is the primary tumor ROI!
        self.tumor_roi_id = tumor_roi_id

        return

    def preprocess_neural_net(self):
        self._global_normalization(neural_net=True)
        self._export_volumes(neural_net=True,
                             images=True)
        self._export_volumes(neural_net=True,
                             images=False)
        self._copy_config_file()

        return

    def preprocess_radiomics(self):
        self._global_normalization(neural_net=False)
        self._export_volumes(neural_net=False,
                             images=True)
        self.extract_radiomics_from_nifti()

        return

    def preprocess_bbox_masks(self):
        self._export_volumes(neural_net=True,
                             images=False)

        return

    def _global_normalization(self,
                              neural_net: bool):
        # NOTE: The key in the config dictionary is called modality (
        # singular) in order to stick with nnU-Net naming conventions,
        # even though it is counter-intuitive.
        modalities_global = [modality for modality in self.modalities if
                             modality in
                             TaskPreprocess.MODALITIES_GLOBAL]

        # Only normalize on the train samples!!
        train = self.dict_task["train"]

        percentile_lower_clip = self.dict_task["percentile_lower_clip"]
        percentile_upper_clip = self.dict_task["percentile_upper_clip"]

        dir_base = os.path.expanduser(self.dict_task["dirBase"])

        dirs_image = [os.path.join(dir_base, img["image"]) for img in train]
        files_mask_foreground = [os.path.join(dir_base, img["maskForeground"])
                                 for img in train]

        cnn_images_2D_3D.GlobalNormalizer.compute_mean_std_bounds(
            dirs_image=dirs_image,
            files_mask_foreground=files_mask_foreground,
            modalities_global=modalities_global,
            percentile_lower_clip=percentile_lower_clip,
            percentile_upper_clip=percentile_upper_clip,
            file_export=self._get_file_normalization(neural_net=neural_net))

        return

    def _get_file_normalization(self,
                                neural_net: bool):
        file_normalization = "normalization_" + self.name + ".csv"
        if neural_net:
            file_normalization = os.path.join(self.dir_preprocessed,
                                              self.name,
                                              file_normalization)
        else:
            file_normalization = os.path.join(self.dir_radiomics,
                                              file_normalization)

        return file_normalization

    def _export_volumes(self,
                        neural_net: bool,
                        images: bool):

        dir_base = os.path.expanduser(self.dict_task["dirBase"])
        rthreshold_component = self.dict_task["rthreshold_component"]

        for tvt in TaskPreprocess.TVT:
            dir_stub = os.path.expanduser(os.path.join(
                self.dict_task["dirPreprocessed"],
                self.name))

            if neural_net:
                dir_export_images = os.path.join(dir_stub, "images", tvt)
            else:
                dir_export_images = os.path.join(dir_stub, "whole_body", tvt)

            dir_export_masks = os.path.join(dir_stub, "masks", tvt)

            # Export either the bbox images or bbox binary masks.
            dir_export = dir_export_images if images else dir_export_masks

            for dict_tvt in self.dict_task[tvt]:
                # Handling for when no mask ROI file was provided or is not
                # required, in the case of radiomics.
                file_mask_roi = dict_tvt.get("maskROI") if neural_net else None
                if file_mask_roi is not None:
                    # Handled this way because the user ALWAYS needs to
                    # provide a maskROI key in their JSON config file.
                    if len(file_mask_roi) == 0:
                        file_mask_roi = None
                    else:
                        file_mask_roi = os.path.join(dir_base, file_mask_roi)

                img_to_bbox = (cnn_images_2D_3D
                .ImageToBBox(
                    dir_image=os.path.join(dir_base, dict_tvt["image"]),
                    modalities=self.modalities,
                    modalities_dynamic=TaskPreprocess.MODALITIES_DYNAMIC,
                    modalities_z_score=TaskPreprocess.MODALITIES_Z_SCORE,
                    file_mask_roi=file_mask_roi,
                    file_mask_foreground=
                    os.path.join(dir_base, dict_tvt["maskForeground"]),
                    file_normalization=self._get_file_normalization(
                        neural_net=neural_net),
                    rthreshold_component=rthreshold_component))
                img_to_bbox.create_connected_components(
                    include_voxels_outside_mask=True)
                img_to_bbox.normalize_global(clip=True)
                img_to_bbox.normalize_dynamic()
                img_to_bbox.normalize_z_score()

                os.makedirs(dir_export,
                            exist_ok=True)
                img_to_bbox.export_volumes(dir_export=dir_export,
                                           format="nii",
                                           exporting_images=images)

        return

    def _copy_config_file(self):
        """
        Make sure there is always a copy of the configuration of a task in
        that task's directory.
        """
        file_config_copy = os.path.expanduser(os.path.join(self.dict_task[
                                                               "dirPreprocessed"],
                                                           self.name,
                                                           "config.json"))
        with open(file_config_copy, "w") as f:
            json.dump(self.dict_task, f, indent=4)

        return

    def extract_radiomics_from_nifti(self):
        """
        Using the normalized whole-body images to extract radiomic features.
        """
        for t in TaskPreprocess.TVT:
            dict_features = {"patient": []}
            list_labels = []

            for patient in self.dict_task[t]:
                patient_name = os.path.basename(patient["image"])
                dict_features["patient"] += [patient_name]
                list_labels += [patient["labels"]]

                print("Computing features for patient", patient_name, ".")

                file_volume = os.path.join(self.dir_radiomics, "whole_body", t,
                                           patient_name + "_0000.nii")
                file_mask = os.path.join(self.dir_base, patient[
                    "maskROI"])

                arr_volume = voxelizer.VoxelArray(
                    source_voxel=file_volume).array
                arr_mask = voxelizer.VoxelArray(source_voxel=file_mask).array[
                    ..., 0]

                for modality_idx, modality in enumerate(self.modalities):
                    X = arr_volume[..., modality_idx][
                        arr_mask == self.tumor_roi_id]
                    idxs_X = np.array(np.where(arr_mask == self.tumor_roi_id)).T
                    if "pet" in modality.lower():
                        radiomics = omics.RadiomicsPET(X=X,
                                                       idxs_X=idxs_X,
                                                       volume=arr_volume[
                                                           ..., modality_idx])
                    else:
                        radiomics = omics.RadiomicsMRI(X=X)
                    features = radiomics.get_features()
                    feature_names = [F"{feature_name}_{modality.lower()}" for
                                     feature_name in
                                     radiomics.get_feature_names()]

                    for feature, feature_name in zip(features, feature_names):
                        if dict_features.get(feature_name) is None:
                            dict_features[feature_name] = [feature]
                        else:
                            dict_features[feature_name] += [feature]

            # Exporting the radiomics features.
            df_features = pd.DataFrame(dict_features)
            file_features = os.path.join(self.dir_radiomics, F"radiomics_"
                                                             F"{self.name}"
                                                             F"_{t}.csv")
            os.makedirs(self.dir_radiomics, exist_ok=True)
            df_features.to_csv(file_features, index=False)

            # Exporting the labels separately.
            arr_labels = np.array(list_labels)
            if arr_labels.shape[0] > 0:
                columns_labels = [F"label_{i}" for i in
                                  range(arr_labels.shape[1])]
            else:
                # Option for cross-validation.
                columns_labels = []
            df_labels = pd.DataFrame(df_features["patient"],
                                     columns=["patient"])
            df_labels[columns_labels] = arr_labels
            file_labels = os.path.join(self.dir_radiomics, F"labels_"
                                                           F"{self.name}"
                                                           F"_{t}.csv")
            df_labels.to_csv(file_labels, index=False)

        return

    def extract_radiomics_from_patches(self):
        """
        Extracting radiomic features from patch files, to train tree-models
        as a baseline against the ResNet patches models.
        """
        # Only train makes sense, since there is no dedicated validation
        # dataset!
        for t in ["train"]:
            for dims in [2, 3]:
                dict_features = {"patient": []}
                list_labels = []

                for patient in self.dict_task[t]:
                    patient_name = os.path.basename(patient["image"])

                    dir_patches = os.path.join(self.dir_preprocessed,
                                               self.name,
                                               t,
                                               "patches",
                                               F"{dims}D",
                                               patient_name)

                    files_patches = sorted([os.path.join(dir_patches, file) for \
                                            file in
                                            os.listdir(dir_patches) if
                                            "labels" not in file])
                    file_labels = os.path.join(dir_patches, "labels.npy")
                    arr_labels = np.load(file_labels)
                    list_labels += arr_labels.tolist()

                    pbar = tqdm(total=len(files_patches),
                                desc=F"{dims}D features from patches of "
                                     F"patient {patient_name}")
                    for patch_idx, file_patches in enumerate(files_patches):
                        # Patient name has to be added for each patch!!
                        dict_features["patient"] += [patient_name]

                        arr_patch = voxelizer.VoxelArray(
                            source_voxel=file_patches).array
                        if (((dims == 2) & (arr_patch.ndim == 4))
                                | ((dims == 3) & (arr_patch.ndim == 5))):
                            assert arr_patch.shape[-1] == 1, (
                                F"Something went very wrong with the {dims}D "
                                F"patches!")
                            arr_patch = arr_patch[..., 0]
                        # Compatibility with 3D indexing.
                        if (dims == 2) & (arr_patch.ndim == 3):
                            arr_patch = arr_patch[None, ...]
                        # For the case of extracting radiomic features from
                        # patches, assume every voxel to be in the foreground.
                        idxs_X = np.array(
                            np.where(np.ones(arr_patch.shape[:3]))).T

                        for modality_idx, modality in enumerate(
                                self.modalities):
                            X = arr_patch[..., modality_idx].flatten()
                            if "pet" in modality.lower():
                                radiomics = omics.RadiomicsPET(X=X,
                                                               idxs_X=idxs_X,
                                                               volume=arr_patch[
                                                                   ..., modality_idx])
                            else:
                                radiomics = omics.RadiomicsMRI(X=X)
                            features = radiomics.get_features(verbose=False)
                            feature_names = [
                                F"{feature_name}_{modality.lower()}"
                                for
                                feature_name in
                                radiomics.get_feature_names()]

                            for feature, feature_name in zip(features,
                                                             feature_names):
                                if dict_features.get(feature_name) is None:
                                    dict_features[feature_name] = [feature]
                                else:
                                    dict_features[feature_name] += [feature]
                        pbar.update(1)
                # Exporting the radiomics features and labels separately.
                df_features = pd.DataFrame(dict_features)
                df_labels = pd.DataFrame(df_features["patient"],
                                         columns=["patient"])
                df_labels["label"] = list_labels
                assert df_features.shape[0] == df_labels.shape[0], \
                    ("The feature and label dataframes don't have the same "
                     "length!")

                os.makedirs(self.dir_radiomics, exist_ok=True)
                file_features = os.path.join(self.dir_radiomics,
                                             F"radiomics"
                                             F"_{dims}D_{t}.csv")
                file_labels = os.path.join(self.dir_radiomics,
                                           F"labels"
                                           F"_{dims}D_{t}.csv")
                df_features.to_csv(file_features, index=False)
                df_labels.to_csv(file_labels, index=False)

        return

    @staticmethod
    def remove_modalities(file_task: str,
                          modalities: list):
        """
        Helper function to remove all features of a specified modality for
        the ablation studies.
        """
        task = Task(file_task=file_task)
        files_radiomics = os.listdir(task.dir_radiomics_input)

        for file_radiomics in files_radiomics:
            file_radiomics = os.path.join(task.dir_radiomics_input,
                                          file_radiomics)
            df_radiomics = pd.read_csv(file_radiomics)
            n_columns_before = df_radiomics.shape[1]
            all_matched_columns = []

            for modality in modalities:
                matched_columns = [column for column in df_radiomics.columns
                                   if modality.lower() in column.lower()]
                df_radiomics = df_radiomics.drop(matched_columns, axis=1)
                all_matched_columns += matched_columns

            assert df_radiomics.shape[1] == n_columns_before - len(
                all_matched_columns), (F"The removal of columns from file "
                                       F"{file_radiomics} failed.")
            df_radiomics.to_csv(file_radiomics, index=False)

            print(
                F"Removed a total of {len(all_matched_columns)} columns from "
                F"file {file_radiomics}:")
            print("\n".join(all_matched_columns))

        return

    @staticmethod
    def salient_features_only(file_task: str,
                              salient_features: list):
        """
        Helper function to limit the input data to only the salient features
        from SHAP analysis.
        """
        ID_COLUMN = "patient"

        task = Task(file_task=file_task)
        files_radiomics = os.listdir(task.dir_radiomics_input)

        for file_radiomics in files_radiomics:
            file_radiomics = os.path.join(task.dir_radiomics_input,
                                          file_radiomics)
            df_radiomics = pd.read_csv(file_radiomics)
            n_columns_before = df_radiomics.shape[1]

            if not set(df_radiomics.columns).issuperset(salient_features):
                if ("labels_" in file_radiomics) | (
                        "normalization_" in file_radiomics):
                    warnings.warn(
                        F"File {file_radiomics} does not contain all salient "
                        F"features, but is assumed to be a "
                        F"labels/normalization file. Skipping.")
                    continue
                else:
                    raise ValueError("File incompatible with salient feature "
                                     "extraction.")

            df_radiomics = df_radiomics[[ID_COLUMN] + salient_features]
            n_removed_columns = n_columns_before - df_radiomics.shape[1]

            df_radiomics.to_csv(file_radiomics, index=False)
            print(
                F"Removed a total of {n_removed_columns} columns from "
                F"file {file_radiomics}:")

            # Sanity check.
            df_radiomics_reload = pd.read_csv(file_radiomics)
            # The first column is the ID column, therefore ignore it.
            assert len(set(df_radiomics_reload.columns[1:]).difference(
                salient_features)) == 0, ("The removal of non-salient features "
                                          "failed.")

        return
