import os
import json

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import resnet_2d
import dataset_2d
import dataset_3d
import evaluation


class HyperEnsembleModel(nn.Module):

    def __init__(self, *ensemble_models):
        super(HyperEnsembleModel).__init__()

        assert len(ensemble_models) > 0, "No ensemble models provided!"
        assert np.all([isinstance(ensemble_model, EnsembleModel) for \
                       ensemble_model in
                       ensemble_models]), "Models need to be ensemble models!"
        assert np.all([ensemble_model.n_channels_input == ensemble_models[
            0].n_channels_input for ensemble_model in ensemble_models]), \
            "Models need to be all accept the same number of image channels!"
        self.ensemble_models = ensemble_models

        return

    def predict(self,
                dir_image: str):
        """
        NOTE: Since the hyper ensembles might consist of 2D and 3D models at
        once, no tensors can be provided as input, but rather the directory
        containing the images such that they may be loaded according to the
        different requirements of the models.
        """
        # TODO: Continue here.
        pass


class EnsembleModel(nn.Module):
    RESNET_VERSIONS = [18, 34, 50, 101, 152]

    def __init__(self,
                 file_ensemble: str,
                 file_task: str,
                 transform_2d,
                 transform_3d):
        super(EnsembleModel, self).__init__()

        with open(file_ensemble, "r") as f:
            dict_config = json.load(f)

        # Legacy compatibility with config files without the n_channels_input
        # and n_classes_per_label keys.
        if ("n_channels_input" not in dict_config.keys()) | (
                "n_classes_per_label"
                not
                in
                dict_config.keys()):
            assert file_task is not None, ("In case the ensemble config "
                                           "file is incomplete, a dataset "
                                           "file must be provided!")
            with open(file_task, "r") as f:
                dict_dataset = json.load(f)
            n_channels_input = len(dict_dataset["modality"].keys())
            n_classes_per_label = [len(dict_dataset["labels"][key_label][
                                           "classes"].keys()) for key_label in
                                   dict_dataset["labels"].keys()]
            dict_config["n_channels_input"] = n_channels_input
            dict_config["n_classes_per_label"] = n_classes_per_label

        dir_models = os.path.join(os.path.dirname(file_ensemble), "folds")
        self.model_instances, self.n_channels_input, self.n_classes_per_label, self.dims \
            = self._initialize_models(
            dir_weights=dir_models,
            dict_hyperparams=dict_config)

        self.transform = transform_2d if self.dims == 2 else transform_3d
        self.ensemble_id = dict_config["id"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")

        return

    def forward(self, x):
        """
        NOTE: This method expects BCE as loss!!
        """
        probs_per_model = []
        for model_instance in self.model_instances:
            # n_images x n_labels x n_classes
            outputs = model_instance(x)
            # SUPER IMPORTANT IN ORDER TO GAIN PROBABILITIES INSTEAD OF
            # SIMPLY REAL NUMBERS!
            probs = F.softmax(outputs, dim=2)
            probs_per_model += [probs]

        # n_models x n_images x n_labels x n_classes
        return torch.stack(probs_per_model, dim=0)

    def predict(self,
                file_task: str,
                t: str,
                axial_only: bool,
                dir_export: str,
                ablation_channel_idxs: list):
        """
        Perform the actual prediction and export the results as 2 JSON files
        per ensemble.
        """
        # This is done in order to enable the use of task files containing
        # just the test samples of another task.
        name_task = os.path.basename(file_task.split(".")[0])

        if self.dims == 2:
            dataset = dataset_2d.VolumeDataset2D(file_task=file_task,
                                                 image_format="images",
                                                 tvt=t,
                                                 transform=self.transform,
                                                 axial_only=axial_only,
                                                 triplets_name_file_labels=None,
                                                 ablation_channel_idxs=ablation_channel_idxs)
        else:
            dataset = dataset_3d.VolumeDataset3D(
                file_task=file_task,
                image_format="images",
                tvt=t,
                transform=self.transform,
                triplets_name_file_labels=None,
                ablation_channel_idxs=ablation_channel_idxs)

        dataloader = dataset_2d.DataLoaderTest(dataset=dataset)
        list_mblc = []
        names = []
        with torch.no_grad():
            for input, labels, name in tqdm(dataloader, desc="Ensemble "
                                                             "prediction"):
                input, labels = input.to(
                    self.device), labels.cpu().numpy().tolist()
                # Probabilistic prediction.
                # n_models x n_batch x n_labels x n_classes
                mblc = self.forward(input)

                list_mblc += [mblc]
                names += name

        # n_models x n_images x n_labels x n_classes
        milc = torch.cat(list_mblc, dim=1).cpu().numpy()
        # n_models x n_patients x n_labels x n_classes
        mplc, names_patient = EnsembleModel._compress_by_patient(milc=milc,
                                                                 names=names)

        # Give a unique name to all slides of a tumor, for the slide-based
        # statistics.
        names_image = []
        name_id = 0
        name_last = ""
        for name in names:
            if name == name_last:
                name_id += 1
            else:
                name_id = 0
            # Support for up to 10,000 slides per tumor!
            name_image_or_patient = name + "_" + (
                "000" + str(name_id) if name_id < 10 else "00" + str(
                    name_id) if name_id < 100 else "0" + str(name_id) if
                name_id < 1000 else str(name_id))
            names_image += [name_image_or_patient]
            name_last = name

        voting_method_names = ["majority", "mean"]
        voting_method_funcs = [EnsembleModel.vote_majority,
                               EnsembleModel.vote_mean]

        # Shape of dict: {name: "", label:{id: "", probabilities: [],
        # prediction: {
        # voting_method: 0}}}
        for i in range(2):
            if i == 0:
                names_image_or_patient = names_image
                sample_method = "image"
                milc_or_mplc = milc
            else:
                names_image_or_patient = names_patient
                sample_method = "patient"
                milc_or_mplc = mplc

            labels_gt_per_image_or_patient = (
                EnsembleModel._create_labels_gt_per_image_or_patient(
                    dataset=dataset,
                    names_image_or_patient=names_image_or_patient))
            dict_total = {"ensemble_id": self.ensemble_id,
                          "task": name_task,
                          "n_labels": milc.shape[2],
                          "n_classes_per_label": milc.shape[3]}

            list_image_or_patient = []

            for ip_idx, name_image_or_patient in enumerate(
                    names_image_or_patient):
                dict_image_or_patient = {"sample": name_image_or_patient,
                                         "labels_gt":
                                             labels_gt_per_image_or_patient[
                                                 name_image_or_patient],
                                         "labels_pred": {}}
                for label_idx in range(milc_or_mplc.shape[2]):
                    dict_class_prob = {
                        str(model_idx): np.round(milc_or_mplc[model_idx, ip_idx,
                        label_idx], 4).tolist() for model_idx in
                        range(milc_or_mplc.shape[0])}
                    dict_labels = {"probs_per_fold": dict_class_prob}

                    dict_voting_method = {}
                    # Redundant to call the voting method here, but more readable.
                    for voting_method_name, voting_method_func in zip(
                            voting_method_names, voting_method_funcs):
                        ipl = voting_method_func(mplc=milc_or_mplc)

                        dict_voting_method[voting_method_name] = int(ipl[ip_idx,
                        label_idx])

                    dict_labels["predictions"] = dict_voting_method
                    dict_image_or_patient["labels_pred"][
                        str(label_idx)] = dict_labels

                    list_image_or_patient += [dict_image_or_patient]

            dict_total["samples"] = list_image_or_patient

            if not os.path.isdir(dir_export):
                os.makedirs(dir_export)
            file_export = os.path.join(dir_export, F"prediction_"
                                                   F"{self.ensemble_id}_"
                                                   F"{name_task}_"
                                                   F"{sample_method}.json")

            with open(file_export, "w") as f:
                json.dump(dict_total, f, indent=4)

        return

    def predict_evaluate(self,
                         file_task: str,
                         dir_prediction: str,
                         dir_evaluation: str,
                         t: str = "test",
                         axial_only: bool = True):
        """
        Wrapper function to bundle the prediction and evaluation
        functionality in one place.
        """
        self.predict(file_task=file_task,
                     t=t,
                     axial_only=axial_only,
                     dir_export=dir_prediction,
                     ablation_channel_idxs=None)
        # Will return just one file in a list.
        file_eval = evaluation.EnsembleEvaluation.evaluate(
            dir_predictions=dir_prediction,
            dir_export=dir_evaluation)[0]
        df_performance_per_ensemble = (
            evaluation.EnsembleEvaluation.tabularize_statistics(
                file_eval=file_eval,
                ensemble_id=None))
        for ensemble_id in df_performance_per_ensemble.keys():
            print(ensemble_id)
            print(df_performance_per_ensemble[ensemble_id].to_string())

        return file_eval, df_performance_per_ensemble

    @staticmethod
    def _compress_by_patient(milc: np.ndarray,
                             names: list):
        """
        This subfunctionality needs to be outsourced in order for the hyper
        ensemble model to be able to majority voting across every single model.
        IMPORTANT: Only allows for mean vote on an intra-patient level!!
        """

        assert milc.ndim == 4, ("Incompatible shape of the input array for "
                                "image compression by patient!")

        column_name = ["name"]
        columns_class = [F"class_{i}" for i in range(milc.shape[3])]

        # n_models x n_labels  x n_patients x n_classes
        mlpc = []
        names_compressed = None

        for model_idx in range(milc.shape[0]):
            lpc = []
            for label_idx in range(milc.shape[2]):
                # n_images x n_classes
                ic = milc[model_idx, :, label_idx, :]
                df_ic = pd.DataFrame(names, columns=column_name)
                df_ic[columns_class] = ic
                df_pc = df_ic.groupby(column_name)[
                    columns_class].mean().reset_index()
                names_compressed = df_pc[column_name].to_numpy(
                ).flatten().tolist() if (
                        names_compressed is None) else names_compressed
                pc = df_pc[columns_class].to_numpy()
                lpc += [pc]
            mlpc += [np.array(lpc)]
        # n_models x n_labels  x n_patients x n_classes
        mplc = np.einsum("mlpc->mplc", np.array(mlpc))

        assert np.allclose(np.sum(mplc, axis=3), 1), ("Matrix transformation "
                                                      "failed.")

        return mplc, names_compressed

    @staticmethod
    def _create_labels_gt_per_image_or_patient(dataset,
                                               names_image_or_patient: list):
        labels_gt_per_image_or_patient = {}

        names_dataset = [triple[0] for triple in
                         dataset.triplets_name_file_labels]
        for name_image_or_patient in names_image_or_patient:
            # IMPORTANT: A single patient can have multiple files associated
            # with them if the tumor is multi-focal. In that case, just pick
            # the first corresponding file in the dataset.
            triple_idx = [i for i, name_dataset in enumerate(names_dataset) if
                          name_dataset in name_image_or_patient]
            assert len(triple_idx) > 0, (F"Failed identifying the correct "
                                         F"label for sample "
                                         F"{name_image_or_patient}.")
            labels_gt_per_image_or_patient[name_image_or_patient] = {str(
                idx): label for idx, label in enumerate(
                dataset.triplets_name_file_labels[triple_idx[0]][2])}

        return labels_gt_per_image_or_patient

    @staticmethod
    def vote_majority(mplc: np.ndarray):
        mpl_majority = np.argmax(mplc, axis=3)
        pl_majority = stats.mode(mpl_majority, axis=0).mode

        return pl_majority

    @staticmethod
    def vote_mean(mplc: np.ndarray):
        plc_mean = np.mean(mplc, axis=0)
        pl_mean = np.argmax(plc_mean, axis=2)

        return pl_mean

    @staticmethod
    def class_probabilities(mplc: np.ndarray):
        plc_prob = np.mean(mplc, axis=0)

        return plc_prob

    @staticmethod
    def _initialize_models(dir_weights: str,
                           dict_hyperparams: dict):
        files_weights = sorted([os.path.join(dir_weights, file) for file in
                                os.listdir(dir_weights) if (
                                        ".pth" in file.lower())])

        list_model_instance = []
        list_n_channels_input = []
        list_n_classes_per_label = []
        list_dims = []

        for file_weights in files_weights:
            (model_instance, n_channels_input, n_classes_per_label, dims) = (
                EnsembleModel.load_model(
                    file_weights=file_weights,
                    dict_hyperparams=dict_hyperparams))

            list_model_instance += [model_instance]
            list_n_channels_input += [dims]
            list_n_classes_per_label += [n_classes_per_label]
            list_dims += [dims]

        assert np.all(np.array(list_n_channels_input) == list_n_channels_input[
            0]), "Models do not all either work on 2D or 3D data!"
        assert np.all(np.array(list_n_channels_input) == list_n_channels_input[
            0]), "Models do not all have the same number of input channels!"
        assert np.all(
            np.array(list_n_classes_per_label) == list_n_classes_per_label[
                0]), "Models do not all have the same number of labels!"
        assert np.all(
            np.array(list_dims) == list_dims[
                0]), ("Models do not all have the same number of input "
                      "dimensions!")

        return (list_model_instance, list_n_channels_input[0],
                list_n_classes_per_label[0], list_dims[0])

    @staticmethod
    def load_model(file_weights: str,
                   dict_hyperparams: dict):
        """
        THE function to handle the loading of different model types.
        Update this to support more model architectures!
        """
        architecture = dict_hyperparams["model"].lower()

        if "resnet" in architecture:
            version = int(architecture.split("resnet")[1].split(", ")[0])
            assert version in EnsembleModel.RESNET_VERSIONS, ("Invalid ResNet "
                                                              "model version.")
            n_classes_per_label = dict_hyperparams["n_classes_per_label"]
            n_channels_input = len(dict_hyperparams["included_modalities"])
            dims = int(dict_hyperparams["model"].split(", ")[1].split("D")[0])

            model_instance = resnet_2d.ResnetFactory(
                n_channels_input=n_channels_input,
                n_classes_per_label=n_classes_per_label,
                pretrained=False,
                dims=dims,
                version=version).create_model()
            model_instance.load_weights(file_weights=file_weights,
                                        eval=True)
        else:
            raise ValueError("Currently only the ResNet architecture is "
                             "supported!")

        return model_instance, n_channels_input, n_classes_per_label, dims


class EnsembleModelInference(nn.Module):
    """
    Helper function to only perform discrete classification.
    """

    def __init__(self,
                 file_ensemble: str,
                 file_task: str,
                 transform_2d,
                 transform_3d):
        super(EnsembleModelInference, self).__init__()
        self.ensemble_model = EnsembleModel(file_ensemble=file_ensemble,
                                            file_task=file_task,
                                            transform_2d=transform_2d,
                                            transform_3d=transform_3d)

        return

    def forward(self, x):
        # n_models x n_images x n_labels x n_classes
        probs = self.ensemble_model(x)

        # n_images x n_labels x n_classes
        mean_probs = torch.mean(probs, dim=0)
        # n_images x n_labels
        # Needs to be turned into a float for SHAP analysis.
        preds = torch.argmax(mean_probs, dim=2).type(torch.float)

        return preds
