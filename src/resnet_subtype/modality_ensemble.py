import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sklearn.metrics as skm
import sklearn.linear_model as sklm

from tqdm import tqdm
import itertools
import warnings

import task
import ensemble
import dataset_2d
import dataset_3d
import linear_solver


class SingleModel:
    """
    Prediction of a single model, trained on only a subset of all available
    imaging modalities.
    """
    ID = "id"
    CLASS = "class"
    LABEL = "label"

    def __init__(self,
                 model_id: str,
                 dir_base: str,
                 transform_2D,
                 transform_3D):
        dir_weights = os.path.join(dir_base, model_id, "models")
        assert len(os.listdir(dir_weights)) == 1, ("There can only be 1 weight "
                                                   "file in a directory for a "
                                                   "SingleModel instance!")
        file_weights = os.path.join(dir_weights, os.listdir(dir_weights)[0])
        with open(os.path.join(dir_base, model_id, "hyperparams.json"),
                  "rb") as f:
            dict_hyperparams = json.load(f)

        (self.model_instance, self.n_channels_input, self.n_classes_per_label,
         self.dims) = (
            ensemble.EnsembleModel.load_model(
                file_weights=file_weights,
                dict_hyperparams=dict_hyperparams))
        # Inelegant (it would be better if the model could just load a
        # transform file), but works for now.
        dims = int(dict_hyperparams["model"].split(", ")[1].split("D")[0])
        self.transform = transform_2D if (dims == 2) else transform_3D
        self.model_id = model_id

        return

    def predict(self,
                file_task: str,
                tvt: str,
                modality_idxs: list,
                axial_only: bool,
                verbose: bool = True):
        dataset = self._get_dataset(file_task=file_task,
                                    tvt=tvt,
                                    axial_only=axial_only,
                                    modality_idxs=modality_idxs)

        df_pred_image, df_pred_patient = self.predict_from_model_and_dataset(
            model=self.model_instance,
            dataset=dataset,
            desc="Single model prediction")

        if verbose:
            task = dataset.task
            print(
                F"The predictions per patient of model {self.model_id} on the"
                F" {tvt} dataset of task {task}, using modalities "
                F"{modality_idxs}:")
            print(df_pred_patient)

        return df_pred_image, df_pred_patient

    def predict_evaluate(self,
                         file_task: str,
                         tvt: str,
                         modality_idxs: list,
                         axial_only: bool):
        dataset = self._get_dataset(file_task=file_task,
                                    tvt=tvt,
                                    axial_only=axial_only,
                                    modality_idxs=modality_idxs)

        dict_summary = self.predict_evaluate_from_model_and_dataset(
            model=self.model_instance,
            dataset=dataset,
            desc=F"Predicting {tvt}")

        return dict_summary

    def _get_dataset(self,
                     file_task: str,
                     tvt: str,
                     axial_only: bool,
                     modality_idxs: list):
        """
        Helper function to create a dataset for the prediction.
        """
        if self.dims == 2:
            dataset = dataset_2d.VolumeDataset2D(file_task=file_task,
                                                 image_format="images",
                                                 tvt=tvt,
                                                 transform=self.transform,
                                                 axial_only=axial_only,
                                                 include_channel_idxs=modality_idxs,
                                                 triplets_name_file_labels=None,
                                                 ablation_channel_idxs=None)
        else:
            dataset = dataset_3d.VolumeDataset3D(
                file_task=file_task,
                image_format="images",
                tvt=tvt,
                transform=self.transform,
                include_channel_idxs=modality_idxs,
                triplets_name_file_labels=None,
                ablation_channel_idxs=None)

        return dataset

    @staticmethod
    def predict_from_model_and_dataset(model,
                                       dataset,
                                       desc=None):
        """
        This is to be used for validation evaluation during training as well!
        """
        model.eval()
        # Loads all test files into batches of size 1!
        dataloader = dataset_2d.DataLoaderTest(dataset=dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probs_per_name = {}
        label_per_name = {}

        y_name = []
        y_probs = []
        y_true = []
        with torch.no_grad():
            for input, label, name in tqdm(dataloader, desc=desc):
                input = input.to(device)
                output = model(input)
                label = label.cpu().numpy().flatten()[0]

                # SUPER IMPORTANT IN ORDER TO GAIN PROBABILITIES INSTEAD OF
                # SIMPLY REAL NUMBERS!
                # n_images x n_labels x n_classes
                lc = F.softmax(output, dim=2)
                # IMPORTANT: For our purposes, there will only ever be 1
                # label! Also, since the images are loaded in 1 by 1,
                # n_images will also always be 1.
                # n_classes
                c = lc[0, 0, :].cpu().numpy()

                # Remove the surrounding list.
                name = name[0]
                if probs_per_name.get(name) is None:
                    probs_per_name[name] = [c]
                else:
                    probs_per_name[name] += [c]
                if label_per_name.get(name) is None:
                    label_per_name[name] = label

                y_name += [name]
                y_probs += [c]
                y_true += [label]

        for name in probs_per_name.keys():
            # Create the final probabilities per patient by
            # calculating the mean over all images of a patient.
            # n_images x n_classes -> n_patients x n_classes
            probs_per_name[name] = np.mean(np.array(probs_per_name[
                                                        name]), axis=0)
            assert np.isclose(np.sum(probs_per_name[name]), 1), (
                "Something went wrong while calculating the mean "
                "probability out of all tumor slides per patient.")

        # The number of classes can be derived from the model outputs.
        y_probs = np.array(y_probs)
        n_classes = y_probs.shape[1]
        class_columns = [F"{SingleModel.CLASS}_{class_idx}" for class_idx in
                         range(n_classes)]

        # Adding the patient id and the label.
        df_pred_image = pd.DataFrame({SingleModel.ID: y_name,
                                      SingleModel.LABEL: y_true, })

        df_pred_image[SingleModel.LABEL+"_pred"]=np.argmax(y_probs, axis=1)
        df_pred_image[class_columns] = y_probs

        df_pred_patient = SingleModel.image_to_patient(
            df_pred_image=df_pred_image)

        return df_pred_image, df_pred_patient

    @staticmethod
    def predict_evaluate_from_model_and_dataset(model,
                                                dataset,
                                                desc=None):
        """
        Combines prediction and evaluation into one dictionary.
        """
        df_pred_image, df_pred_patient = (
            SingleModel.predict_from_model_and_dataset(
                model=model,
                dataset=dataset,
                desc=desc))

        dict_eval_image = SingleModel.evaluate_image_or_patient(
            df_pred=df_pred_image)
        dict_eval_patient = SingleModel.evaluate_image_or_patient(
            df_pred=df_pred_patient)

        dict_summary = {"patient": dict_eval_patient,
                        "image": dict_eval_image}

        return dict_summary

    @staticmethod
    def evaluate(y_probs: np.ndarray,
                 y_true: np.ndarray):
        dict_eval = {}

        if (np.min(y_probs) < 0) | (np.max(y_probs) > 1):
            warnings.warn("Invalid probability detected! Will be replaced "
                          "with 1.")
            y_probs[y_probs < 0] = 0
            y_probs[y_probs > 1] = 1

        dict_eval["bce"] = skm.log_loss(y_true=y_true,
                                        y_pred=y_probs)
        dict_eval["mse"] = skm.mean_squared_error(y_true=y_true,
                                                  y_pred=y_probs[..., 1])

        # ROC-AUC always expects the probability of the second class.
        dict_eval["roc_auc"] = skm.roc_auc_score(y_score=y_probs[..., 1],
                                                 y_true=y_true)
        dict_eval["f1"] = skm.f1_score(y_pred=np.argmax(y_probs, axis=1),
                                       y_true=y_true)

        # Expects binary labels.
        dict_eval["acc"] = skm.accuracy_score(y_pred=np.argmax(y_probs, axis=1),
                                              y_true=y_true)
        dict_eval["acc_balanced"] = skm.balanced_accuracy_score(
            y_pred=np.argmax(y_probs, axis=1),
            y_true=y_true)

        tn, fp, fn, tp = skm.confusion_matrix(y_pred=np.argmax(y_probs, axis=1),
                                              y_true=y_true).ravel()
        dict_eval["sensitivity"] = float(tp / (tp + fn))
        dict_eval["specificity"] = float(tn / (tn + fp))

        return dict_eval

    @staticmethod
    def evaluate_image_or_patient(df_pred: pd.DataFrame):
        assert SingleModel.LABEL in df_pred.columns, \
            F"Missing column {SingleModel.LABEL} in prediction dataframe!"
        class_columns = sorted([column for column in df_pred.columns if
                                F"{SingleModel.CLASS}_" in column])

        y_probs = df_pred[class_columns].to_numpy()
        y_true = df_pred[F"{SingleModel.LABEL}"].to_numpy()

        return SingleModel.evaluate(y_probs=y_probs, y_true=y_true)

    @staticmethod
    def image_to_patient(df_pred_image: pd.DataFrame):
        """
        Converts a dataframe for single images into one per patient.
        """
        df_pred_patient = df_pred_image.groupby(
            [SingleModel.ID]).mean().reset_index()

        return df_pred_patient


class SelectEnsemble:
    """
    Selecting the best model for each modality and then ensembling them,
    either by each modality contributing equally or only selecting the model
    combination which optimizes the criterion.
    """
    VALID_SCORES = ["bce", "mse", "roc_auc", "acc"]
    MINIMIZE_SCORES = ["bce", "mse"]
    VALID_MODES = ["image", "patient"]
    VALID_METHODS_ENSEMBLE_SELECTION = ["all",
                                        "equal_contribution",
                                        "linear_solver",
                                        "reg_linear",
                                        "reg_lasso",
                                        "reg_ridge",
                                        "log_reg_l1",
                                        "log_reg_l2"]

    def __init__(self,
                 file_task: str,
                 dir_base: str,
                 criterion: list,
                 modality_idxs_per_version: dir,
                 method_ensemble_selection: str,
                 alpha: float,
                 C: float,
                 eps_preds: float,
                 eps_labels: float,
                 axial_only: bool,
                 transform_2D,
                 transform_3D,
                 load_best_model_info: dict):
        # Dismiss certain versions if they do not improve the validation score.
        self.method_ensemble_selection = method_ensemble_selection
        assert self.method_ensemble_selection in self.VALID_METHODS_ENSEMBLE_SELECTION, \
            "No valid ensemble selection method provided."

        self.regression_model = None
        self.scaler = None

        self.file_task = file_task
        self.task = task.Task(file_task=file_task)
        self.dir_base = dir_base

        self.modality_idxs_per_version = modality_idxs_per_version
        self.versions = list(self.modality_idxs_per_version.keys())
        # For lasso and ridge.
        self.alpha = alpha
        # For l1 and l2 regularization of logistic regression.
        self.C = C
        # For logit transformation.
        self.eps_preds = eps_preds
        self.eps_labels = eps_labels

        self.axial_only = axial_only
        self.transform_2D = transform_2D
        self.transform_3D = transform_3D

        self.criterion = criterion
        self.mode = self.criterion[0]
        assert self.mode in self.VALID_MODES, \
            "No valid mode provided."
        assert self.criterion[-1].lower() in self.VALID_SCORES, \
            "No valid score provided."
        self.minimize = self.criterion[-1].lower() in self.MINIMIZE_SCORES

        ####

        # Selecting the best model for each version.
        # In case they were calculated already, there is the option to just
        # load them from a dictionary.
        (self.best_score_per_version, self.best_model_per_version,
         self.best_id_per_version,
         self.dir_model_per_version) = self._get_model_details_per_version(
            load_best_model_info=load_best_model_info)

        ####

        # Determining the optimal weights of each version.
        self.weights_per_version, self.best_score = (
            self._determine_weights_per_version())

        return

    def weighted_prediction(self,
                            tvt: str,
                            verbose: bool = True):
        """
        The actual weighted prediction. The only relevant method for inference.
        """
        assert self.weights_per_version is not None, ("Weights per version "
                                                      "have not yet been "
                                                      "initialized!")
        # Linear regression requires logits!
        to_logit = "reg_" in self.method_ensemble_selection

        preds_per_version, labels, sample_ids = self._preds_per_version(
            tvt=tvt,
            to_logit=to_logit)

        # 2D weights ensures element-wise multiplication along correct axis.
        preds_weighted = np.sum(preds_per_version *
                                self.weights_per_version.reshape((
                                    1, -1)), axis=1)

        # Output needs to be range 0 to 1!
        if to_logit:
            preds_weighted = self._sigmoid(arr=preds_weighted)
            labels = np.round(self._sigmoid(arr=labels), 0).astype(int)

        df_preds_weighted = pd.DataFrame({SingleModel.ID: sample_ids,
                                          "label": labels,
                                          "class_0": 1 - preds_weighted,
                                          "class_1": preds_weighted})

        if verbose:
            print("The weighted output of the models:")
            print(df_preds_weighted)

        return df_preds_weighted

    def _preds_per_version(self,
                           tvt: str,
                           to_logit):
        """
        Return the predictions of all versions.
        """
        # n_samples x n_versions
        preds_per_version = None
        sample_ids = None
        labels = None
        for version_idx, version in enumerate(self.versions):
            model = self.best_model_per_version[version]
            modality_idxs = self.modality_idxs_per_version[version]
            df_pred_image, df_pred_patient = model.predict_single_modality(
                file_task=self.file_task,
                tvt=tvt,
                modality_idxs=modality_idxs,
                axial_only=self.axial_only,
                verbose=False
            )
            df_pred = df_pred_image if (self.mode == "image") else (
                df_pred_patient)
            # Initializing the accumulated array.
            if preds_per_version is None:
                class_columns = sorted([column for column in df_pred.columns
                                        if SingleModel.CLASS in column.lower()])
                assert len(class_columns) == 2, ("Only binary classification "
                                                 "supported for now.")
                assert class_columns == ["class_0", "class_1"], \
                    "Something went wrong when assigning class names."
                sample_ids = df_pred[SingleModel.ID]
                labels = df_pred[SingleModel.LABEL].to_numpy().flatten()
                # Only ever work with logits in the case of linear regression!!
                if to_logit:
                    labels = self._logit(arr=labels,
                                         eps=self.eps_labels)

                n_versions = len(self.versions)
                n_samples = df_pred.shape[0]

                preds_per_version = np.zeros((n_samples, n_versions))

            # Only consider the probabilities of class 1,
            # since class 0 == 1 - class 1.
            preds_single_version = df_pred["class_1"].to_numpy().flatten()
            # Transform the probabilities to logits, but NOT for linear_solver!
            if to_logit:
                preds_single_version = self._logit(preds_single_version,
                                                   eps=self.eps_preds)
            # Insert the values.
            preds_per_version[..., version_idx] = preds_single_version

        return preds_per_version, labels, sample_ids

    def predict_evaluate(self,
                         tvt: str,
                         verbose: bool = True):
        """
        Allows the prediction to be immediately followed by an evaluation.
        """
        df_preds_weighted = self.weighted_prediction(tvt=tvt,
                                                     verbose=verbose)
        dict_eval = SingleModel.evaluate_image_or_patient(
            df_pred=df_preds_weighted)

        if verbose:
            print(F"Performance of ensemble on the {tvt} dataset:")
            for score in dict_eval.keys():
                print(F"{score}: ", dict_eval[score])

        return df_preds_weighted, dict_eval

    def _determine_weights_per_version(self):
        """
        Compute the mean vote across all possible combos of versions and
        select the combo optimizing the validation loss.
        """
        # Only linear regression works with logits!
        to_logit = False

        if self.method_ensemble_selection == "all":
            optimizer = self._all
        elif self.method_ensemble_selection == "equal_contribution":
            optimizer = self._equal_contribution
        elif self.method_ensemble_selection == "linear_solver":
            optimizer = self._linear_solver
        else:
            optimizer = self._regression
            to_logit = True

        print("Computing the best combination of versions...")

        # ALWAYS fit on the validation dataset!
        preds_per_version, labels, sample_ids = self._preds_per_version(
            tvt="val",
            to_logit=to_logit)

        # The actual optimization step.
        weights_per_version, best_score = optimizer(
            preds_per_version=preds_per_version,
            labels=labels)

        print("Finished computing the best combination of versions.")

        print(
            "The weights of each version optimizing the validation score are:")
        print(",\n".join([F"{version}: {round(weight, 3)} ("
                          F"Contribution"
                          F" of {round(weight / sum(weights_per_version), 3)})"
                          for version, weight in zip(self.versions,
                                                     weights_per_version)]))
        # Escape symbol for readable formatting.
        print(F"The optimal score ({self.criterion}) is: {best_score}\n")

        return weights_per_version, best_score

    def _all(self,
             preds_per_version: np.ndarray,
             labels: np.ndarray):
        """
        Skip any optimization in case a model of each version is to be selected.
        """
        n_versions = len(self.versions)
        weights_per_version = np.repeat(1 / n_versions, n_versions)

        preds_per_version = np.mean(preds_per_version, axis=1)
        best_score = self._score_from_class_1(preds=preds_per_version,
                                              labels=labels,
                                              criterion=self.criterion,
                                              apply_sigmoid=False)

        return weights_per_version, best_score

    def _equal_contribution(self,
                            preds_per_version: np.ndarray,
                            labels: np.ndarray):
        """
        Select only the models which maximize the score under their mean when
        contributing equally (a special case of l1).
        """

        # All possible combinations of versions by indices.
        version_idxs = range(len(self.versions))
        combos_version_idxs = [list(combo)
                               for len_combo in range(1, len(version_idxs) + 1)
                               for combo in
                               itertools.combinations(version_idxs, len_combo)]

        best_combo = None
        best_score = float("inf") if self.minimize else float("-inf")
        for combo in combos_version_idxs:
            preds_combo = preds_per_version[..., combo]
            # IMPORTANT: In the case of equal_contribution, just compute the
            # mean over all versions' outputs.
            preds_combo = np.mean(preds_combo, axis=1)

            score = self._score_from_class_1(preds=preds_combo,
                                             labels=labels,
                                             criterion=self.criterion,
                                             apply_sigmoid=False)
            print(F"Score of combo {combo}: {score}")

            if (((score < best_score) & self.minimize)
                    | ((score > best_score) & ~self.minimize)):
                # Ensures that the versions are printed in order.
                best_combo = sorted(combo)
                best_score = score

        # Since all selected versions are to contribute equally, just set
        # their weight to 1 divided by the number of selected versions and the
        # others to 0.
        weights_per_version = np.array([1 / len(best_combo)
                                        if version_idx in best_combo
                                        else 0
                                        for version_idx in version_idxs])

        return weights_per_version, best_score

    def _linear_solver(self,
                       preds_per_version: np.ndarray,
                       labels: np.ndarray):
        """
        Applying a linear solver for the convex problem of optimal weighting
        of ensembles, with constraint sum_of_w = 1.
        """
        ls = linear_solver.PositiveLinearEnsembler(reg_l2=0.0,
                                                   eps=1e-12,
                                                   maxiter=1000,
                                                   ftol=1e-9)
        ls.fit(P=preds_per_version, y=labels)

        preds = ls.predict(P=preds_per_version)
        best_score = self._score_from_class_1(preds=preds,
                                              labels=labels,
                                              criterion=self.criterion,
                                              apply_sigmoid=False)

        return ls.weights_, best_score

    def _regression(self,
                    preds_per_version: np.ndarray,
                    labels: np.ndarray):
        """
        Applying l1 or l2 for version selection, positive weights only.
        """
        penalty = self.method_ensemble_selection
        if "log_reg" in penalty:
            self.regression_model = sklm.LogisticRegression(
                penalty=penalty.split("_")[-1],
                C=self.C,
                solver="liblinear")
        elif penalty == "reg_linear":
            self.regression_model = sklm.LinearRegression(fit_intercept=False,
                                                          positive=True)
        elif penalty == "reg_lasso":
            self.regression_model = sklm.Lasso(alpha=self.alpha,
                                               positive=True)
        elif penalty == "reg_ridge":
            self.regression_model = sklm.Ridge(alpha=self.alpha,
                                               positive=True)
        else:
            raise ValueError(F"Invalid penalty {penalty}!")

        # Fitting the model.
        self.regression_model.fit(X=preds_per_version, y=labels)
        # The weights per version. The weights must always be 1D!
        weights_per_version = self.regression_model.coef_.flatten()

        # Evaluating the fit.
        if "log_reg" in penalty:
            y_proba = self.regression_model.predict_proba(X=preds_per_version)
            preds = y_proba[:, 1]
        else:
            # 1-class output has to be extended to 2-class output.
            preds = self.regression_model.predict(X=preds_per_version)

        best_score = self._score_from_class_1(preds=preds,
                                              labels=labels,
                                              criterion=self.criterion,
                                              apply_sigmoid=True)

        return weights_per_version, best_score

    def _apply_regression(self,
                          preds_per_version: np.ndarray):
        """
        OUTDATED: Serves as the weight application function in the case of
        regression.
        """

        # Actual prediction.
        if "log_reg" in self.method_ensemble_selection:
            y_probs = self.regression_model.predict_proba(X=preds_per_version)
        else:
            output = self.regression_model.predict_single_modality(X=preds_per_version)
            y_probs = np.zeros((input.shape[0], 2))
            y_probs[:, 1] = output
            y_probs[:, 0] = 1 - output

        return y_probs

    def _get_model_details_per_version(self,
                                       load_best_model_info: dict):
        """
        For EACH version, identify the best model by the provided criterion.
        """
        if load_best_model_info is not None:
            warnings.warn("Dictionary of best model per version provided! "
                          "Skipping the computation of the best model.")

        best_score_per_version = {}
        best_model_per_version = {}
        best_id_per_version = {}
        dir_per_version = {}

        print("Identifying the best models per version...")

        for i, version in enumerate(self.modality_idxs_per_version.keys()):
            print("===========================================================")
            print(F"Version {i + 1}/"
                  F"{len(self.modality_idxs_per_version.keys())}: {version}")
            if load_best_model_info is None:
                modality_idxs = self.modality_idxs_per_version[version]
                (best_id, best_model, best_score,
                 dir_version) = self._compute_best_model_single_version(
                    modality_idxs=modality_idxs)
            else:
                (best_id, best_model, best_score,
                 dir_version) = self._load_best_model_single_version(
                    load_best_model_info=load_best_model_info,
                    version=version)
            best_score_per_version[version] = best_score
            best_model_per_version[version] = best_model
            best_id_per_version[version] = best_id
            dir_per_version[version] = dir_version

        # Printing the results:
        print("Finished identifying the best models per version.")
        print("Best model per version:")
        for version in self.modality_idxs_per_version.keys():
            print(F"######\n{version}")
            print("Score:", best_score_per_version[version])
            print("Model ID:", best_id_per_version[version])

        return (best_score_per_version, best_model_per_version,
                best_id_per_version, dir_per_version)

    def _compute_best_model_single_version(self,
                                           modality_idxs: list):
        """
        For A SINGLE version, identify the best model by the provided criterion.
        """
        str_version = self.task.channel_couple_to_str(
            channel_couple=modality_idxs)
        dir_version = os.path.join(self.dir_base, str_version)
        model_ids = os.listdir(dir_version)
        best_id = None
        best_model = None
        best_score = float("inf") if self.minimize else float("-inf")
        for i, model_id in enumerate(model_ids):
            print("######")
            print(F"Model {i + 1}/{len(model_ids)}: {model_id}")
            single_model, score = self._evaluate_single_model(model_id=model_id,
                                                              dir_version=dir_version,
                                                              modality_idxs=modality_idxs)
            print(F"{model_id}: {score}")
            if (((score < best_score) & self.minimize)
                    | ((score > best_score) & ~self.minimize)):
                best_id = model_id
                best_model = single_model
                best_score = score

        return best_id, best_model, best_score, dir_version

    def _load_best_model_single_version(self,
                                        load_best_model_info: dict,
                                        version: str):
        """
        Load the optimal model for a single version from a dictionary
        provided by the user.
        """
        model_id = load_best_model_info[version]["model_id"]

        modality_idxs = load_best_model_info[version][
            "channel_couple"]
        str_version = self.task.channel_couple_to_str(
            channel_couple=modality_idxs)
        dir_version = os.path.join(self.dir_base, str_version)
        single_model, score = self._evaluate_single_model(model_id=model_id,
                                                          dir_version=dir_version,
                                                          modality_idxs=modality_idxs)

        return model_id, single_model, score, dir_version

    def _evaluate_single_model(self,
                               model_id: str,
                               dir_version: str,
                               modality_idxs: list):
        """
        Helper function to predict and evaluate a single model from its ID.
        """
        single_model = SingleModel(model_id=model_id,
                                   dir_base=dir_version,
                                   transform_2D=self.transform_2D,
                                   transform_3D=self.transform_3D)
        dict_summary = single_model.predict_evaluate(
            file_task=self.file_task,
            tvt="val",
            modality_idxs=modality_idxs,
            axial_only=self.axial_only
        )
        score = dict_summary
        for c in self.criterion:
            score = score[c]

        return single_model, score

    def print_summary(self,
                      file_export=None):
        """"
        Prints and exports the salient information of an ensemble, including
        the best models per version and the versions which were used for the
        final prediction.
        """
        str_versions = ", ".join(self.versions)

        dict_ensemble_summary = {"criterion": self.criterion,
                                 "versions": str_versions,
                                 "ensemble_selection": self.method_ensemble_selection,
                                 "alpha": self.alpha,
                                 "C": self.C,
                                 "normalize": self.normalize,
                                 "weights_version": self.weights_per_version}
        print("Details of version ensembling:")
        print("Criterion: ", self.criterion)
        print("Versions: ", str_versions)
        print("Ensemble selection method: ", self.method_ensemble_selection)
        print("Lasso/ridge alpha: ", self.alpha)
        print("Logistic regression C: ", self.C)
        print("Normalize input: ", self.normalize)
        print("Weights per version: ", self.weights_per_version)

        # The validation scores of the ensemble.
        df_pred_ensemble, arr_acc, dict_eval = self.predict_evaluate(
            verbose=True)
        dict_ensemble_summary["performance_validation"] = dict_eval

        print("")

        for version_idx, version in enumerate(self.versions):
            # 1. Name of the version.
            dict_ensemble_summary[version] = {}
            print(F"{version_idx + 1}. version:", version)

            score = self.best_score_per_version[version]
            print(F"Validation score ({self.criterion[-1]}): {score}")

            model_id = self.best_id_per_version[version]
            dir_version = self.dir_model_per_version[version]
            # 2. Exact location of model weights file.
            file_weights \
                = os.path.join(dir_version, "models", model_id + ".pth")
            dict_ensemble_summary[version]["file_weights"] = file_weights
            print("Weights file: ", file_weights)

            file_hyperparam_settings = os.path.join(dir_version,
                                                    model_id,
                                                    "hyperparams.json")
            with open(file_hyperparam_settings, "r") as f:
                dict_hyperparams = json.load(f)

            # 3. Model hyperparameters.
            dict_ensemble_summary[version]["hyperparams"] = dict_hyperparams
            print("Hyperparameters:")
            for key in dict_hyperparams.keys():
                # Allows the pretty printing of subdicts of depth 1.
                if isinstance(dict_hyperparams[key], dict):
                    for subkey in dict_hyperparams[key].keys():
                        print(F"{key}_{subkey}:", dict_hyperparams[key][subkey])
                else:
                    print(F"{key}:", dict_hyperparams[key])

            print("")

        # Optionally: Export the details of the selected models and versions.
        if file_export is not None:
            with open(file_export, "w") as f:
                json.dump(dict_ensemble_summary, f)

        return

    @staticmethod
    def _logit(arr: np.ndarray,
               eps: float):
        """
        Performs the logit transformation, as the inverse of sigmoid.
        """
        # Avoid divide-by-zero errors.
        arr = np.clip(arr, eps, 1 - eps)

        logits = np.log(arr / (1 - arr))

        return logits

    @staticmethod
    def _sigmoid(arr: np.ndarray):
        """
        Performs the sigmoid transformation, as the inverse of logit.
        """
        return 1 / (1 + np.exp(-arr))

    @staticmethod
    def _score_from_class_1(preds: np.ndarray,
                            labels: np.ndarray,
                            criterion: list,
                            apply_sigmoid: bool):
        """
        Helper function to extract the best score from an ensemble which only
        returns the probabilities of class 1.
        """
        assert ((preds.ndim == 1)
                or ((preds.ndim == 2) and (preds.shape[1] == 1))), \
            "Parameter preds needs to be 1D!"

        probs_per_class = np.zeros((preds.shape[0], 2))
        if apply_sigmoid:
            probs_per_class[:, 1] = SelectEnsemble._sigmoid(preds)
            labels = np.round(SelectEnsemble._sigmoid(arr=labels), 0)
        else:
            probs_per_class[:, 1] = preds

        probs_per_class[:, 0] = 1 - probs_per_class[:, 1]

        dict_eval = SingleModel.evaluate(y_probs=probs_per_class,
                                         y_true=labels)
        # No iterating over the components of the criterion needed,
        # since evaluate() returns no subdicts!
        score = dict_eval[criterion[-1]]

        return score
