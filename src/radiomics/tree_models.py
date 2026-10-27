import sys
import os
import platform
import itertools

import json
import warnings
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              GradientBoostingClassifier)
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

from tqdm import tqdm
import joblib
from datetime import datetime

import omics

from resnet_subtype.task import Task


class RadiomicsExtractor:
    def __init__(self,
                 file_task: str,
                 roi_id: int = 1):

        # Carries with it A LOT of dependencies, so only load those packages
        # in this class.
        import task_preprocess
        from data_processing.voxelizer import VoxelArrayProcessor

        with open(file_task, 'r') as f:
            self.dict_task = json.load(f)
        self.task = task_preprocess.TaskPreprocess(file_task=file_task)

        # IMPORTANT: 1 is the tumor ROI!
        self.roi_id = roi_id

        return

    def extract(self,
                dir_export: str):
        self.task.preprocess_radiomics()

        for t in ["training", "test"]:
            dict_features = {"patient": []}

            for patient in self.dict_task[t]:
                patient_name = os.path.basename(patient["image"])
                dict_features["patient"] = patient_name

                print("Computing features for patient", patient_name, ".")

                files_volume = [os.path.join(self.task.dir_base, patient[
                    "image"], patient_name + "_" + modality + ".nii") for
                                modality in
                                self.task.modalities]
                file_mask = os.path.join(self.task.dir_base, patient[
                    "maskROI"])
                vap_patient = VoxelArrayProcessor(
                    volume_files=files_volume,
                    ROI_file=file_mask)
                arr_volume = vap_patient.voxel_array.array
                arr_mask = vap_patient.ROI_array[..., 0]

                for modality_idx, modality in enumerate(self.task.modalities):
                    X = arr_volume[..., modality_idx][
                        arr_mask == self.roi_id]
                    idxs_X = np.array(np.where(arr_mask == self.roi_id)).T
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

            df_features = pd.DataFrame(dict_features)
            file_features = os.path.join(dir_export, F"radiomics_"
                                                     F"{self.task.name}"
                                                     F"_{t}.csv")
            os.makedirs(dir_export, exist_ok=True)
            df_features.to_csv(file_features, index=False)

        return


class RadiomicsTrain:
    PATIENT_ID_COLUMN = "patient"
    KEY_MODEL_ID = "id"
    KEY_MODEL_TYPE = "model_type"
    RANDOM_STATE = "random_state"
    N_ITERATIONS = "n_iterations"
    KEY_HYPERPARAMS = "hyperparams"
    STR_TO_MODEL = {"random_forest_classifier":
                        lambda random_state: RandomForestClassifier(
                            random_state=random_state),
                    "random_forest_regressor":
                        lambda random_state: RandomForestRegressor(
                            random_state=random_state),
                    "logistic_regression":
                        lambda random_state: LogisticRegression(
                            random_state=random_state),
                    "svm":
                        lambda random_state: MultiOutputClassifier(
                            estimator=SVC(random_state=random_state)),
                    "gradient_boosting":
                        lambda random_state: MultiOutputClassifier(
                            GradientBoostingClassifier(
                                random_state=random_state)),
                    "xgboost_classifier":
                        lambda random_state: xgb.XGBClassifier(
                            random_state=random_state),
                    "xgboost_regressor":
                        lambda random_state: xgb.XGBRegressor(
                            random_state=random_state,
                            # Bounds the output between 0 and 1.
                            objective="reg:logistic")}

    @staticmethod
    def preprocess_train(model_type: str,
                         random_state: int,
                         arr_radiomics: np.ndarray,
                         arr_labels: np.ndarray,
                         ):
        """
        Helper function to preprocess input and labels, in order to reuse
        this common functionality in other classes.
        """
        if model_type is not None:
            model = RadiomicsTrain.STR_TO_MODEL[model_type](
                random_state=random_state)
        else:
            model = None
        model_id = RadiomicsTrain.create_model_id()

        arr_radiomics, scaler = RadiomicsTrain.normalize_features(
            arr_radiomics=arr_radiomics)

        # Avoids warnings with RandomForestClassifier.
        if arr_labels.shape[1] == 1:
            arr_labels = arr_labels.reshape((-1,))
        else:
            arr_labels = arr_labels

        return model, model_id, arr_radiomics, arr_labels, scaler

    def __init__(self,
                 model_type: str,
                 file_radiomics: str,
                 file_labels: str,
                 random_state: int):
        self.arr_radiomics = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_radiomics)
        self.arr_labels = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_labels)

        self.model_type = model_type
        self.random_state = random_state

        return

    def train(self,
              dir_export: str):
        model, model_id, arr_radiomics, arr_labels, scaler = (
            self.preprocess_train(model_type=self.model_type,
                                  random_state=self.random_state,
                                  arr_radiomics=self.arr_radiomics,
                                  arr_labels=self.arr_labels))

        # One final sanity check.
        assert np.allclose(np.mean(arr_radiomics, axis=0), 0), \
            "Mean of input data is not 0!"
        RadiomicsTrain.check_success_normalization(arr=arr_radiomics)

        # Actual training.
        print(F"Starting training model {model_id}, type:"
              F" {self.model_type}.")
        model.fit(arr_radiomics, arr_labels)
        print(F"Finished training.")

        # Storing the metadata of the model.
        dict_info = {RadiomicsTrain.KEY_MODEL_ID: model_id,
                     RadiomicsTrain.KEY_MODEL_TYPE: self.model_type,
                     RadiomicsTrain.INFO_COLUMN: str(model)}

        # Saving the model and model info.
        file_export_model = os.path.join(dir_export, model_id,
                                         F"{model_id}_model.pkl")
        file_export_info = os.path.join(dir_export, model_id,
                                        F"{model_id}_info.json")
        file_export_scaler = os.path.join(dir_export, model_id,
                                          F"{model_id}_scaler.pkl")

        os.makedirs(os.path.dirname(file_export_model), exist_ok=True)
        joblib.dump(model, file_export_model)
        with open(file_export_info, 'w') as f:
            json.dump(dict_info, f, indent=4)
        joblib.dump(scaler, file_export_scaler)

        return file_export_model, file_export_info, file_export_scaler

    @staticmethod
    def normalize_features(arr_radiomics: np.ndarray,
                           scaler=None):
        scaler = StandardScaler() if (scaler is None) else scaler

        arr_normalized = scaler.fit_transform(X=arr_radiomics)
        RadiomicsTrain.check_success_normalization(arr=arr_normalized)

        return arr_normalized, scaler

    @staticmethod
    def load_csv_as_numpy(file_csv: str):
        df_labels = pd.read_csv(file_csv)
        arr_labels = df_labels.drop(
            columns=[RadiomicsTrain.PATIENT_ID_COLUMN]).to_numpy()

        return arr_labels

    @staticmethod
    def create_model_id():
        # Special handling for Windows, since it cannot handle colons.
        system = platform.system()
        if system.lower() == "windows":
            format_str = "%Y-%m-%d_%H-%M-%S"
        else:
            format_str = "%Y-%m-%d_%H:%M:%S"
        return datetime.now().strftime(format_str)

    @staticmethod
    def check_success_normalization(arr: np.ndarray):
        assert np.allclose(np.mean(arr, axis=0), 0), \
            "Mean of input data is not 0!"
        if np.allclose(np.std(arr, axis=0), 1):
            std_ok = True
        else:
            # A std can still be valid even if it is not 0 in the event that
            # every single value for the affected features is identical.
            std_fail_idxs = np.where(~np.isclose(np.std(arr, axis=0),
                                                 1))[0]
            std_ok = np.unique(arr[..., std_fail_idxs]).shape[0] == 1
        assert std_ok, "Std of input data is not 1!"

        return


class RadiomicsTrainValidation:
    """
    Training and validating a radiomics model in order to select the optimal
    hyperparameters.
    NOTE: Hyperparameter scale_pos_weight (=0.7/0.3) from xgboost can correct
    for a class imbalance between Luminal A and B by using their priors.
    """

    METRICS = {"roc_auc":
                   {"func": skm.roc_auc_score,
                    "proba": True,
                    "maximize": True},
               "acc": {"func": skm.accuracy_score,
                       "proba": False,
                       "maximize": True},
               "acc_balanced": {"func": skm.balanced_accuracy_score,
                                "proba": False,
                                "maximize": True},
               # Is the same as BCE!!
               "log_loss": {"func": skm.log_loss,
                            "proba": True,
                            "maximize": False},
               "f1": {"func": skm.f1_score,
                      "proba": False,
                      "maximize": True},
               }
    SEARCHES = ["grid", "random", "no_search"]

    @staticmethod
    def evaluate(y_probs: np.ndarray,
                 y_true: np.ndarray):
        """
        Applies all relevant metrics to some predicted results.
        """
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

    def __init__(self,
                 search_type: str,
                 file_task: str,
                 file_hyperparam_settings: str,
                 model_type: str,
                 ranked_metrics: list,
                 random_state: int,
                 n_iterations: int):
        self.search_type = search_type
        self.task = Task(file_task=file_task)

        with open(file_hyperparam_settings, "rb") as f:
            self.dict_hyperparam_settings = json.load(f)

        task_name = self.task.name
        file_radiomics_train = os.path.join(self.task.dir_radiomics_input,
                                            F"radiomics_{task_name}_train.csv")
        file_radiomics_val = os.path.join(self.task.dir_radiomics_input,
                                          F"radiomics_{task_name}_val.csv")
        file_labels_train = os.path.join(self.task.dir_radiomics_input,
                                         F"labels_{task_name}_train.csv")
        file_labels_val = os.path.join(self.task.dir_radiomics_input,
                                       F"labels_{task_name}_val.csv")

        self.arr_radiomics_train = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_radiomics_train)
        self.arr_labels_train = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_labels_train)
        if search_type != "no_search":
            self.arr_radiomics_val = RadiomicsTrain.load_csv_as_numpy(
                file_csv=file_radiomics_val)
            self.arr_labels_val = RadiomicsTrain.load_csv_as_numpy(
                file_csv=file_labels_val)
        else:
            self.arr_radiomics_val = None
            self.arr_labels_val = None

        self.n_iterations = n_iterations
        self.model_type = model_type
        self.ranked_metrics = ranked_metrics
        self.random_state = random_state
        # Classifiers and regressors share the same possible hyperparameter
        # values.
        self.search_settings = self.dict_hyperparam_settings[model_type.split(
            "_classifier")[0].split("_regressor")[0]]

        return

    def train(self):
        _, model_id, arr_radiomics_train, arr_labels_train, scaler = (
            RadiomicsTrain.preprocess_train(model_type=self.model_type,
                                            random_state=self.random_state,
                                            arr_radiomics=self.arr_radiomics_train,
                                            arr_labels=self.arr_labels_train))
        # Applying the settings of training to the validation dataset as well.
        if self.arr_radiomics_val is not None:
            arr_radiomics_val = scaler.transform(self.arr_radiomics_val)
        else:
            arr_radiomics_val = None

        # Actual training.
        print(F"Starting training model {model_id}, type:"
              F" {self.model_type}.")

        best_model, best_combo, ranked_best_scores, ranked_scores_per_setting = (
            self._hyperparam_search(
                search=self.search_type,
                n_iterations=self.n_iterations,
                model_type=self.model_type,
                random_state=self.random_state,
                hyperparams=self.search_settings,
                ranked_metrics=self.ranked_metrics,
                arr_radiomics_train=arr_radiomics_train,
                arr_labels_train=arr_labels_train,
                arr_radiomics_val=arr_radiomics_val,
                arr_labels_val=self.arr_labels_val))
        print(F"Finished training.")

        (file_export_model, file_export_info,
         file_export_scaler, file_export_all_scores) = (
            self.export_training_results(
                model=best_model,
                model_id=model_id,
                scaler=scaler,
                hyperparams=best_combo,
                ranked_best_scores=ranked_best_scores,
                ranked_scores_per_setting=ranked_scores_per_setting))

        return (file_export_model, file_export_info,
                file_export_scaler, file_export_all_scores)

    @staticmethod
    def _hyperparam_search(search: str,
                           n_iterations: int,
                           model_type: str,
                           random_state: int,
                           hyperparams: dict,
                           ranked_metrics: list,
                           arr_radiomics_train: np.ndarray,
                           arr_labels_train: np.ndarray,
                           arr_radiomics_val: np.ndarray,
                           arr_labels_val: np.ndarray
                           ):
        """
        Performs the actual training while iterating over all possible grid
        search settings.
        """
        assert search in RadiomicsTrainValidation.SEARCHES, (
            F"{search} is not a valid search mode!")
        assert (search != "random") | (n_iterations is not None), \
            "Must specify n_iterations if mode random is selected!"

        if search != "no_search":
            ranked_metric_infos = [RadiomicsTrainValidation.METRICS[metric]
                                   for metric in ranked_metrics]
            ranked_maximizes = [metric_info["maximize"]
                                for metric_info in ranked_metric_infos]
        else:
            ranked_metric_infos = None
            # Just to avoid indexing errors.
            ranked_maximizes = [1]

        hyperparam_names = hyperparams.keys()
        n_combos = np.prod([len(hyperparams[key]) for key in
                            hyperparams])
        if search == "grid":
            n_iterations = n_combos
            hyperparam_combos = list(itertools.product(*hyperparams.values()))

        elif search == "random":
            random_combo_idxs = []
            # In case there are more requested than possible hyperparameter
            # settings.
            if n_iterations >= n_combos:
                warnings.warn(F"The number of specified iterations "
                              F"({n_iterations}) is greater than the number of "
                              F"possible hyperparameter combinations "
                              F"({n_combos})! Will iterate over all {n_combos} "
                              F"possible combos.")
                random_combo_idxs = np.arange(n_combos)
            else:
                n_remaining = n_iterations
                while n_remaining > 0:
                    idx = np.random.randint(0, n_combos)
                    if idx not in random_combo_idxs:
                        random_combo_idxs += [idx]
                        n_remaining -= 1
        else:
            # In case a search is not requested.
            n_iterations = 1

        total = min(n_iterations, n_combos)
        pbar = tqdm(total=total, desc=F"{search.capitalize()} search")

        best_model = None
        best_combo = None
        # Differentiate between metrics being optimal for 0 and metrics being
        # optimal for 1!!
        ranked_best_scores = [0 if maximize else float("inf")
                              for maximize in ranked_maximizes]
        performs_better = lambda new, old, maximize: (new > old) if maximize \
            else (new < old)

        ranked_scores_per_setting = []

        for iteration_idx in range(n_iterations):
            # IMPORTANT: A new model instance needs to be created for
            # every single hyperparameter setting!!
            model = RadiomicsTrain.STR_TO_MODEL[model_type](
                random_state=random_state)

            # Either work through each possible setting, or work through the
            # number of specified random settings.
            if search == "grid":
                combo = {name: hyperparam_combos[iteration_idx][i]
                         for i, name in enumerate(hyperparam_names)}
            elif search == "random":
                combo_idx = random_combo_idxs[iteration_idx]
                combo = RadiomicsTrainValidation.get_hyperparam_combination(
                    hyperparams=hyperparams,
                    idx=combo_idx)
            else:
                # Skip setting hyperparameters in the case of no_search and
                # just stick with the default param values.
                hyperparam_names = []
                combo = {}

            # Special handling for one specific XGBoostError arising from an
            # exact combination of hyperparameters.
            if "xgboost" in model_type:
                if (combo.get("tree_method") == "exact") & (combo.get(
                        "max_depth") == 0):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue
            if "logistic_regression" in model_type:
                if (combo.get("solver") != "liblinear") & (combo.get(
                        "dual") == True):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue
                if (combo.get("penalty") == "l1") & (combo.get(
                        "dual") == True):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue
                if (combo.get("penalty") == "elasticnet") & (combo.get(
                        "solver") != "saga"):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue
                if (combo.get("solver") == "liblinear") & (combo.get(
                        "penalty") is None):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue
                if (((combo.get("solver") == "lbfgs")
                     | (combo.get("solver") == "newton-cg")
                     | (combo.get("solver") == "newton-cholesky")
                     | (combo.get("solver") == "sag"))
                        & ((combo.get("penalty") is not None) | (
                                combo.get("penalty") == "l2"))):
                    warnings.warn(
                        "Invalid combination of parameters. Skipping.")
                    pbar.update(1)
                    continue

            for name in hyperparam_names:
                setattr(model, name, combo[name])

            # One final sanity check.
            RadiomicsTrain.check_success_normalization(arr=arr_radiomics_train)
            # The actual training of a single gridsearch setting.
            model.fit(arr_radiomics_train, arr_labels_train)

            # No validation evaluation needed in case there is no search
            # being performed.
            if search == "no_search":
                best_model = model
                pbar.update(1)
                continue

            ranked_scores = []
            for metric_idx in range(len(ranked_metrics)):
                if ranked_metric_infos[metric_idx]["proba"]:
                    # n_labels x n_samples x n_classes
                    y_val = model.predict_proba(arr_radiomics_val)
                    # Extend the discrete labels to probabilities by setting
                    # their class probabilities to 1 for their true class and
                    # 0 for the rest.
                    if arr_labels_val.shape[1] == 1:
                        n_samples = arr_labels_val.shape[0]
                        n_classes = np.unique(arr_labels_val).shape[0]
                        arr_labels_val_proba = np.zeros((n_samples, n_classes))
                        arr_labels_val_proba[
                            (np.arange(n_samples),
                             arr_labels_val.flatten())] = 1
                        arr_labels_val = arr_labels_val_proba
                else:
                    y_val = model.predict(arr_radiomics_val)

                # Determine the validation performance. Remember the combo if
                # it increases the current best.
                ranked_scores += [ranked_metric_infos[metric_idx]["func"](
                    arr_labels_val, y_val)]

            # Go through the ranked scores. In case a model performs better
            # than the current best for the iteration metric, select it
            # instead. In case they have the same score, iterate to the next
            # metric to differentiate them. If it is worse, quit.
            for metric_idx in range(len(ranked_metrics)):
                if performs_better(ranked_scores[metric_idx],
                                   ranked_best_scores[metric_idx],
                                   ranked_maximizes[metric_idx]):
                    best_model = model
                    best_combo = combo
                    ranked_best_scores = ranked_scores
                    break
                elif ranked_scores[metric_idx] == ranked_best_scores[
                    metric_idx]:
                    continue
                else:
                    break

            ranked_scores_per_setting += [ranked_scores]

            # Don't forget to update the progress bar!
            pbar.update(1)

        if search != "no_search":
            for metric_idx in range(len(ranked_metrics)):
                print(F"Best {ranked_metrics[metric_idx]} score: "
                      F"{ranked_best_scores[metric_idx]}")
            print(F"Selected hyperparameters:")
            for hyperparam, val in best_combo.items():
                print(F"{hyperparam}: {val}")

        return best_model, best_combo, ranked_best_scores, ranked_scores_per_setting

    @staticmethod
    def get_hyperparam_combination(hyperparams, idx):
        """
        Helper function, to be used in case there are too many hyperparameter
        combos to reasonably use itertools.product.
        """
        keys = list(hyperparams.keys())
        bases = [len(hyperparams[k]) for k in keys]

        combo_indices = []
        for base in reversed(bases):
            combo_indices.append(idx % base)
            idx //= base
        combo_indices = list(reversed(combo_indices))

        return {
            k: hyperparams[k][i] for k, i in zip(keys, combo_indices)
        }

    def export_training_results(self,
                                model,
                                model_id: str,
                                scaler,
                                hyperparams: dict,
                                ranked_best_scores: list,
                                ranked_scores_per_setting: list):
        """
        Storing the metadata of the model.
        """
        dir_export = self.task.dir_radiomics_models

        # Compatibility check.
        if hyperparams is not None:
            for key, val in hyperparams.items():
                if isinstance(val, np.int64):
                    hyperparams[key] = int(val)
        dict_info = {"id": model_id,
                     "model_type": self.model_type,
                     "random_state": self.random_state,
                     "search_type": self.search_type,
                     "n_iterations": self.n_iterations,
                     "ranked_metrics": self.ranked_metrics,
                     "scores": ranked_best_scores,
                     "hyperparams": hyperparams}

        # Saving the model and model info.
        file_export_model = os.path.join(dir_export,
                                         model_id,
                                         F"{model_id}_model.pkl")
        file_export_info = os.path.join(dir_export,
                                        model_id,
                                        F"{model_id}_info.json")
        file_export_search_settings = os.path.join(dir_export,
                                                   model_id,
                                                   F"{model_id}_search_settings.json")
        file_export_scaler = os.path.join(dir_export,
                                          model_id,
                                          F"{model_id}_scaler.pkl")

        # The actual exporting.
        os.makedirs(os.path.dirname(file_export_model), exist_ok=True)

        joblib.dump(model, file_export_model)
        joblib.dump(scaler, file_export_scaler)

        with open(file_export_info, 'w') as f:
            json.dump(dict_info, f, indent=4)
        with open(file_export_search_settings, 'w') as f:
            json.dump(self.search_settings, f, indent=4)

        # Exporting the ranked scores for each grid search model.
        if ranked_scores_per_setting is not None:
            ranked_scores_per_setting = np.array(ranked_scores_per_setting)
            # Flatten the scores if there is only 1 metric.
            if ranked_scores_per_setting.ndim > 1:
                if ranked_scores_per_setting.shape[1] == 1:
                    ranked_scores_per_setting = ranked_scores_per_setting.flatten()
            df_all_scores = pd.DataFrame(ranked_scores_per_setting,
                                         columns=self.ranked_metrics)
            file_export_all_scores = os.path.join(dir_export,
                                                  model_id,
                                                  F"{model_id}_all_scores.csv")
            df_all_scores.to_csv(file_export_all_scores, index=False)

        return file_export_model, file_export_info, file_export_scaler, file_export_all_scores


class RadiomicsTrainReproduce:
    """
    Reproducing a radiomics model 1:1 by its hyperparameters, in order to
    increase portability and having to avoid to move model files around.
    """

    def __init__(self,
                 file_task: str,
                 model_id: str):
        self.task = Task(file_task=file_task)
        file_hyperparams = os.path.join(self.task.dir_radiomics_models,
                                        model_id,
                                        F"{model_id}_info.json")

        (self.model_id, self.model_type, self.model, self.random_state,
         self.hyperparams,
         self.arr_radiomics_train, self.arr_labels_train, self.scaler) \
            = self._reproduce_model(task=self.task,
                                    file_hyperparams=file_hyperparams)

        return

    def train(self):
        self.model.fit(self.arr_radiomics_train, self.arr_labels_train)

        return self.model

    @staticmethod
    def _reproduce_model(task: Task,
                         file_hyperparams: str):
        """
        Reproduces a radiomics model from the information file created after
        training.
        """
        # Loading the hyperparam info.
        with open(file_hyperparams, 'r') as f:
            dict_hyperparams = json.load(f)
        model_id = dict_hyperparams[RadiomicsTrain.KEY_MODEL_ID]
        model_type = dict_hyperparams[RadiomicsTrain.KEY_MODEL_TYPE]
        # Random state receives special treatment compared to the other
        # hyperparameters, since it is shared among all models.
        random_state = dict_hyperparams[RadiomicsTrain.RANDOM_STATE]
        hyperparams = dict_hyperparams[RadiomicsTrain.KEY_HYPERPARAMS]

        file_radiomics_train = task.get_file_radiomics(labels=False,
                                                       tvt="train")
        file_labels_train = task.get_file_radiomics(labels=True,
                                                    tvt="train")

        # Normalizing the input data.
        arr_radiomics_train = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_radiomics_train)
        arr_labels_train = RadiomicsTrain.load_csv_as_numpy(
            file_csv=file_labels_train)
        model, _, arr_radiomics_train, arr_labels_train, scaler = (
            RadiomicsTrain.preprocess_train(model_type=model_type,
                                            random_state=random_state,
                                            arr_radiomics=arr_radiomics_train,
                                            arr_labels=arr_labels_train))
        # Applying the hyperparameter setting.
        for key in hyperparams:
            setattr(model, key, hyperparams[key])

        return (model_id, model_type, model, random_state, hyperparams,
                arr_radiomics_train, arr_labels_train, scaler)


class RadiomicsTrainLOO(RadiomicsTrain):
    """
    Handles the training of a model on the entire dataset using leave-one-out
    cross-validation.
    """

    def __init__(self,
                 model_type: str,
                 file_radiomics: str,
                 file_labels: str,
                 random_state: int):
        super().__init__(model_type=model_type,
                         file_radiomics=file_radiomics,
                         file_labels=file_labels,
                         random_state=random_state)
        # These arrays are not to be changed!
        self.arr_radiomics_all = np.copy(self.arr_radiomics)
        self.arr_labels_all = np.copy(self.arr_labels)

        return

    def train(self,
              dir_export: str):
        model_id = RadiomicsTrain.create_model_id()
        dir_export = os.path.join(dir_export, model_id)

        for sample_idx in range(self.arr_labels_all.shape[0]):
            self.arr_radiomics = np.delete(self.arr_radiomics_all,
                                           sample_idx,
                                           axis=0)
            self.arr_labels = np.delete(self.arr_labels_all,
                                        sample_idx,
                                        axis=0)
            dir_export_sample = os.path.join(dir_export, str(sample_idx))
            _a, file_export_info, _b = super().train(
                dir_export=dir_export_sample)

        # Just pick the last info file, since they will be all identical except
        #  the model id.
        with open(file_export_info, 'r') as f:
            dict_info = json.load(f)
        dict_info[RadiomicsTrain.KEY_MODEL_ID] = model_id
        file_export_info_loo = os.path.join(dir_export, model_id + ".json")
        with open(file_export_info_loo, 'w') as f:
            json.dump(dict_info, f, indent=4)

        return dir_export


class AbstractTreeModel(ABC):
    """
    Class to encapsulate the shared functionality between classes
    RadiomicsModel and RadiomicsModelLOO.
    """

    SUFFIX_TRUE = "_true"
    SUFFIX_PRED = "_pred"

    SCORES_REGRESSION = {"mse": skm.mean_squared_error}

    @staticmethod
    def export_df(df_pred: pd.DataFrame,
                  dir_export: str,
                  model_id: str,
                  task_name: str):
        # Saving the ground-truth labels, the predicted labels, as well as
        # the probabilities per class per label.
        file_export = None
        if dir_export is not None:
            file_export = os.path.join(dir_export, model_id,
                                       "prediction_" +
                                       task_name + ".csv")
            os.makedirs(os.path.dirname(file_export), exist_ok=True)
            df_pred.to_csv(file_export, index=False)
        return file_export

    @staticmethod
    def evaluate_classification(model_id: str,
                                file_prediction: str,
                                task_name: str,
                                dir_export: str):
        file_eval = os.path.join(
            dir_export,
            model_id,
            "evaluation_" + task_name + ".json")
        EnsembleEvaluation.evaluate_hook_radiomics(
            file_prediction=file_prediction,
            file_eval=file_eval,
            model_id=model_id)

        return file_eval

    @staticmethod
    def evaluate_regression(df_pred: pd.DataFrame,
                            dir_export: str,
                            model_id: str):
        label_ids = sorted(
            [column.split(AbstractTreeModel.SUFFIX_PRED)[0] for column in
             df_pred.columns
             if
             AbstractTreeModel.SUFFIX_PRED in column])
        dict_eval = {}
        for label_id in label_ids:
            y_true = df_pred[label_id +
                             AbstractTreeModel.SUFFIX_TRUE].to_numpy().flatten()
            y_pred = df_pred[label_id +
                             AbstractTreeModel.SUFFIX_PRED].to_numpy().flatten()

            dict_label = {score_name: score_func(y_true, y_pred) for
                          score_name, score_func in
                          AbstractTreeModel.SCORES_REGRESSION.items()}
            dict_eval[label_id] = dict_label

        file_export = os.path.join(dir_export, "models", model_id, "evaluation",
                                   F"evaluation_regression_{model_id}.json")
        os.makedirs(os.path.dirname(file_export), exist_ok=True)
        with open(file_export, 'w') as f:
            json.dump(dict_eval, f, indent=4)

        return dict_eval, file_export

    @staticmethod
    def tabularize_statistics(file_eval: str):
        df_stats = EnsembleEvaluation.tabularize_statistics(
            file_eval=file_eval,
            ensemble_id=None)
        # This has to be done since the ensemble method will return a dict
        # with the statistics for each ensemble saved by the ensemble id as key.
        # However, we will only ever create statistics a single radiomics model
        # at a time.
        df_stats = df_stats[list(df_stats.keys())[0]]

        return df_stats

    def _evaluate(self,
                  file_radiomics_train: str,
                  file_prediction: str,
                  dir_export_evaluation: str, ):
        task_name = os.path.basename(file_radiomics_train).split(".")[0]
        file_eval = AbstractTreeModel.evaluate_classification(
            model_id=self.model_id,
            file_prediction=file_prediction,
            task_name=task_name,
            dir_export=dir_export_evaluation)

        df_stats = AbstractTreeModel.tabularize_statistics(file_eval=file_eval)

        # Pretty-printing the results.
        print(F"Predictions for model {self.model_id}:")
        print("=========================================")
        print(F"Model information:")
        print(self.model_info)
        print("Model evaluation:")
        print(df_stats.to_string())

        # Export the summary statistics file.
        file_stats = os.path.join(dir_export_evaluation,
                                  self.model_id,
                                  "statistics_" + task_name + ".csv")
        df_stats.to_csv(file_stats, index=False)

        return df_stats


class TreeModel(AbstractTreeModel):
    """
    Load a model and predict a test radiomics file, given the normalization
    of a training radiomics file.
    """

    @staticmethod
    def load_model(file_task: str,
                   model_id: str):
        """
        Helper to simplify loading a model.
        """
        task = Task(file_task=file_task)
        file_hyperparams = os.path.join(task.dir_radiomics_models,
                                        model_id,
                                        F"{model_id}_info.json")
        file_weights = os.path.join(task.dir_radiomics_models,
                                    model_id,
                                    F"{model_id}_model.pkl")
        file_scaler = os.path.join(task.dir_radiomics_models,
                                   model_id,
                                   F"{model_id}_scaler.pkl")

        return TreeModel(file_task=file_task,
                         file_weights=file_weights,
                         file_hyperparams=file_hyperparams,
                         file_scaler=file_scaler)

    def __init__(self,
                 file_task: str,
                 file_weights: str,
                 file_hyperparams: str,
                 file_scaler: str):
        self.task = Task(file_task=file_task)

        self.model = joblib.load(file_weights)

        # Make the info file optional, since it is only relevant for exporting.
        if file_hyperparams is not None:
            with open(file_hyperparams, 'r') as f:
                dict_info = json.load(f)
            self.id = dict_info[RadiomicsTrain.KEY_MODEL_ID]
            self.type = dict_info[RadiomicsTrain.KEY_MODEL_TYPE]
            self.hyperparams = dict_info[RadiomicsTrain.KEY_HYPERPARAMS]

            self.regression = "_regressor" in self.type.lower()
        else:
            self.id, self.type, self.hyperparams = None, None, None
            self.regression = False

        self.scaler = joblib.load(file_scaler)

        return

    def predict(self,
                tvt: str,
                save: bool = False,
                included_sample_idxs=None):
        """
        Performs the prediction and optionally exports the results as a CSV
        file.
        """
        task_name = self.task.name
        file_radiomics = os.path.join(self.task.dir_radiomics_input,
                                      F"radiomics_{task_name}_{tvt}.csv")
        file_labels = os.path.join(self.task.dir_radiomics_input,
                                   F"labels_{task_name}_{tvt}.csv")
        X = pd.read_csv(file_radiomics)
        y_true = pd.read_csv(file_labels)

        assert X.shape[0] == y_true.shape[
            0], "The radiomics and label data have different sizes!"
        assert np.all(X[RadiomicsTrain.PATIENT_ID_COLUMN] == y_true[
            RadiomicsTrain.PATIENT_ID_COLUMN]), ("The radiomics and label data "
                                                 "are differently sorted or "
                                                 "are not matching entirely!")

        # For LOO: Only predict the specified indices.
        if included_sample_idxs is not None:
            X = X.iloc[included_sample_idxs]
            y_true = y_true.iloc[included_sample_idxs]

        # Turn the dataframes into model-compatible arrays.
        patient_names = X[RadiomicsTrain.PATIENT_ID_COLUMN]
        X = X.drop(
            columns=[RadiomicsTrain.PATIENT_ID_COLUMN]).to_numpy()
        y_true = y_true.drop(
            columns=[RadiomicsTrain.PATIENT_ID_COLUMN]).to_numpy()

        # Apply the normalization from training.
        X = self.scaler.transform(X)

        # The actual prediction.
        # n_samples x n_labels.
        y_pred = self.model.predict(X)
        # Compatibility of single-label with multi-label problems.
        if y_pred.ndim == 1:
            y_pred = y_pred[..., None]
        # n_labels x n_samples x n_classes.
        try:
            y_pred_proba = np.array(self.model.predict_proba(X))
        except AttributeError:
            # In case a model has no implicit probabilistic interpretation (
            # such as SVM), just assign probability 1 to the predicted class.
            n_classes = np.unique(y_pred).shape[0]
            y_pred_proba = np.zeros(y_pred.shape[::-1] + (n_classes,))
            for clss in range(n_classes):
                label_idxs, sample_idxs = np.where(y_pred.T == clss)
                clss_idxs = np.repeat(clss, label_idxs.shape[0])
                y_pred_proba[label_idxs, sample_idxs, clss_idxs] = 1.0

        # Again, ensuring compatibility with single-label tasks.
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[None, ...]

        columns_true = [F"label_{i}{self.SUFFIX_TRUE}" for i in
                        range(y_true.shape[1])]
        columns_pred = [F"label_{i}{self.SUFFIX_PRED}" for i in
                        range(y_true.shape[1])]

        df_pred = pd.DataFrame(
            {RadiomicsTrain.PATIENT_ID_COLUMN: patient_names})
        df_pred[columns_true] = y_true
        df_pred[columns_pred] = y_pred
        # There are no probabilities per class in a regression task!!
        if not self.regression:
            for label_idx in range(y_pred_proba.shape[0]):
                columns_pred_proba = [
                    F"prob_label_{label_idx}_class_{class_idx}"
                    for class_idx in range(y_pred_proba.shape[2])]
                df_pred[columns_pred_proba] = y_pred_proba[label_idx, ...]

        if save:
            dir_export = os.path.join(self.task.dir_radiomics,
                                      "models",
                                      self.id,
                                      "inference")
            file_export = TreeModel.export_df(
                df_pred=df_pred,
                dir_export=dir_export,
                model_id=self.id,
                task_name=task_name)
            print(F"Saved inference to file {file_export}.")

        return df_pred, y_pred, y_pred_proba, y_true

    def predict_evaluate(self,
                         tvt: str,
                         included_sample_idxs: list = None):
        """
        Combining the prediction of the local class with evaluation inspired
        by the ResNetSubtype package.
        """
        df_pred, y_pred, y_pred_proba, y_true = self.predict(
            tvt=tvt,
            save=False,
            included_sample_idxs=included_sample_idxs)

        # n_labels x n_samples x n_classes -> n_samples x n_classes
        y_probs = y_pred_proba[0, ...]

        print("The model outputs:")
        print(df_pred.to_string())

        if not self.regression:
            dict_eval = RadiomicsTrainValidation.evaluate(y_probs=y_probs,
                                                          y_true=y_true)
        else:
            dir_export = self.task.dir_radiomics
            dict_eval, file_export_eval = AbstractTreeModel.evaluate_regression(
                df_pred=df_pred,
                dir_export=dir_export,
                model_id=self.id)

        print(F"The evaluation of the {tvt} dataset:")
        print(dict_eval)

        return dict_eval


class TreeModelLOO(AbstractTreeModel):

    def __init__(self,
                 dir_models: str):

        # The info file will be the only file in the directory with all the
        # LOO subdirectories.
        file_info = [os.path.join(dir_models, file) for file in os.listdir(
            dir_models) if os.path.isfile(
            os.path.join(dir_models, file))][0]
        with open(file_info, 'r') as f:
            dict_info = json.load(f)
        self.model_id = dict_info[RadiomicsTrain.KEY_MODEL_ID]
        self.model_info = dict_info[RadiomicsTrain.INFO_COLUMN]

        self.models = []
        self.left_out_idxs = []
        # Initialize the LOO models and store them in a list.
        self._create_models(dir_models=dir_models)

        return

    def predict(self,
                file_radiomics_training: str,
                file_labels_training: str,
                dir_export: str):
        task_name = os.path.basename(file_labels_training).split(".")[0]

        list_df_loo = []
        for left_out_idx, model in zip(self.left_out_idxs, self.models):
            _, df_loo = model.predict(file_radiomics=file_radiomics_training,
                                      file_labels=file_labels_training,
                                      dir_export=None,
                                      included_sample_idxs=[left_out_idx])
            list_df_loo += [df_loo]
        # Concatenate all 1-element dataframes into a single large dataframe
        # with the length of the training dataset.
        df_results = pd.concat(list_df_loo, axis=0).reset_index(drop=True)

        file_export = TreeModel.export_df(
            df_pred=df_results,
            dir_export=dir_export,
            model_id=self.model_id,
            task_name=task_name)

        return file_export, df_results

    def predict_evaluate(self,
                         file_radiomics_train: str,
                         file_labels_train: str,
                         dir_export_prediction: str,
                         dir_export_evaluation: str):
        file_prediction, _ = self.predict(
            file_radiomics_training=file_radiomics_train,
            file_labels_training=file_labels_train,
            dir_export=dir_export_prediction)
        df_stats = self._evaluate(
            file_radiomics_train=file_radiomics_train,
            file_prediction=file_prediction,
            dir_export_evaluation=dir_export_evaluation)

        return df_stats

    def _create_models(self,
                       dir_models: str):
        """
        Helper function to initialize all models of an LOO run.
        """
        # Make sure to kick out the shared info.json file for the loo models!
        list_dir_fold = [os.path.join(dir_models, dir_fold) for dir_fold in
                         os.listdir(dir_models) if
                         os.path.isdir(os.path.join(dir_models, dir_fold))]

        for dir_fold in list_dir_fold:
            self.left_out_idxs += [int(os.path.basename(dir_fold))]
            # Directory structure: base -> fold_id -> model_id
            subdir_fold = os.path.join(dir_fold, os.listdir(dir_fold)[0])
            assert len(os.listdir(subdir_fold)) == 3, ("Expects exactly 1 "
                                                       "weights "
                                                       "file, 1 info file, "
                                                       "and 1 scaler file in "
                                                       "the fold subdirectory!")
            file_model = os.path.join(subdir_fold, [file for file in os.listdir(
                subdir_fold) if "_model.pkl" in file.lower()][0])
            file_info = os.path.join(subdir_fold, [file for file in os.listdir(
                subdir_fold) if "_info.json" in file.lower()][0])
            file_scaler = os.path.join(subdir_fold, [file for file in
                                                     os.listdir(
                                                         subdir_fold) if
                                                     "_scaler.pkl" in file.lower()][
                0])

            self.models += [TreeModel(file_weights=file_model,
                                      file_hyperparams=file_info,
                                      file_scaler=file_scaler)]

        return
