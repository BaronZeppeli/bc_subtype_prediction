import os

import json
import re

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize


class EnsembleEvaluation:

    @staticmethod
    def evaluate(dir_predictions: str,
                 dir_export: str):
        """
        Calculates the performance of an ensemble and saves it as a JSON file.
        """
        files_export = []

        dict_tasks = EnsembleEvaluation._create_dict_tasks(
            dir_predictions=dir_predictions)

        for task_name in dict_tasks.keys():
            dict_task = dict_tasks[task_name]

            dict_summary_statistics = {"task": task_name,
                                       "ensembles": {}}

            for ensemble_id in dict_task.keys():
                dict_statistics_ensemble = {"sample_type": {}}

                for sample_type in dict_task[ensemble_id].keys():
                    file_prediction = dict_task[ensemble_id][sample_type]

                    with open(file_prediction, "r") as f:
                        dict_prediction = json.load(f)

                    n_labels = dict_prediction["n_labels"]
                    n_classes_per_label = dict_prediction["n_classes_per_label"]

                    samples = dict_prediction["samples"]

                    (gt_per_label, pred_per_label,
                     probs_mean_per_label) = (EnsembleEvaluation
                    ._create_gt_pred_probs_mean_per_sample(
                        samples=samples,
                        n_labels=n_labels,
                        n_classes_per_label=n_classes_per_label))

                    dict_statistics_single_file = (
                        EnsembleEvaluation._create_statistics_single_file(
                            gt_per_label=gt_per_label,
                            pred_per_label=pred_per_label,
                            probs_mean_per_label=probs_mean_per_label,
                            n_classes_per_label=n_classes_per_label))

                    dict_statistics_ensemble["sample_type"][sample_type] = (
                        dict_statistics_single_file)
                dict_summary_statistics["ensembles"][
                    ensemble_id] = dict_statistics_ensemble

            file_export = EnsembleEvaluation._export_summary_statistics(
                dict_summary_statistics=dict_summary_statistics,
                dir_export=dir_export,
                task_name=task_name)
            files_export += [file_export]
        return files_export

    @staticmethod
    def evaluate_hook_radiomics(file_prediction: str,
                                file_eval: str,
                                model_id: str):
        """
        This method serves as a hook for the radiomics predictions to have
        access to the same extensive evaluation metrics as the neural network
        ensemble models they are pitted against.
        """
        dict_summary_statistics = {"task": "radiomics",
                                   "ensembles": {}}
        dict_statistics_ensemble = {"sample_type": {}}

        df_prediction = pd.read_csv(file_prediction)
        columns_gt = sorted([column for column in df_prediction.columns
                             if ("_gt" in column)])
        columns_pred = sorted([column for column in df_prediction.columns
                               if ("_pred" in column)])
        columns_pred_proba = sorted([column for column in df_prediction.columns
                                     if ("prob_" in column)])

        n_classes_per_label = int(len(columns_pred_proba) / len(columns_pred))
        gt_per_label = {F"{label_idx}": df_prediction[column].to_numpy()
                        for (label_idx, column) \
                        in enumerate(columns_gt)}
        pred_per_label = {
            F"{label_idx}": {"no_voting_method_used": df_prediction[
                column].to_numpy()} for
            (label_idx, column) \
            in enumerate(columns_pred)}

        probs_mean_per_label = {F"{label_idx}": df_prediction[
            columns_pred_proba[
            label_idx * n_classes_per_label
            :(label_idx + 1) * n_classes_per_label]].to_numpy()
                                for label_idx \
                                in range(len(columns_pred))}

        dict_statistics_single_file = (
            EnsembleEvaluation._create_statistics_single_file(
                gt_per_label=gt_per_label,
                pred_per_label=pred_per_label,
                probs_mean_per_label=probs_mean_per_label,
                n_classes_per_label=n_classes_per_label))
        dict_statistics_ensemble["sample_type"]["patient"] = (
            dict_statistics_single_file)
        dict_summary_statistics["ensembles"][model_id] = (
            dict_statistics_ensemble)

        os.makedirs(os.path.dirname(file_eval), exist_ok=True)
        with open(file_eval, "w") as f:
            json.dump(dict_summary_statistics, f, indent=4)

        return dict_summary_statistics

    @staticmethod
    def tabularize_statistics(file_eval: str,
                              ensemble_id: str):
        """
        Takes an evaluation JSON file and prints a summary table per ensemble.
        """
        dict_table = EnsembleEvaluation._flatten_statistics(file_eval=file_eval)
        df_per_ensemble = EnsembleEvaluation._dict_to_df(d=dict_table)

        if ensemble_id is not None:
            output = df_per_ensemble[ensemble_id]
        else:
            output = df_per_ensemble

        return output

    @staticmethod
    def _flatten_statistics(file_eval: str):
        """
        Creates a flattened dictionary for easy conversion into a dataframe.
        """
        dict_table = {}

        with open(file_eval, "r") as f:
            dict_eval = json.load(f)

        ensembles = dict_eval["ensembles"]

        for ensemble_id in ensembles.keys():
            ensemble = ensembles[ensemble_id]
            for sample_type in ensemble["sample_type"].keys():
                labels = ensemble["sample_type"][sample_type]["labels"]
                label_names = labels.keys()

                for label_name in label_names:
                    label = labels[label_name]
                    # This is done in order to get the number of classes
                    # without knowing the name of the voting methods.
                    voting_method_temp = [key for key in label.keys() if
                                          isinstance(label[key], dict)][0]
                    classes = sorted(list(label[voting_method_temp][
                                              "classes"].keys()))

                    for key_label in label.keys():
                        # Handling for BCE.
                        if not isinstance(label[key_label], dict):
                            # Copies the same BCE value to all classes.
                            for clss in classes:
                                dict_table = EnsembleEvaluation._add_dict_key_value(
                                    d=dict_table,
                                    ensemble_id=ensemble_id,
                                    label=label_name,
                                    clss=clss,
                                    sample_type=sample_type,
                                    key=key_label,
                                    value=label[
                                        key_label])
                        else:
                            voting_method = label[key_label]
                            vm_name = key_label
                            for key_vm in voting_method.keys():
                                # Handling for accuracy.
                                if not isinstance(voting_method[key_vm],
                                                  dict):
                                    for clss in classes:
                                        dict_table = EnsembleEvaluation._add_dict_key_value(
                                            d=dict_table,
                                            ensemble_id=ensemble_id,
                                            label=label_name,
                                            clss=clss,
                                            sample_type=sample_type,
                                            key=F"{key_vm}_total_{vm_name}",
                                            value=voting_method[key_vm])
                                else:
                                    # This assumes that a voting method only
                                    # has keys "accuracy" and "classes"!
                                    for clss in sorted(list(voting_method[
                                                                key_vm].keys())):
                                        for metric in voting_method[key_vm][
                                            clss].keys():
                                            dict_table = EnsembleEvaluation._add_dict_key_value(
                                                d=dict_table,
                                                ensemble_id=ensemble_id,
                                                label=label_name,
                                                clss=clss,
                                                sample_type=sample_type,
                                                key=F"{metric}_{vm_name}",
                                                value=voting_method[key_vm][
                                                    clss][metric])
        return dict_table

    @staticmethod
    def _dict_to_df(d: dict):
        """
        Creates a dataframe per ensemble from a dictionary.
        """
        df_per_ensemble = {}

        for ensemble_id in d.keys():
            dict_ensemble = {}
            metric_names = None
            for label in d[ensemble_id].keys():
                for clss in d[ensemble_id][label].keys():
                    for sample_type in d[ensemble_id][label][clss].keys():
                        if metric_names is None:
                            metric_names = sorted(list(d[ensemble_id][label][
                                                           clss][
                                                           sample_type].keys()))
                            dict_ensemble["metric"] = metric_names

                        column_name = (F"label={label}_class={clss}"
                                       F"_{sample_type}")
                        dict_ensemble[column_name] = [
                            d[ensemble_id][label][clss][
                                sample_type][metric_name] for metric_name in
                            metric_names]
            # Grouping the dataframe columns by sampling method.
            dict_ensemble = {key: dict_ensemble[key] for key in sorted(
                dict_ensemble.keys(),
                key=EnsembleEvaluation._sort_func_columns)}
            df_ensemble = pd.DataFrame(dict_ensemble).set_index(
                "metric")
            df_per_ensemble[ensemble_id] = df_ensemble

        return df_per_ensemble

    @staticmethod
    def _add_dict_key_value(d: dict,
                            ensemble_id: str,
                            label: str,
                            clss: str,
                            sample_type: str,
                            key: str,
                            value: float):
        """
        Helper function to complete the dictionary for the statistics
        dataframe output.
        """
        if d.get(ensemble_id) is None:
            d[ensemble_id] = {}
        if d[ensemble_id].get(label) is None:
            d[ensemble_id][label] = {}
        if d[ensemble_id][label].get(clss) is None:
            d[ensemble_id][label][clss] = {}
        if d[ensemble_id][label][clss].get(sample_type) is None:
            d[ensemble_id][label][clss][sample_type] = {}

        d[ensemble_id][label][clss][sample_type][key] = value

        return d

    @staticmethod
    def _sort_func_columns(column_name: str):
        if "_" not in column_name:
            sort_name = column_name
        else:
            sort_name = column_name.split("_")[-1] + "_".join(column_name.split(
                "_")[:-1])

        return sort_name

    @staticmethod
    def rank_ensembles_by_performance(file_eval: str,
                                      sample_type: str,
                                      voting_method: str,
                                      metric: str,
                                      label: int,
                                      clss: int):
        """
        Prints the ensembles according to the given performance metric and
        additional settings such as the voting method.
        """
        with open(file_eval, "r") as f:
            dict_eval = json.load(f)

        ensembles = dict_eval["ensembles"]
        dict_metric = {}

        for ensemble_id in ensembles.keys():
            vals_ensemble = ensembles[ensemble_id]["sample_type"][
                sample_type]["labels"][str(label)]
            if metric.lower() == "bce":
                val_max = vals_ensemble[metric]
            else:
                if voting_method is None:
                    # Roundabout way of determining all available voting methods
                    # for a given ensemble.
                    voting_methods = [key for key in vals_ensemble.keys() if
                                      isinstance(vals_ensemble[key], dict)]
                else:
                    voting_methods = [voting_method]

                # IMPORTANT: This assumes all metrics besides BCE to be
                # better when larger!
                val_max = -1
                for voting_method in voting_methods:
                    if (metric.lower() == "accuracy") & (clss is None):
                        val = vals_ensemble[voting_method][metric]
                    else:
                        val = vals_ensemble[voting_method][clss][metric]

                    val_max = val if (val > val_max) else val_max

            dict_metric[ensemble_id] = val_max

        column_metric = metric + "_" + str(clss) if ((clss is not None) & (
                metric.lower() != "bce")) else metric
        df_metric = pd.DataFrame([[key, dict_metric[key]] for key in
                                  dict_metric.keys()],
                                 columns=["ensemble_id", column_metric])
        df_metric = df_metric.sort_values(by=[column_metric], ascending=False)

        print(df_metric.to_string())

        return df_metric

    @staticmethod
    def _create_dict_tasks(dir_predictions: str):
        """
        Creates the dictionary which organizes the prediction files according to
        their, task, their ensemble, and their sampling type (i.e. image or
        patient).
        """
        date_pattern = R"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"

        dict_tasks = {}
        for file in sorted(os.listdir(dir_predictions)):
            ensemble_id = re.search(date_pattern, file).group()

            sample_type = file.split("_")[-1].split(".")[0]
            task_name = \
                file.split(ensemble_id + "_")[1].split("_" + sample_type)[0]

            if task_name not in dict_tasks.keys():
                dict_tasks[task_name] = {}

            if ensemble_id not in dict_tasks[task_name].keys():
                dict_tasks[task_name][ensemble_id] = {}

            dict_tasks[task_name][ensemble_id][sample_type] = os.path.join(
                dir_predictions, file)

        return dict_tasks

    @staticmethod
    def _create_gt_pred_probs_mean_per_sample(samples: list,
                                              n_labels: int,
                                              n_classes_per_label: int):
        """
        Creates for each label a list of the ground truth classes,
        the predicted classes, and the probabilities of the predictions.
        """
        gt_per_label = {}
        pred_per_label = {}
        probs_mean_per_label = {}

        for sample in samples:
            gt = sample["labels_gt"]
            labels_pred = sample["labels_pred"].keys()

            assert len(gt) == len(labels_pred) == n_labels, \
                "The prediction file is not valid in terms of the number of labels!"

            for label in labels_pred:
                if gt_per_label.get(label) is None:
                    gt_per_label[label] = [gt[label]]
                else:
                    gt_per_label[label] += [gt[label]]

                preds_per_voting_method = sample["labels_pred"][
                    label]["predictions"]
                if pred_per_label.get(label) is None:
                    pred_per_label[label] = {}
                for voting_method in preds_per_voting_method.keys():
                    if pred_per_label[label].get(
                            voting_method) is None:
                        pred_per_label[label][voting_method] = [
                            preds_per_voting_method[voting_method]]
                    else:
                        pred_per_label[label][voting_method] += [
                            preds_per_voting_method[voting_method]]

                probs_per_fold = sample["labels_pred"][label][
                    "probs_per_fold"]
                probs_mean = np.mean([probs_per_fold[fold_key] for \
                                      fold_key
                                      in probs_per_fold.keys()],
                                     axis=0)
                assert probs_mean.shape[0] == n_classes_per_label, \
                    ("The prediction file is not valid in "
                     "terms of the number of classes per "
                     "label!")

                if probs_mean_per_label.get(label) is None:
                    probs_mean_per_label[label] = [probs_mean]
                else:
                    probs_mean_per_label[label] += [probs_mean]

        return gt_per_label, pred_per_label, probs_mean_per_label

    @staticmethod
    def _create_statistics_single_file(gt_per_label: dict,
                                       pred_per_label: dict,
                                       probs_mean_per_label: dict,
                                       n_classes_per_label: int):
        """
        Creates the performance of a single ensemble for a single sampling type.
        """
        dict_statistics_single_file = {"labels": {}}

        for label in gt_per_label.keys():
            dict_label = {}

            y_true = np.array(gt_per_label[label])
            y_probs = np.array(probs_mean_per_label[label])

            # Long-winded sanity check.
            y_pred_per_voting_method = pred_per_label[label]
            voting_methods = y_pred_per_voting_method.keys()
            n_preds_per_voting_method = np.array([len(
                y_pred_per_voting_method[voting_method]) for
                voting_method in voting_methods])
            assert np.all(n_preds_per_voting_method ==
                          n_preds_per_voting_method[
                              0]), ("Not all samples have the same "
                                    "number of available voting "
                                    "method predictions!")

            # The 1 statistic independent of the voting method
            # and the class.
            bce = skm.log_loss(y_true=y_true, y_pred=y_probs)
            dict_label["bce"] = bce

            y_true_binarized = EnsembleEvaluation._binarize_labels(arr=y_true,
                                                                   n_classes_per_label=n_classes_per_label)

            for voting_method in voting_methods:
                dict_voting_method = {}
                y_pred = y_pred_per_voting_method[voting_method]
                y_pred_binarized = EnsembleEvaluation._binarize_labels(
                    arr=y_pred,
                    n_classes_per_label=n_classes_per_label)

                # Independent of the class.
                dict_voting_method[
                    "accuracy"] = EnsembleEvaluation._accuracy(
                    y_true=y_true,
                    y_pred=y_pred)
                dict_voting_method["classes"] = {}

                for class_idx in range(n_classes_per_label):
                    dict_class = {}

                    # It is redundant to repeat ROC-AUC for each voting
                    # method, however since the voting method is higher in
                    # the hierarchy than the classes, it has to be repeated.

                    # VERY IMPORTANT: roc_auc_score expects the probabilities
                    # for the class with the highest CLASS INDEX, i.e. the
                    # second class, or in the case of a multi-class setting,
                    # the summed probabilities of all but the current class.
                    roc_auc = skm.roc_auc_score(y_true_binarized[:, class_idx],
                                                1 - y_probs[:, class_idx])
                    f1 = skm.f1_score(y_true_binarized[:, class_idx],
                                      y_pred_binarized[:, class_idx])
                    acc_balanced = skm.balanced_accuracy_score(
                        y_true_binarized[:,
                        class_idx],
                        y_pred_binarized[:, class_idx])
                    acc = EnsembleEvaluation._accuracy(
                        y_true=y_true_binarized[
                               :, class_idx],
                        y_pred=y_pred_binarized[
                               :, class_idx])
                    sensitivity = EnsembleEvaluation._sensitivity(
                        y_true=y_true_binarized[
                               :, class_idx],
                        y_pred=y_pred_binarized[
                               :, class_idx])
                    specificity = EnsembleEvaluation._specificity(
                        y_true=y_true_binarized[
                               :, class_idx],
                        y_pred=y_pred_binarized[
                               :, class_idx])

                    dict_class["accuracy"] = acc
                    dict_class["accuracy_balanced"] = acc_balanced
                    dict_class["roc_auc"] = roc_auc
                    dict_class["f1"] = f1
                    dict_class["sensitivity"] = sensitivity
                    dict_class["specificity"] = specificity
                    dict_voting_method["classes"][str(class_idx)] \
                        = (
                        dict_class)

                dict_label[voting_method] = dict_voting_method
            dict_statistics_single_file["labels"][
                label] = dict_label

        return dict_statistics_single_file

    @staticmethod
    def _binarize_labels(arr: np.ndarray,
                         n_classes_per_label: int):
        """
        Handles the binarization of the ground truth or predictions per label.
        """
        arr_binarized = label_binarize(arr,
                                       classes=range(
                                           n_classes_per_label))
        # Handling for a two-class setting. For some reason,
        # label_binarize returns an array of shape (n,1) instead of (n,2).
        if n_classes_per_label == 2:
            arr_binarized_two_class = np.zeros(arr_binarized.shape[
                                               :1] + (2,))
            arr_binarized_two_class[:, 0] = arr_binarized.flatten()
            arr_binarized_two_class[:, 1] = 1 - arr_binarized.flatten()
            arr_binarized = arr_binarized_two_class

        return arr_binarized

    @staticmethod
    def _export_summary_statistics(dict_summary_statistics: dict,
                                   dir_export: str,
                                   task_name: str):
        """
        Exports the summary statistics per task to a JSON file.
        """
        if not os.path.isdir(dir_export):
            os.makedirs(dir_export)
        file_export = os.path.join(dir_export, task_name + ".json")
        with open(file_export, "w") as f:
            f.write(json.dumps(dict_summary_statistics, indent=4))

        return file_export

    @staticmethod
    def _roc_auc(y_true, y_pred):
        return skm.roc_auc_score(y_true, y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return np.sum(y_pred == y_true) / y_true.shape[0]

    @staticmethod
    def _sensitivity(y_true, y_pred):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        all_positive = np.sum(y_true == 1)

        return tp / all_positive

    @staticmethod
    def _specificity(y_true, y_pred):
        tn = np.sum((y_pred == 0) & (y_true == 0))
        all_negative = np.sum(y_true == 0)

        return tn / all_negative
