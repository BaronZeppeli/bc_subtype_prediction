import json
import sys
import os

import warnings

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import radiomics.tree_models as tree_models
from resnet_subtype.task import Task


class SHAPRadiomics:
    """
    Class to apply SHAP to either a random forest or XGBoost model.
    """
    COLUMN_ID = "patient"
    CATEGORIES = ["t1", "stir", "adc", "pet"]
    SHAP_BLUE = shap.plots.colors.red_blue(0)
    SHAP_RED = shap.plots.colors.red_blue(255)

    def __init__(self,
                 file_task: str,
                 model_id: str,
                 tvt: str,
                 summation_mode: str,
                 file_weights:str=None,
                 file_scaler:str=None,
                 file_hyperparams:str=None):
        self.task = Task(file_task=file_task)
        self.model_id = model_id
        self.tvt = tvt
        self.summation_mode = summation_mode

        if file_weights is None:
            file_weights = os.path.join(self.task.dir_radiomics_models,
                                        model_id,
                                        F"{model_id}_model.pkl")
            # For exporting the PNG plots.
            self.dir_export = os.path.join(
                self.task.dir_radiomics_explainability,
                self.model_id,
                self.tvt,
                self.summation_mode)
        else:
            self.dir_export = "data"
        if file_scaler is None:
            file_scaler = os.path.join(self.task.dir_radiomics_models,
                                       model_id,
                                       F"{model_id}_scaler.pkl")
        if file_hyperparams is None:
            file_hyperparams = os.path.join(self.task.dir_radiomics_models,
                                            model_id,
                                            F"{model_id}_info.json")

        self.model = tree_models.TreeModel(file_task=file_task,
                                           file_weights=file_weights,
                                           file_hyperparams=file_hyperparams,
                                           file_scaler=file_scaler)
        with open(file_hyperparams, "r") as file:
            dict_hyperparams = json.load(file)
        self.model_type = dict_hyperparams["model_type"]

        self.X = None
        self.Y = None
        self.feature_names = None
        self.explainer = None
        self.shap_arr = None
        self._initialize_shap()

        self.category_shaps_sum = None
        self.category_shaps_mean = None
        self._categorize()

        # Global export settings.
        # plt.rcParams['figure.figsize'] = (6.4, 3.6)  # smaller inches
        # plt.rcParams['figure.dpi'] = 300

        return

    def _initialize_shap(self):
        """
        Performing the actual SHAP analysis, except for clustering.
        """
        # For initialization of SHAP. Always the same, regardless of tvt.
        file_base_X = self.task.get_file_radiomics(labels=False, tvt="train")

        file_X = self.task.get_file_radiomics(labels=False, tvt=self.tvt)
        file_Y = self.task.get_file_radiomics(labels=True, tvt=self.tvt)

        # Load the radiomics data into an array.
        self.base_X = pd.read_csv(file_base_X)
        self.base_X = self.base_X.drop(columns=[self.COLUMN_ID])
        self.feature_names = list(self.base_X.columns)
        self.base_X = self.base_X.to_numpy()

        self.X = pd.read_csv(file_X)
        self.X = self.X.drop(columns=[self.COLUMN_ID]).to_numpy()

        # Apply the normalization from training.
        self.base_X = self.model.scaler.transform(self.base_X)
        self.base_X = pd.DataFrame(self.base_X, columns=self.feature_names)

        self.X = self.model.scaler.transform(self.X)
        self.X = pd.DataFrame(self.X, columns=self.feature_names)

        # Load the labels.
        self.Y = pd.read_csv(file_Y)
        self.Y = self.Y.drop(columns=[self.COLUMN_ID])

        # The SHAP computation.
        if ("xgboost" in self.model_type) | (
                "random_forest" in self.model_type):
            self.explainer = shap.TreeExplainer(self.model.model)
        elif "logistic_regression" in self.model_type:
            self.explainer = shap.LinearExplainer(self.model.model, self.base_X)
        else:
            raise ValueError(F"Unsupported model type {self.model_type}.")
        # SHAP values in 3rd dimension are symmetrical for each class,
        # therefore only take the first ones.
        self.shap_obj = self.explainer(self.X)
        # Compatibility with random forests.
        if len(self.shap_obj.shape) == 3:
            self.shap_obj = self.shap_obj[..., 0]
        self.shap_arr = self.shap_obj.values

        return

    def _categorize(self):
        """
        Group several features by either summing or meaning their values.
        """
        feat_per_cat = {}
        feat = self.feature_names.copy()
        for category in self.CATEGORIES:
            features_cat = [feature for feature in feat
                            if F"_{category}" in feature]
            feat_per_cat[category] = features_cat
            # Pop the elements.
            feat = [feature for feature in feat if feature not in features_cat]
        assert len(feat) == 0, (F"Features {feat} could not be assigned to a "
                                F"category.")

        self.category_shaps_sum = np.zeros((self.shap_arr.shape[0],
                                            len(self.CATEGORIES)))
        # The mean provides additional information if there is not an equal
        # amount of features for each modality.
        self.category_shaps_mean = np.zeros(self.category_shaps_sum.shape)
        self.data_sum = np.zeros(self.category_shaps_sum.shape)
        self.data_mean = np.zeros(self.category_shaps_sum.shape)
        for cat_idx, category in enumerate(self.CATEGORIES):
            feature_idxs = [i for i, feature in enumerate(self.feature_names) if
                            feature in feat_per_cat[category]]

            self.category_shaps_sum[..., cat_idx] = np.sum(
                self.shap_arr[..., feature_idxs], axis=1)
            self.category_shaps_mean[..., cat_idx] = np.mean(
                self.shap_arr[..., feature_idxs], axis=1)

            self.data_sum[..., cat_idx] = np.sum(
                self.shap_obj.data[..., feature_idxs], axis=1)
            self.data_mean[..., cat_idx] = np.mean(
                self.shap_obj.data[..., feature_idxs], axis=1)

        return

    @staticmethod
    def _plot_bar(shap_values,
                  max_display: int,
                  title: str,
                  file_export: str):
        """
        Helper function to add a title to a SHAP bar plot.
        """
        shap.plots.bar(shap_values=shap_values,
                       max_display=max_display,
                       show=False)
        plt.title(title)
        plt.tight_layout()
        if file_export is not None:
            plt.savefig(file_export)
        else:
            plt.show()

        return

    @staticmethod
    def _set_cohort_decimals(dict_cohort: dict,
                             n_decimals: int = 2):
        """
        Workaround to deal with the ugly formatting of the cohorts splitting
        criterion.
        """
        dict_cohort_new = {}
        for key in dict_cohort.keys():
            assert (" < " in key) | (" >= " in key), (F"The key {key} has "
                                                      F"unexpected formatting.")
            splitter = " < " if (" < " in key) else " >= "
            extracted_float = F"{float(str(key).split(splitter)[1]):.{n_decimals}f}"
            new_key = key.split(splitter)[0] + splitter + extracted_float
            dict_cohort_new[new_key] = dict_cohort[key]

        return dict_cohort_new

    def visual_summary(self,
                       max_display: int,
                       class_names: list,
                       example_idx: int,
                       show: bool):
        """
        Visualizing several statistics about the SHAP values.
        """
        if not show:
            os.makedirs(self.dir_export, exist_ok=True)

        viable_summation_modes = ["single", "sum", "mean"]
        assert self.summation_mode in viable_summation_modes,\
            F"Mode must be one of {viable_summation_modes}."
        assert np.max(self.Y) < len(class_names),\
            "Not enough class names provided."

        if self.summation_mode == "single":
            shap_arr = self.shap_arr
            data = self.shap_obj.data
            # Old version.
            # feature_names = self.feature_names
            # For pretty labeling.
            feature_names = [
                feature.replace("_t1_post_dynamic", "_dce-mri").replace(
                    "_stir", "_t2w") for feature in self.feature_names]

        else:
            if self.summation_mode == "sum":
                shap_arr = self.category_shaps_sum
                data = self.data_sum
            else:
                shap_arr = self.category_shaps_mean
                data = self.data_mean
            # Old version.
            # feature_names = self.CATEGORIES
            # For pretty labeling.
            feature_names = [cat.replace("t1", "DCE-MRI")
                             .replace("stir","T2W")
                             .replace("adc", "ADC")
                             .replace("pet", "PET")
                             for cat in self.CATEGORIES]

        shap_values = shap.Explanation(values=shap_arr,
                                       feature_names=feature_names,
                                       data=data,
                                       base_values=self.shap_obj.base_values)
        expected_value = self.explainer.expected_value

        # Compatibility with random forests.
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]

        ###
        # Summary.

        # Old version from before the thesis.
        # SUFFIX = F", {self.summation_mode}, {self.tvt}"
        SUFFIX = ""

        # 1.1 Summary ranked bar plot absolute values.
        file_export_11 = os.path.join(self.dir_export, "11_bar_summary.png") \
            if not show else None
        self._plot_bar(shap_values=shap_values,
                       max_display=max_display,
                       title=F"Mean absolute values" + SUFFIX,
                       file_export=file_export_11)

        # 1.2 Cohort bar plot, separated by the class of the samples.
        file_export_12 = os.path.join(self.dir_export,
                                      "12_bar_classes.png") if not show else \
            None
        clss = [class_names[y] for y in self.Y.to_numpy().flatten()]
        self._plot_bar(shap_values=shap_values.cohorts(clss).abs.mean(0),
                       max_display=max_display,
                       title=F"Mean absolute values, split by classes, " + SUFFIX,
                       file_export=file_export_12)

        # 1.3 Finding the optimal separation criterion for splitting the data
        # into 2 feature importance groups.
        file_export_13 = os.path.join(self.dir_export,
                                      "13_bar_optimal_split.png") if not show \
            else None
        # Remove the ugly 10-decimals formatting from the splitting criterion.
        cohorts = shap_values[..., None].cohorts(2).abs.mean(0)
        cohorts.cohorts = self._set_cohort_decimals(
            dict_cohort=cohorts.cohorts)
        self._plot_bar(
            shap_values=cohorts,
            max_display=max_display,
            title=F"Mean absolute values, optimal separation criterion" + SUFFIX,
            file_export=file_export_13)

        # 1.4 Beeswarm plot.
        shap.plots.beeswarm(shap_values,
                            max_display=max_display,
                            show=False)
        plt.title(F"Bee swarm" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_14 = os.path.join(self.dir_export, "14_beeswarm.png")
            plt.savefig(file_export_14)

        # 1.5 Violin plot.
        shap.plots.violin(shap_values,
                          max_display=max_display,
                          show=False)
        plt.title(F"Violin swarm" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_15 = os.path.join(self.dir_export, "15_violin.png")
            plt.savefig(file_export_15)

        # 1.6 Decision plot.
        shap.plots.decision(base_value=expected_value,
                            shap_values=shap_arr,
                            feature_order="importance",
                            features=data,
                            feature_names=feature_names,
                            feature_display_range=range(shap_arr.shape[1],
                                                        shap_arr.shape[
                                                            1] - max_display,
                                                        -1),
                            show=False)
        plt.title(F"Decision plot" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_16 = os.path.join(self.dir_export,
                                          "16_decision_all.png")
            plt.savefig(file_export_16)

        # 1.7 Heat map.
        shap.plots.heatmap(shap_values,
                           max_display=max_display,
                           show=False)
        plt.title(F"Heat map" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_17 = os.path.join(self.dir_export,
                                          "17_heatmap.png")
            plt.savefig(file_export_17)

        # 1.8 Scatter plot.
        shap.plots.scatter(shap_values=shap_values[:, shap_values.abs.mean(
            0).argsort[-1]],
                           color=shap_values,
                           show=False)
        plt.title(
            F"Scatter plot for the most important feature" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_18 = os.path.join(self.dir_export,
                                          "18_scatter_plot.png")
            plt.savefig(file_export_18)

        ###
        # Single example.

        # 2.1 Single example ranked bar plot absolute values.
        file_export_21 = os.path.join(self.dir_export,
                                      "21_bar_single.png") if not show else None
        self._plot_bar(shap_values=shap_values[example_idx],
                       max_display=max_display,
                       title=F"Shap values, single example" + SUFFIX,
                       file_export=file_export_21)

        # 2.2 Force plot.
        shap.plots.force(base_value=expected_value,
                         shap_values=shap_arr[example_idx],
                         features=np.round(data[example_idx], 2),
                         feature_names=feature_names,
                         matplotlib=True,
                         show=False)
        plt.title(
            F"Force plot, single example" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_22 = os.path.join(self.dir_export,
                                          "22_force_single.png")
            plt.savefig(file_export_22)

        # 2.3 Waterfall plot.
        try:
            shap.plots.waterfall(shap_values=shap_values[example_idx],
                                 show=False)
            plt.title(
                F"Waterfall plot, single example" + SUFFIX)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                file_export_23 = os.path.join(self.dir_export,
                                              "23_waterfall_single.png")
                plt.savefig(file_export_23)
        except IndexError:
            warnings.warn("Weird waterfall plot error arose. Skipping.")

        # 2.4 Decision plot.
        shap.plots.decision(base_value=expected_value,
                            shap_values=shap_arr[example_idx],
                            feature_order="importance",
                            features=data[example_idx],
                            feature_names=feature_names,
                            feature_display_range=range(shap_arr.shape[1],
                                                        shap_arr.shape[
                                                            1] - max_display,
                                                        -1),
                            show=False)
        plt.title(
            F"Decision plot, single example" + SUFFIX)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            file_export_24 = os.path.join(self.dir_export,
                                          "24_decision_single.png")
            plt.savefig(file_export_24)

        return

    def clustering(self,
                   cutoff: float,
                   example_idx: int,
                   show: bool):
        """
        Encapsulating the feature clustering visualization since it is VERY
        time intensive.
        """
        # Summary.
        SUFFIX = F", {self.tvt}"

        clustering = shap.utils.hclust(self.X, self.Y)
        shap.plots.bar(shap_values=self.shap_obj,
                       clustering=clustering,
                       clustering_cutoff=cutoff,
                       show=False)
        plt.title(F"Clustering feature importance" + SUFFIX)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            file_export_all = os.path.join(self.dir_export,
                                           "clustering_all.png")
            plt.savefig(file_export_all)

        # Single example.
        shap.plots.bar(shap_values=self.shap_obj[example_idx],
                       clustering=clustering,
                       clustering_cutoff=cutoff,
                       show=False)
        plt.title(F"Clustering feature importance, single example" + SUFFIX)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            file_export_single = os.path.join(self.dir_export,
                                              "clustering_single.png")
            plt.savefig(file_export_single)

        return

    def feature_importances(self,
                            max_display: int,
                            show: bool):
        """
        Displaying the feature importances for a tree-based model.
        """
        if "logistic_regression" in self.model_type:
            warnings.warn(
                "Feature importances not supported for logistic regression.")
            return
        importances = self.model.model.feature_importances_
        df_importances = pd.DataFrame({"feature_name": self.feature_names,
                                       "importance": importances})
        df_importances.sort_values(by="importance", ascending=False,
                                   inplace=True)

        print("Feature importances, sorted:")
        print(df_importances.reset_index().to_string())

        plt.figure(figsize=(8, 5))
        plt.barh(df_importances['feature_name'][:max_display],
                 df_importances['importance'][:max_display], )
        plt.xlabel("Importance")
        plt.ylabel("Feature name")
        plt.gca().invert_yaxis()
        plt.title("Feature importances")
        plt.tight_layout()

        if show:
            plt.show()
        else:
            file_export = os.path.join(self.dir_export,
                                       "feature_importances.png")
            plt.savefig(file_export)

        return

    def feature_rankings(self,
                         show: bool = False):
        mean_shaps = np.mean(np.abs(self.shap_arr), axis=0)
        df_rankings = pd.DataFrame({"feature_name": self.feature_names,
                                    "mean_shap": mean_shaps})
        df_rankings = df_rankings.sort_values(by="mean_shap")[
                      ::-1].reset_index(drop=True)

        if show:
            print("Feature rankings, sorted:")
            print(df_rankings.to_string())

        return df_rankings

    @staticmethod
    def get_all_salient_features(file_task: str,
                                 model_id: str,
                                 fraction_total_shap: float):
        """
        Displays all salient features per train, val, test, and highlights
        the discrepancies in salient features between the subsets.
        """
        TVT = ["train", "val", "test"]

        df_rankings_tvt = {}
        salient_features_tvt = {}

        n_features = None

        for tvt in TVT:
            sr = SHAPRadiomics(file_task=file_task,
                               model_id=model_id,
                               tvt=tvt,
                               summation_mode="single")
            df_rankings = sr.feature_rankings(show=False)
            salient_features = df_rankings.loc[df_rankings["mean_shap"] > 0,
            "feature_name"].tolist()

            df_rankings_tvt[tvt] = df_rankings
            salient_features_tvt[tvt] = salient_features

            if n_features is None:
                n_features = df_rankings.shape[0]

        # Check whether the salient features are the same for all subsets.
        salient_features_all = list(set([feature for tvt in TVT for feature in
                                         salient_features_tvt[tvt]]))
        if len([salient_features_tvt[tvt] for tvt in TVT if
                len(set(salient_features_tvt[tvt]).difference(
                    salient_features_all)) > 0]):
            warnings.warn("The salient features for train, val, test are not "
                          "identical!")

        df_rankings_all = pd.DataFrame({"feature_name": salient_features_all})
        for tvt in TVT:
            df_rankings_all[F"mean_shap_{tvt}"] = df_rankings_tvt[
                tvt].set_index("feature_name").loc[
                salient_features_all, "mean_shap"].to_numpy()

        df_rankings_all = df_rankings_all.sort_values(by="mean_shap_train")[
                          ::-1].reset_index(drop=True)

        df_rankings_modality = SHAPRadiomics._sum_SHAP_per_modality(
            df_rankings=df_rankings_all)

        # Computing the cutoff where a total fraction of SHAP values has been
        # reached, for training a linear regression model.
        shap_vals = df_rankings_tvt["train"]["mean_shap"].to_numpy()
        total_shap = np.sum(shap_vals)
        shap_cutoff = fraction_total_shap * total_shap
        # The first feature for which the cumulative SHAP values up to this
        # point are greater than the desired fraction of the whole.
        cutoff_idx = np.min(np.where(np.cumsum(shap_vals) > shap_cutoff)[0])
        print(F"For the fraction of total SHAP of {fraction_total_shap}, "
              F"the cutoff index is {cutoff_idx}.")

        print(F"Feature rankings, split by train, val, test:")
        print(df_rankings_all.to_string())

        print(F"\nSummed feature rankings for each modality, split by train, "
              F"val, test:")
        print(df_rankings_modality.to_string())

        salient_features_str = "[\"" + "\", \"".join(
            df_rankings_all["feature_name"].tolist()[:cutoff_idx + 1]) + "\"]"
        print(F"\nThe {cutoff_idx + 1} first salient features as a portable "
              F"list:")
        print(salient_features_str)

        # Visualizing the gradient of SHAP values.
        fig, ax = plt.subplots(1)
        ax.bar(np.arange(n_features),
               shap_vals,
               color=SHAPRadiomics.SHAP_RED)
        ax.axvline(cutoff_idx,
                   color=SHAPRadiomics.SHAP_BLUE,
                   linewidth=4,
                   label=F"{fraction_total_shap} quantile")
        ax.set_title(F"Ordered SHAP values on the train dataset")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("SHAP value")
        ax.legend()

        plt.show()

        return df_rankings_all

    @staticmethod
    def _sum_SHAP_per_modality(df_rankings: pd.DataFrame):
        """
        Sums up the SHAP values for all features of each modality.
        """
        TVT = ["train", "val", "test"]
        MODALITIES = ["T1_POST_DYNAMIC", "STIR", "ADC", "PET"]
        features_per_modality = {}

        for modality in MODALITIES:
            features_modality = [feature for feature in
                                 df_rankings["feature_name"].to_numpy() if
                                 modality.lower() in feature.lower()]
            features_per_modality[modality] = features_modality

        # Sanity check.
        assert sum(
            [len(features_per_modality[modality]) for modality in
             MODALITIES]) == df_rankings.shape[
                   0], ("Some features are either not assigned to a modality "
                        "or assigned more than once!")

        sum_shap_per_modality_tvt = {}
        for tvt in TVT:
            sum_shap_per_modality_tvt[tvt] = {}
            for modality in MODALITIES:
                sum_shap_per_modality_tvt[tvt][modality] = np.sum(
                    df_rankings.set_index(
                        "feature_name").loc[
                        features_per_modality[
                            modality], F"mean_shap_{tvt}"].to_numpy())

        df_rankings_modality = pd.DataFrame({"modality": MODALITIES})
        for tvt in TVT:
            df_rankings_modality[F"sum_shap_{tvt}"] = [
                sum_shap_per_modality_tvt[tvt][modality] for modality in
                MODALITIES]

        return df_rankings_modality
