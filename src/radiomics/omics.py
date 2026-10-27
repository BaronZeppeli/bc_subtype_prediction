from abc import ABC, abstractmethod
from xml.etree.ElementTree import XMLParser

import numpy as np
import pandas as pd
import json

from tqdm import tqdm


class AbstractLoader(ABC):
    COLUMNS_RAW = ["Sample", "Subtype"]
    COLUMNS_PREPROCESSED = ["ID", "Center?"]

    @abstractmethod
    def __init__(self):
        self.df = None

    @staticmethod
    @abstractmethod
    def preprocess(*args):
        pass

    def __getitem__(self, key):
        return self.df[self.df["ID"] == key]

    def __setitem__(self, key, value):
        """
        Prevent the setting of elements.
        """
        return

    def __delitem__(self, key):
        """
        Prevent the deletion of elements.
        """
        return

    def __contains__(self, key):
        return key in self.df["ID"].to_list()


class MetabolomicsLoader(AbstractLoader):

    @staticmethod
    def preprocess(file_metabolomics: str,
                   file_id_conversion: str,
                   file_export: str):
        """
        Increases the readability of the metabolomics measurement file.
        """
        df_metabolomics = pd.read_csv(file_metabolomics)
        df_id_conversion = pd.read_csv(file_id_conversion)

        columns_metabolite = df_metabolomics.columns.difference(
            AbstractLoader.COLUMNS_RAW)

        ids_master = df_metabolomics["Sample"].apply(lambda sample:
                                                     sample.split("MaCa_P")[
                                                         1].split("_")[
                                                         0]).astype(int)
        ids_tue = ids_master.apply(lambda id_master: df_id_conversion.loc[
            df_id_conversion["ID_Master"] == id_master, "ID_Tue"].iloc[0])
        center = df_metabolomics["Sample"].apply(lambda sample:
                                                 sample.split("_T")[1][
                                                     0].lower() ==
                                                 "z").astype(int)
        df_metabolomics_preprocessed = pd.DataFrame({"ID": ids_tue,
                                                     "Center?": center})
        df_metabolomics_preprocessed[columns_metabolite] = df_metabolomics[
            columns_metabolite]

        assert (df_metabolomics_preprocessed[
                    columns_metabolite] == df_metabolomics[
                    columns_metabolite]).all(axis=None), ("Something went "
                                                          "wrong during "
                                                          "preprocessing of "
                                                          "the metabolite "
                                                          "measurement file.")

        # Sort the values by their TUE index.
        df_metabolomics_preprocessed = df_metabolomics_preprocessed.sort_values(
            by=AbstractLoader.COLUMNS_PREPROCESSED,
            inplace=False).reset_index(drop=True)

        df_metabolomics_preprocessed.to_csv(file_export, index=False)

        return

    def __init__(self,
                 file_metabolomics: str,
                 file_metabolites_salient: str,
                 column_salient: str):
        df_metabolomics = pd.read_csv(file_metabolomics)
        df_metabolites_salient = pd.read_csv(file_metabolites_salient)

        # Remove all non-salient metabolites.
        columns_metabolites_nonsalient = df_metabolites_salient.loc[
            df_metabolites_salient[column_salient] == 0,
            "metabolite"].to_list()
        self.df = df_metabolomics.drop(
            columns_metabolites_nonsalient, axis=1)

        return


class GenomicsLoader(AbstractLoader):

    @staticmethod
    def preprocess(file_genomics: str,
                   file_sample_conversion: str,
                   file_id_conversion: str,
                   file_export: str):
        """
        Increases the readability of the metabolomics measurement file.
        """
        # The separator has to be changed since the files exported from
        # Excel.
        df_genomics = pd.read_csv(file_genomics, sep=";")
        df_sample_conversion = pd.read_csv(file_sample_conversion, sep=";")
        df_id_conversion = pd.read_csv(file_id_conversion)

        # VERY IMPORTANT NOTE: The genomics measurements are saved using
        # column-store!
        columns_gene = df_genomics["Symbol"].to_list()
        df_genomics_rowstore = df_genomics.T.reset_index()
        # Rename the columns and drop the first row containing only the
        # designated names of columns.
        df_genomics_rowstore.columns = df_genomics_rowstore.iloc[0].to_list()
        # The first column is renamed only for code readability.
        df_genomics_rowstore = df_genomics_rowstore.rename(columns={"Symbol":
                                                                        "Sample"})
        df_genomics_rowstore = df_genomics_rowstore.iloc[1:].reset_index(
            drop=True)

        df_id_master_region = df_genomics_rowstore["Sample"].apply(
            lambda sample:
            df_sample_conversion.loc[
                df_sample_conversion[
                    "Sample "
                    "Replacement"] == sample, [
                    "Patient ID",
                    "Tumor Areal"]].iloc[
                0])
        # Same as for the metabolomics data.
        ids_tue = df_id_master_region.iloc[:, 0].astype(int).apply(lambda
                                                                       id_master:
                                                                   df_id_conversion.loc[
                                                                       df_id_conversion[
                                                                           "ID_Master"] == id_master, "ID_Tue"].iloc[
                                                                       0])

        center = df_id_master_region.iloc[:, 1].apply(lambda sample:
                                                      sample == "TZ").astype(
            int)
        df_genomics_preprocessed = pd.DataFrame({"ID": ids_tue,
                                                 "Center?": center})
        df_genomics_preprocessed = pd.concat([df_genomics_preprocessed,
                                              df_genomics_rowstore[
                                                  columns_gene]], axis=1)
        assert np.allclose(
            df_genomics_preprocessed[columns_gene].T.to_numpy().astype(float),
            df_genomics.iloc[:, 1:].to_numpy().astype(float),
            rtol=1e-05), \
            ("Something went wrong during preprocessing of the gene "
             "measurement "
             "file.")

        # Sort the values by their TUE index.
        df_genomics_preprocessed = df_genomics_preprocessed.sort_values(
            by=AbstractLoader.COLUMNS_PREPROCESSED,
            inplace=False).reset_index(drop=True)

        df_genomics_preprocessed.to_csv(file_export, index=False)

        return

    def __init__(self,
                 file_genomics: str,
                 file_hallmarks_salient: str,
                 file_hallmarks_counts: str,
                 column_salient: str):
        df_genomics = pd.read_csv(file_genomics)
        df_hallmarks_salient = pd.read_csv(file_hallmarks_salient)
        df_hallmarks_counts = pd.read_csv(file_hallmarks_counts)

        # Basic logic: The salient genes are all those measured genes which
        # are present in at least 1 salient hallmark.
        salient_hallmarks = df_hallmarks_salient.loc[
            df_hallmarks_salient[column_salient] == 1, "hallmark"].to_list()
        salient_genes = df_hallmarks_counts.loc[df_hallmarks_counts[
                                                    salient_hallmarks].sum(
            axis=1) > 0, "gene"].to_list()

        self.df = df_genomics[
            AbstractLoader.COLUMNS_PREPROCESSED + salient_genes]

        return


class GenomicsAnalysis:
    """
    Class for computing the distribution of hallmarks across the given
    measurement.
    """

    def __init__(self,
                 file_genomics: str,
                 file_hallmarks: str):
        self.df_genomics = pd.read_csv(file_genomics)
        with open(file_hallmarks, 'r') as file:
            self.dict_hallmarks = json.load(file)

        # List comprehension ensures the consistency of the gene order (which
        # is not quite alphabetic.)
        self.genes_genomics = [column for column in self.df_genomics.columns if
                               column
                               not in AbstractLoader.COLUMNS_PREPROCESSED]

        self.hallmarks = sorted(self.dict_hallmarks.keys())
        self.df_hallmarks = self.create_hallmark_counts(file_export=None)

        return

    def create_hallmark_stats(self,
                              file_export: str):
        df_stats = pd.DataFrame(columns=["hallmark",
                                         "n_genes_total",
                                         "n_genes_present",
                                         "ratio_present_total"])
        for hallmark in self.hallmarks:
            genes_hallmark = self.dict_hallmarks[hallmark]["geneSymbols"]
            n_genes_present = len(
                [gene for gene in genes_hallmark if gene in
                 self.genes_genomics])
            ratio_genes_present = n_genes_present / len(genes_hallmark)
            row_hallmark = pd.DataFrame({"hallmark": hallmark,
                                         "n_genes_total": len(genes_hallmark),
                                         "n_genes_present": n_genes_present,
                                         "ratio_present_total":
                                             ratio_genes_present},
                                        index=[0])
            df_stats = pd.concat([df_stats, row_hallmark],
                                 ignore_index=True)
        df_stats["ratio_present_all_measurements"] = [self.df_hallmarks[
                                                          hallmark].sum() / len(
            self.genes_genomics)
                                                      for
                                                      hallmark in
                                                      self.hallmarks]
        df_stats = df_stats.sort_values(by=["ratio_present_total"]).iloc[
                   ::-1].reset_index(drop=True)
        print(df_stats.to_string())

        df_stats = df_stats.sort_values(by=[
            "ratio_present_all_measurements"]).iloc[
                   ::-1].reset_index(drop=True)
        print(df_stats.to_string())

        genes_without_hallmark = self.df_hallmarks[
            self.df_hallmarks[self.hallmarks].sum(axis=1) == 0]
        print("Genes not present in any hallmarks: ",
              genes_without_hallmark.shape[0])

        if file_export is not None:
            df_stats = df_stats.sort_values(by=[
                "hallmark"]).iloc[
                       ::-1].reset_index(drop=True)
            df_stats.to_csv(file_export, index=False)
        return

    def create_hallmark_counts(self,
                               file_export: str):
        """
        Internal helper function, but can also be called externally.
        """

        df_hallmarks = pd.DataFrame(columns=["gene"] + self.hallmarks)

        for gene_genomics in tqdm(self.genes_genomics):
            row_gene = {"gene": gene_genomics}
            for hallmark in self.hallmarks:
                genes_hallmark = self.dict_hallmarks[hallmark]["geneSymbols"]
                row_gene[hallmark] = gene_genomics in genes_hallmark
            df_hallmarks = pd.concat([df_hallmarks, pd.DataFrame(row_gene,
                                                                 index=[0])],
                                     ignore_index=True)

        # Convert the type of the contains-information into an integer.
        df_hallmarks = df_hallmarks.astype({hallmark: int for
                                            hallmark in
                                            self.hallmarks})
        if file_export is not None:
            df_hallmarks.to_csv(file_export, index=False)

        return df_hallmarks


class Radiomics(ABC):
    """
    Radiomic features implemented based on Aerts et al. (2013). Selection of
    features based on Oliveira et al. (2024).
    """

    N_BINS = 10
    # Following the PERCIST-guidelines, a 6mm radius sphere is used to
    # determine the region with the largest mean uptake, by moving the center
    # of the sphere to each voxel in the ROI.
    RADIUS_PEAK = 6
    MOVE_SPHERE = False
    # Decided more or less arbitrarily.
    ATHRESHOLD_MTV = 2.0
    FEATURE_NAMES = None

    def __init__(self,
                 X: np.ndarray,
                 idxs_X: np.ndarray,
                 volume: np.ndarray, ):
        self.X = X
        self.idxs_X = idxs_X
        self.volume = volume

    @staticmethod
    def get_method(feature_name: str):
        """
        Metaprogramming to allow subclasses to specify the desired features
        via strings.
        """
        fn = getattr(Radiomics, "get_" + feature_name)
        return fn

    def get_features(self,
                     verbose: bool = True):
        """
        Returns the features specified by a derived subtype.
        """
        features = []
        if verbose:
            loop_obj= tqdm(self.FEATURE_NAMES,desc="Computing radiomics")
        else:
            loop_obj = self.FEATURE_NAMES
        for feature_name in loop_obj:
            # Obviously an alternative to this approach would have been to
            # ensure a structural conformity of the signatures of all getter
            # functions, but this would lead to a lot of nonfunctional code.
            if feature_name == "peak":
                feature = self.get_method(feature_name)(self.idxs_X,
                                                        self.volume,
                                                        self.RADIUS_PEAK,
                                                        self.MOVE_SPHERE)
            elif (feature_name == "entropy") | (feature_name == "uniformity"):
                feature = self.get_method(feature_name)(self.X,
                                                        self.N_BINS)
            elif (feature_name == "metabolic_tumor_volume") | (
                    feature_name == "total_lesion_glycolysis"):
                feature = self.get_method(feature_name)(self.X,
                                                        self.ATHRESHOLD_MTV)
            else:
                feature = self.get_method(feature_name)(self.X)

            features += [feature]

        return features

    def get_feature_names(self):
        return self.FEATURE_NAMES

    @staticmethod
    def get_energy(X):
        return np.sum(np.power(X, 2))

    @staticmethod
    def get_entropy(X,
                    n_bins: int):
        probability_per_bin = Radiomics.get_probability_per_bin(X=X,
                                                                n_bins=n_bins)
        # Avoid log errors induced by inserting 0 by adding a small constant.
        entropy = np.sum(probability_per_bin * np.log2(
            probability_per_bin + 1e-10))

        return entropy

    @staticmethod
    def get_kurtosis(X):
        # Split the computation into numerator and denominator of the
        # fraction, for increased readability.
        # NOTE: This corresponds to Pearson's kurtosis. The definition in
        # Aerts et al. is wrong!!!
        numerator = Radiomics.get_central_moment(X=X,
                                                 order=4)
        denominator = np.power(Radiomics.get_standard_deviation(X), 4)

        # Avoid dividing by 0 errors.
        return numerator / (denominator + 1e-10)

    @staticmethod
    def get_skewness(X):
        numerator = Radiomics.get_central_moment(X=X,
                                                 order=3)
        denominator = np.power(Radiomics.get_standard_deviation(X), 3)

        # Avoid dividing by 0 errors.
        return numerator / (denominator + 1e-10)

    @staticmethod
    def get_maximum(X):
        return np.max(X)

    @staticmethod
    def get_minimum(X):
        return np.min(X)

    @staticmethod
    def get_mean(X):
        return np.mean(X)

    @staticmethod
    def get_median(X):
        return np.median(X)

    @staticmethod
    def get_range(X):
        return np.max(X) - np.min(X)

    @staticmethod
    def get_mean_absolute_deviation(X):
        absolute_deviation = np.abs(X - np.mean(X))
        return np.mean(absolute_deviation)

    @staticmethod
    def get_root_mean_square(X):
        return np.sqrt(Radiomics.get_energy(X) / X.shape[0])

    @staticmethod
    def get_uniformity(X,
                       n_bins: int):
        return np.sum(
            np.power(Radiomics.get_probability_per_bin(X=X, n_bins=n_bins), 2))

    @staticmethod
    def get_central_moment(X,
                           order: int):
        return np.sum(np.power(X - np.mean(X), order)) / X.shape[0]

    @staticmethod
    def get_standard_deviation(X):
        # IMPORTANT: Aerts et al. use 1 as delta degree of freedom for std as
        # a feature, in contrast to the other central moments!!
        return np.sqrt(Radiomics.get_variance(X))

    @staticmethod
    def get_variance(X):
        return Radiomics.get_central_moment(X=X, order=2)

    @staticmethod
    def get_coefficient_of_variation(X):
        return Radiomics.get_standard_deviation(X) / Radiomics.get_mean(X)

    @staticmethod
    def get_probability_per_bin(X,
                                n_bins: int):
        bin_counts, bin_edges = np.histogram(X, bins=n_bins)
        probability_per_bin = bin_counts / X.shape[0]

        return probability_per_bin

    @staticmethod
    def get_peak(idxs: np.ndarray,
                 volume: np.ndarray,
                 radius: int,
                 move_sphere: bool):
        # IMPORTANT: THIS ONLY WORKS IF THE VOLUME IS ISOTROPIC AND VOXEL
        # SIZE 1 mm!!!
        # Determine the 3D index for each voxel of the target volume.
        volume_idxs = np.einsum("ijkl->jkli", np.indices(volume.shape[
                                                         :3])).reshape((-1, 3))
        if move_sphere:
            peak = 0
            for idx in tqdm(idxs, desc="Computing SUV peak"):
                radius_idxs = volume_idxs[
                    np.linalg.norm(volume_idxs - idx, axis=1) <= radius]
                assert radius_idxs.shape[0] <= np.power(radius * 2, 3), \
                    (
                        "Somehow, the computation of the sphere around a voxel failed.")
                # Reshape the indices into the shape of output of np.where,
                # to ensure that the correct indices are being accessed,
                # instead on relying on the order of volume indices being
                # compatible with the flattened order of elements in the volume.
                sphere_vals = volume[
                    radius_idxs[:, 0],
                    radius_idxs[:, 1],
                    radius_idxs[:, 2]]
                mean_sphere = np.mean(sphere_vals)
                if mean_sphere > peak:
                    peak = mean_sphere
        else:
            # More efficient alternate way of computing the
            # SUV peak by simply centering the sphere around the voxel of SUV max.
            idx_suv_max = idxs[np.argmax(volume[
                                             idxs[:, 0],
                                             idxs[:, 1],
                                             idxs[:, 2]
                                         ].flatten())]
            radius_idxs = volume_idxs[
                np.linalg.norm(volume_idxs - idx_suv_max, axis=1) <= radius]
            assert radius_idxs.shape[0] <= np.power(radius * 2, 3), \
                (
                    "Somehow, the computation of the sphere around a voxel failed.")
            peak = np.mean(volume[
                               radius_idxs[:, 0],
                               radius_idxs[:, 1],
                               radius_idxs[:, 2]])
            pass

        return peak

    @staticmethod
    def get_tumor_volume(X):
        """
        NOTE: This method requires X to be isotropic!
        """
        return X.shape[0]

    @staticmethod
    def get_metabolic_tumor_volume(X,
                                   athreshold: float):
        """
        NOTE: This method requires X to be isotropic!
        """
        return np.sum(X > athreshold)

    @staticmethod
    def get_total_lesion_glycolysis(X,
                                    athreshold: float):
        """
        NOTE: This method requires X to be isotropic!
        """
        return (Radiomics.get_metabolic_tumor_volume(X=X,
                                                     athreshold=athreshold)
                * Radiomics.get_mean(X))


class RadiomicsPET(Radiomics):
    FEATURE_NAMES = ["maximum", "peak", "mean", "median", "standard_deviation",
                     "coefficient_of_variation", "energy", "entropy",
                     "uniformity", "skewness", "kurtosis", "range",
                     "metabolic_tumor_volume", "total_lesion_glycolysis"]

    def __init__(self,
                 X: np.ndarray,
                 idxs_X: np.ndarray,
                 volume: np.ndarray):
        assert X.shape[0] == idxs_X.shape[0], ("The data and the indices must "
                                               "have the same length.")
        assert (idxs_X.ndim == 2) & (idxs_X.shape[1] == 3), ("The indices "
                                                             "must be a 2D "
                                                             "array of 3D "
                                                             "positions.")
        Radiomics.__init__(self,
                           X=X,
                           idxs_X=idxs_X,
                           volume=volume
                           )


class RadiomicsMRI(Radiomics):
    FEATURE_NAMES = ["maximum", "mean", "median", "standard_deviation",
                     "coefficient_of_variation", "energy", "entropy",
                     "uniformity", "skewness", "kurtosis", "range",
                     "tumor_volume"]

    def __init__(self, X):
        Radiomics.__init__(self,
                           X=X,
                           idxs_X=None,
                           volume=None,
                           )
