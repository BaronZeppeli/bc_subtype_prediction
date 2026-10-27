import os
import json


class Task:
    TVT = ["train", "val", "test"]

    def __init__(self,
                 file_task: str):
        with open(file_task, 'r') as f:
            self.dict_dataset = json.load(f)

        self.file = file_task
        self.name = self.dict_dataset["name"]
        # Make sure the modalities are sorted by their keys, not their order!!
        modality_keys = sorted(map(int, self.dict_dataset["modality"].keys()))
        self.modalities = [self.dict_dataset["modality"][str(modality_key)]
                           for modality_key in modality_keys]
        self.n_modalities = len(self.modalities)

        self.n_labels = len(self.dict_dataset["labels"].keys())
        self.n_classes_per_label = [len(self.dict_dataset["labels"][key_label][
                                            "classes"].keys()) for key_label in
                                    self.dict_dataset["labels"].keys()]

        self.dir_base = self.dict_dataset["dirBase"]
        self.dir_preprocessed = self.dict_dataset["dirPreprocessed"]
        self.dir_task = os.path.join(self.dir_preprocessed, self.name)
        self.dir_models = os.path.join(self.dir_task, "models")
        self.dir_explainability = os.path.join(self.dir_task, "explainability")
        self.dir_images = os.path.join(self.dir_task, "images")
        self.dir_whole_body = os.path.join(self.dir_task, "whole_body")
        self.dir_masks = os.path.join(self.dir_task, "masks")

        self.dir_radiomics = os.path.join(self.dir_task, "radiomics")
        self.dir_radiomics_input = os.path.join(self.dir_radiomics, "input")
        self.dir_radiomics_models = os.path.join(self.dir_radiomics, "models")
        self.dir_radiomics_inference = os.path.join(self.dir_radiomics,
                                                    "inference")
        self.dir_radiomics_explainability = os.path.join(self.dir_radiomics,
                                                         "explainability")

        return

    def get_file_radiomics(self,
                           labels: bool,
                           tvt: str):
        """
        Returns either the input or labels radiomics CSV file for a current
        project.
        """
        prefix = "labels" if labels else "radiomics"
        file_radiomics = F"{prefix}_{self.name}_{tvt}.csv"
        file_radiomics = os.path.join(self.dir_radiomics_input, file_radiomics)

        return file_radiomics

    def get_patient_ids(self):
        """
        Returns all patient names, split by train/val/test.
        """
        dict_patient_ids = {}
        for tvt in self.TVT:
            dict_patient_ids[tvt] = []
            for patient in self.dict_dataset[tvt]:
                patient_id = os.path.basename(patient["image"])
                dict_patient_ids[tvt] += [patient_id]

        return dict_patient_ids

    def get_patient_label(self,
                          patient_id: str,
                          tvt: str):
        """
        Returns the label of a patient of either the train, val, or test set.
        """
        for patient in self.dict_dataset[tvt]:
            p_id = os.path.basename(patient["image"])
            if patient_id == p_id:
                label = patient["labels"]
                break
        else:
            raise ValueError(F"Patient {patient_id} not in {tvt} dataset!")

        return label

    def channel_couple_to_str(self,
                              channel_couple: list):
        """
        Returns a consistent string representation of the channel couple.
        """
        included_modalities = [self.modalities[idx] for idx in channel_couple]
        str_included_modalities = "_".join(included_modalities)

        return str_included_modalities

    def str_to_channel_couple(self,
                              str: str):
        """
        Returns the channel couple of a string representation.
        """
        channel_couple = []

        segments = str.split("_")
        while len(segments) > 0:
            modality_idx = None
            for segment_idx in range(len(segments)):
                modality_str = "_".join(segments[:segment_idx + 1])
                if modality_str in self.modalities:
                    modality_idx = [i for i, modality in enumerate(
                        self.modalities) if modality == modality_str][0]
                    break

            assert modality_idx is not None, "No matching modality found!"
            channel_couple += [modality_idx]

            segments = segments[segment_idx + 1:]

        return channel_couple
