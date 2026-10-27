import os

from typing import List, TypeVar, Optional, Tuple, Union, Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.errors import ParserError

import h5py
from pydicom.errors import InvalidDicomError
import nibabel as nib
from nibabel.filebasedimages import ImageFileError
from PIL import Image, UnidentifiedImageError

# Row Spacing / Column Spacing / Slice Spacing.
import handler_nifti

DEFAULT_ORIENTATION = "axial"
DEFAULT_VOXEL_SIZE: np.ndarray = np.array([0.30, 0.30, 0.30])
DEFAULT_COORDINATE_COLUMNS: List[str] = ['x', 'y', 'z']
# y,x,z, in accordance with the order of DICOM coordinates.
DEFAULT_MIN_VALUES: np.ndarray = np.array([0, 0, 0])

S1 = TypeVar('S1', np.ndarray, str, pd.DataFrame)
F = TypeVar('F', list, str)


class VoxelArray:
    """
    The class representing the array containing all the information of
    a .csv file, in addition to the according metadata.
    """

    def __init__(self,
                 source_voxel: S1,
                 volume_info_file: str = None,
                 metadata: Optional[S1] = None,
                 name: Optional[str] = None,
                 channels: Optional[List[str]] = None,
                 voxel_size: Optional[List[float]] = None,
                 coordinate_columns: List[str] = None,
                 acquisition_time: Optional[List[float]] = None,
                 min_values: Optional[List[int]] = None,
                 weight=None,
                 injected_dose=None,
                 decay_factor=None,
                 affine=None
                 ) -> None:

        array = None
        voxel_df = None
        # Variables to define the smallest values,
        # for the dataframe conversion.
        min_x_value = None
        min_y_value = None
        min_z_value = None
        if min_values is not None:
            if len(min_values) != 3:
                raise ValueError("Exactly 3 values must be provided for the "
                                 "min values per dimension.")
            min_x_value = min_values[1]
            min_y_value = min_values[0]
            min_z_value = min_values[2]

        ########################################################################
        if isinstance(source_voxel, str):
            # DICOM DIRECTORY
            if os.path.isdir(source_voxel):
                try:
                    array, name, channels, voxel_size, acquisition_time, weight, \
                        injected_dose, decay_factor, metadata, \
                        n_channels_per_volume_dicom, affine, roi_names_per_modality = \
                        handler_dicom.fuse_and_convert_DICOM_NIFTI(
                            volume_paths=source_voxel,
                            output_path=None,
                            voxel_size=None,
                            method="linear",
                            fill_value=None)
                except OSError:
                    raise ValueError("The specified directory cannot be read "
                                     "in as a set of DICOM files.")
            else:
                # DICOM FILE
                try:
                    array, name, channels, voxel_size, acquisition_time, \
                        weight, injected_dose, decay_factor, metadata, affine = \
                        handler_dicom.load_dicom_file(source_voxel)
                except InvalidDicomError:
                    # NIFTI FILE
                    try:
                        nifti_file = nib.load(source_voxel)
                        array_proxy = nifti_file.get_fdata()
                        array = np.copy(array_proxy)
                        affine = np.copy(nifti_file.affine)
                        header = nifti_file.header.copy()

                        # MAKES URE TO DELETE ALL NIFTI IMAGE RELATED
                        # OBJECTS, IN ORDER TO ALLOW THE ORIGINAL NIFTI FILE
                        # TO BE OVERWRITTEN.
                        nifti_file.uncache()
                        del nifti_file
                        del array_proxy
                        # TODO: Remove all semblances of the custom affine
                        #  format.
                        # Make sure the array is 4-dimensional.
                        if array.ndim < 3:
                            array = array[None, ...]
                        if array.ndim < 4:
                            array = array[..., None]
                        # Ensure that the volume is aligned in axial
                        # orientation.
                        # And don't forget to decode the affine matrix!
                        array, affine = handler_nifti.standardize_orientation(
                            volume=array,
                            affine=affine,
                            orientation=DEFAULT_ORIENTATION,
                            decode=True)

                        # affine = handler_nifti.decode_affine(affine=affine,
                        #
                        #                                     encoded_array_shape=array.shape)
                        # Transform the array from the standard NIFTI shape
                        # to a numpy interpretable shape.
                        # array = handler_nifti.decode_array(arr=array)
                        voxel_size = to_voxel_array_converter.get_voxel_size_from_affine(
                            affine=affine)
                        # Make sure to copy the metadata by the amount of channels,
                        # i.e. every channel simply displays the same metadata.
                        metadata = [header] * array.shape[3]

                    except ImageFileError:
                        # FORMATTED CSV FILE
                        try:
                            voxel_df = pd.read_csv(source_voxel)

                        except (ParserError, UnicodeDecodeError):
                            # SIEMENS CSV FILE
                            try:
                                # The _ represents the ROI names, which are
                                # not used in this case.
                                array, affine, _, acquisition_time, \
                                    = convert_siemens_to_volume(
                                    siemens_file=source_voxel,
                                    mask_only=False,
                                    padding=0)
                            # AttributeError for .NPY files,
                            # UnicodeDecodeError for everything else.
                            except (AttributeError, UnicodeDecodeError):
                                # HDF FILE
                                try:
                                    with h5py.File(source_voxel,
                                                   'r') as in_file:
                                        for name_dataset in in_file.keys():
                                            dataset_voxel = in_file[
                                                name_dataset]
                                            array = dataset_voxel[:]
                                            if array.ndim <= 4:
                                                break
                                        else:
                                            raise ValueError(
                                                "The supplied .hdf "
                                                "file does not "
                                                "contain an array "
                                                "with ndim <= 4.")
                                        name = dataset_voxel.attrs.get('name')

                                        channels = dataset_voxel.attrs.get(
                                            'channels')
                                        if channels is not None:
                                            channels = list(channels)

                                        voxel_size = dataset_voxel.attrs.get(
                                            'voxel_size')
                                        acquisition_time = dataset_voxel.attrs.get(
                                            'acquisition_time')
                                        # LEGACY COMPATIBILITY channel.
                                        if acquisition_time is None:
                                            acquisition_time = dataset_voxel.attrs.get(
                                                'acquisition_times')

                                        metadata = dataset_voxel.attrs.get(
                                            'metadata')
                                        if metadata is not None:
                                            metadata = list(metadata)

                                        min_x_value = dataset_voxel.attrs.get(
                                            'min_x_value')
                                        min_y_value = dataset_voxel.attrs.get(
                                            'min_y_value')
                                        min_z_value = dataset_voxel.attrs.get(
                                            'min_z_value')

                                        weight = dataset_voxel.attrs.get(
                                            'weight')
                                        injected_dose = dataset_voxel.attrs.get(
                                            'injected_dose')
                                        decay_factor = dataset_voxel.attrs.get(
                                            'decay_factor')
                                        affine = dataset_voxel.attrs.get(
                                            'affine')

                                except OSError:
                                    # NPY FILE
                                    try:
                                        array = np.load(source_voxel)
                                    except ValueError:
                                        # MAT FILE
                                        try:
                                            array, channels, metadata = \
                                                convert_mat_to_voxel_array(
                                                    mat_file=source_voxel)
                                        except ValueError:
                                            try:
                                                array = np.array(Image.open(
                                                    source_voxel)).astype(float)
                                                if array.ndim == 2:
                                                    array = array[..., None]
                                                if array.ndim == 3:
                                                    array = array[None, ...]
                                            except UnidentifiedImageError:
                                                raise ValueError(
                                                    F"Source file {source_voxel} could not be read "
                                                    F"as either a .csv, .hdf, "
                                                    F".nii.gz or .npy file.")
        elif isinstance(source_voxel, pd.DataFrame):
            voxel_df = source_voxel.copy()
        elif isinstance(source_voxel, np.ndarray):
            array = source_voxel
        else:
            raise ValueError("The source for the voxel information for the "
                             "VoxelArray must either be a "
                             "string path, array or a dataframe.")
        ########################################################################

        self.coordinate_columns = coordinate_columns if \
            coordinate_columns is not None else DEFAULT_COORDINATE_COLUMNS
        # In case in some way a table was passed as source for the voxels.
        if voxel_df is not None:
            # The pad_to_same_length parameter here is very important,
            # which is why it is explicitly written out.
            min_x_value, min_y_value, min_z_value, array = \
                to_voxel_array_converter.convert_dataframe_into_4D_array(
                    df=voxel_df,
                    coordinate_columns=self.coordinate_columns,
                    pad_to_same_length=False,
                    reduce_to_minimal_size=True)
            channels = to_voxel_array_converter.extract_channel_columns(
                voxel_df, self.coordinate_columns)

            # Delete the voxel df to only ever explicitly store the array
            # form of the voxel array.
            del voxel_df
        if array.ndim < 4:
            self.array = array.reshape((1,) * (3 - array.ndim) + array.shape + (
                1,))
        else:
            self.array = array.reshape((array.shape[:3] + (-1,)))
        # Determine the name of the voxel array from the volume source,
        # if a file string was passed and no extra name was specified.
        if name is not None:
            self.name = name
        elif isinstance(source_voxel, str):
            self.name = source_voxel.split("\\")[-1].split("/")[-1].split(".")[
                0]
        else:
            self.name = ""
        self.channels = channels if channels is not None else [
            F"Channel_{i + 1}" for i in range(self.array.shape[3])]
        # In case the metadata parameter is a path to a metadata table,
        # call that table to determine the metadata of the voxel array.
        if (metadata is None) | (isinstance(metadata, str)):
            self.metadata = to_voxel_array_converter.get_metadata_for_tumor(
                channel_columns=self.channels,
                source_metadata=metadata)
        else:
            self.metadata = metadata

        self.voxel_size = np.array(voxel_size).astype(
            float) if voxel_size is not None else \
            DEFAULT_VOXEL_SIZE
        self.acquisition_time = np.array(acquisition_time).astype(float) \
            if acquisition_time is not None else np.repeat(np.nan,
                                                           self.array.shape[3])
        self.weight = np.array(weight).astype(float) \
            if weight is not None else np.repeat(np.nan, self.array.shape[3])
        self.injected_dose = np.array(injected_dose).astype(float) \
            if injected_dose is not None else np.repeat(np.nan,
                                                        self.array.shape[3])
        self.decay_factor = np.array(decay_factor).astype(float) \
            if decay_factor is not None else np.repeat(np.nan,
                                                       self.array.shape[3])

        self.min_x_value = min_x_value if (min_x_value is not None) else \
            DEFAULT_MIN_VALUES[1]
        self.min_y_value = min_y_value if (min_y_value is not None) else \
            DEFAULT_MIN_VALUES[0]
        self.min_z_value = min_z_value if (min_z_value is not None) else \
            DEFAULT_MIN_VALUES[2]

        # Set a default value for the affine matrix, pointing at the
        # canonical span vectors and being positioned around the min values.
        if affine is None:
            self.affine = np.zeros((4, 4))
            # Row direction.
            self.affine[0, 1] = 1
            # Column direction.
            self.affine[1, 0] = 1
            # Slice direction.
            self.affine[2, 2] = 1
            # Intercept.
            self.affine[:, 3] = [self.min_x_value,
                                 self.min_y_value,
                                 self.min_z_value,
                                 1]
            self.affine = handler_nifti.update_affine_voxel_size(
                affine=self.affine,
                voxel_size=self.voxel_size,
                encoded=False)
        else:
            # TODO: Fix the issue of a parameter affine value being ignored
            #  in case the loaded volume already specifies an affine.
            self.affine = affine
            # In case an affine was provided, the affine takes precedence
            # over the voxel_size parameter, even if it was specified!
            self.voxel_size = handler_nifti.get_voxel_size(affine=self.affine,
                                                           decode=False)

        # Call the helper function to load the volume info file, if any was
        # provided and no previous normalization data was loaded form the
        # input file.
        if volume_info_file is not None:
            weight, injected_dose, decay_factor, acquisition_time, \
                pet_channels = to_voxel_array_converter.extract_pet_info(
                info_file=volume_info_file,
                scan_name=self.name)
            # In case the column "pet_channels" did not exist, just assume
            # every channel is a PET channel.
            if isinstance(pet_channels, float):
                pet_channels = np.arange(len(self.channels))
            if not np.all(np.isnan(pet_channels)):
                # Only continue if any info about the pet channels was found.
                if np.max(pet_channels) == len(self.channels):
                    pet_channels = np.arange(len(pet_channels))
                if np.max(pet_channels) < len(self.channels):
                    self.weight = np.repeat(np.nan, len(self.channels)).astype(
                        float)
                    self.injected_dose = np.repeat(np.nan,
                                                   len(self.channels)).astype(
                        float)
                    self.decay_factor = np.repeat(np.nan,
                                                  len(self.channels)).astype(
                        float)
                    self.acquisition_time = np.repeat(0.0,
                                                      len(self.channels)).astype(
                        float)
                    self.weight[pet_channels] = weight
                    self.injected_dose[pet_channels] = injected_dose
                    self.decay_factor[pet_channels] = decay_factor
                    self.acquisition_time[pet_channels] = acquisition_time

    def get_array(self,
                  crop: bool = False,
                  replace_nan: bool = False,
                  norm=None,
                  voxel_size=None,
                  method="linear") -> Tuple[
        Union[ndarray, Any], Union[ndarray, Any]]:
        """
        Returns the actual array of the voxel array, with or without padding.
        :return: The array attribute of the voxel array object, with or
        without NaN padding.
        """
        array = self.array
        affine = self.affine
        if crop:
            array = to_voxel_array_converter.crop_volume(volume=array,
                                                         inplace=False)
        if replace_nan:
            # Only create a new copy of the array if the cropping didn't
            # happen, as the cropped array is a copy.
            array = to_voxel_array_converter.replace_values(volume=array,
                                                            old_val=np.nan,
                                                            new_val=0.0,
                                                            inplace=crop)
        elif not crop:
            # Make sure to always create a copy.
            array = np.copy(array)

        if norm is None:
            pass
        elif norm.lower() == "idcc":
            idcc_idxs = self.get_idcc_channels()
            array[..., idcc_idxs] = array[..., idcc_idxs] \
                                    / (self.decay_factor[idcc_idxs] *
                                       self.injected_dose[idcc_idxs]) * 100
        elif norm.lower() == "suv":
            suv_idxs = self.get_suv_channels()
            array[..., suv_idxs] = array[..., suv_idxs] * self.weight[suv_idxs] \
                                   / (self.decay_factor[suv_idxs] *
                                      self.injected_dose[suv_idxs]) * 1000
        else:
            raise ValueError(
                "For parameter norm only values idcc and suv are supported.")

        if voxel_size is not None:
            array, affine = handler_nifti.scale_volume(arr=array,
                                                       old_affine=self.affine,
                                                       new_voxel_size=voxel_size,
                                                       method=method,
                                                       fill_value=0,
                                                       decode=False)

        return array, affine

    def get_name(self) -> str:
        return self.name

    def get_coordinate_columns(self) -> List[str]:
        return self.coordinate_columns

    def get_channels(self) -> List[str]:
        return self.channels

    def get_voxel_size(self) -> np.ndarray:
        return np.array(self.voxel_size, dtype=float)

    def get_acquisition_time(self) -> np.ndarray:
        return np.array(self.acquisition_time, dtype=float)

    def get_weight(self) -> np.ndarray:
        return np.array(self.weight, dtype=float)

    def get_injected_dose(self) -> np.ndarray:
        return np.array(self.injected_dose, dtype=float)

    def get_decay_factor(self) -> np.ndarray:
        return np.array(self.decay_factor, dtype=float)

    def get_metadata(self) -> list:
        return self.metadata

    def get_affine(self,
                   encode: bool = False) -> np.ndarray:
        if encode:
            return handler_nifti.encode_affine(affine=self.affine,
                                               decoded_array_shape=self.array.shape)
        else:
            return self.affine

    def get_idcc_channels(self):
        valid_idcc_channels = (~np.isnan(self.injected_dose)) \
                              & (~np.isnan(self.decay_factor))
        valid_idcc_channels = np.where(valid_idcc_channels)[0].astype(int)
        return valid_idcc_channels

    def get_suv_channels(self):
        valid_suv_channels = (~np.isnan(self.weight)) \
                             & (~np.isnan(self.injected_dose)) \
                             & (~np.isnan(self.decay_factor))
        valid_suv_channels = np.where(valid_suv_channels)[0].astype(int)
        return valid_suv_channels

    def get_n_voxels(self) -> int:
        """
        Get the number of entries of the array that contain at least one
        non-Nan entry, i.e. they represent a voxel.
        :return:
        """
        return int(np.sum(np.any(~np.isnan(self.array), axis=3)))

    def pop(self, channel_idxs: list):
        """
        Remove channels of the specified indices and their corresponding
        attributes from the voxel array and return the popped array,
        much like in the list pop function.
        """
        # Make sure that the channel indices are a list, in case just a
        # singular channel was provided, either as a string or an integer.
        if (not isinstance(channel_idxs, Iterable)) | (isinstance(channel_idxs,
                                                                  str)):
            channel_idxs = [channel_idxs]
        # In case no channels are to be deleted, return an empty array.
        if len(channel_idxs) == 0:
            return self.array[..., []]
        if isinstance(channel_idxs[0], str):
            channel_idxs_int = []
            for channel in channel_idxs:
                try:
                    channel_idxs_int += [self.channels.index(channel)]
                except ValueError:
                    raise KeyError(F"No channel with name {channel} in voxel "
                                   F"array.")
            channel_idxs = channel_idxs_int

        keep_idxs = [idx for idx in range(len(self.channels)) if
                     idx not in list(channel_idxs)]
        popped_arr = self.array[..., channel_idxs]

        self.array = self.array[..., keep_idxs]
        self.channels = np.array([var for idx, var in enumerate(self.channels) \
                                  if idx not
                                  in channel_idxs])
        self.metadata = np.array(
            [var for idx, var in enumerate(self.metadata) if idx not
             in channel_idxs])
        self.acquisition_time = np.array([var for idx, var in
                                          enumerate(self.acquisition_time) if
                                          idx
                                          not in channel_idxs])
        self.weight = np.array(
            [var for idx, var in enumerate(self.weight) if idx
             not in channel_idxs])
        self.injected_dose = np.array(
            [var for idx, var in enumerate(self.injected_dose)
             if idx not in channel_idxs])
        self.decay_factor = np.array(
            [var for idx, var in enumerate(self.decay_factor)
             if idx not in channel_idxs])

        return popped_arr

    def __getitem__(self, channel_idxs: F):
        """
        Simulates dataframe column indexing with the voxel arrays.
        """
        # Make sure that the channel indices are a list, in case just a
        # singular channel was provided, either as a string or an integer.
        if (not isinstance(channel_idxs, Iterable)) \
                | (isinstance(channel_idxs, str)):
            channel_idxs = [channel_idxs]
        # In case no channels are to be deleted, return an empty array.
        if len(channel_idxs) == 0:
            return self.array[..., []]
        if isinstance(channel_idxs[0], str):
            channel_idxs_int = []
            for channel_idxs in channel_idxs:
                try:
                    channel_idxs_int += [self.channels.index(channel_idxs)]
                except ValueError:
                    raise KeyError(
                        F"No channel with name {channel_idxs} in voxel "
                        F"array.")
            channel_idxs = channel_idxs_int

        return self.array[..., channel_idxs]

    def get_min_values(self) -> Tuple[int, int, int]:
        return self.min_x_value, self.min_y_value, self.min_z_value

    def to_pandas(self, dtype=None) -> pd.DataFrame:
        return to_voxel_array_converter.convert_array_to_xyz_dataframe(
            input_arr=self.array,
            column_names=self.coordinate_columns + self.channels,
            min_values=(self.min_x_value, self.min_y_value, self.min_z_value),
            dtype=dtype)

    def to_csv(self, output_path: str, dtype=None) -> None:
        self.to_pandas(dtype=dtype).to_csv(output_path, index=False)

    def to_hdf(self, output_path: str) -> None:
        to_voxel_array_converter.convert_input_to_hdf(source=self,
                                                      output_name=output_path)

    def to_nifti(self,
                 output_path: str,
                 enforce_3D: bool = False,
                 channel_indexer: str = "channel",
                 mode: str = "index",
                 start_idx: int = 0,
                 n_digits: int = 2) -> None:
        # Make sure the output directory and its subdirectories exist!
        try:
            os.makedirs(os.path.dirname(output_path))
        except FileExistsError:
            pass

        array = handler_nifti.encode_array(arr=self.array)
        affine = handler_nifti.encode_affine(affine=self.affine,
                                             decoded_array_shape=self.array.shape)
        # Enable the option to split the volume in to 3D NIFTI files.
        if enforce_3D:
            # This is necessary to enable compatibility with nnUNet!
            if channel_indexer is None:
                channel_indexer = ""
                eq = "="
            else:
                eq = ""

            for channel_idx in range(array.shape[3]):
                channel_arr = array[..., channel_idx]

                dir, file = os.path.split(output_path)
                name = file.split(".")[0]
                suffix = ".".join(file.split(".")[1:])

                if mode.lower() == "index":
                    # Only add the channel index if more than 1 channel exist.
                    if array.shape[-1] > 1:
                        output_idx = str(start_idx + channel_idx)
                        # Make sure the channel indices always have length n_digits!
                        output_idx = "0" * max(n_digits - len(output_idx),
                                               0) + output_idx
                        channel_path = os.path.join(dir,
                                                    name + "_" \
                                                    + channel_indexer
                                                    + eq \
                                                    + output_idx \
                                                    + "." + suffix)
                    else:
                        channel_path = output_path
                elif mode.lower() == "channel":
                    channel_path = os.path.join(dir, name + "_" + self.channels[
                        channel_idx] + "." + suffix)
                else:
                    raise ValueError("Only modes index and channel are "
                                     "supported.")

                nifti_img = nib.Nifti1Image(dataobj=channel_arr, affine=affine)
                nib.save(img=nifti_img, filename=channel_path)

                # Free up the space occupied by the NIFTI image.
                nifti_img.uncache()
                del nifti_img
        else:
            nifti_img = nib.Nifti1Image(dataobj=array, affine=affine)
            nib.save(img=nifti_img, filename=output_path)

            # Free up the space occupied by the NIFTI image.
            nifti_img.uncache()
            del nifti_img

    def to_npy(self, output_path: str) -> None:
        with open(output_path, 'wb') as file:
            np.save(file, self.array)


class VoxelArrayProcessor:
    """
    Class that returns several voxel arrays which were co-registered to each
    other.
    """

    def __init__(self,
                 volume_files: List[str],
                 ROI_file: Optional[str] = None,
                 volume_info_file=None,
                 fill_val_interpolation=None):

        # Make sure that the paths to the volume files are always contained
        # within a list, even if just one file was specified.
        if isinstance(volume_files, str):
            volume_files = [volume_files]

        voxel_array_collection = []
        n_channels_per_volume_dicom_nifti = []
        n_channels_per_volume_non_dicom = []
        ROI_idxs = None

        # The standard settings for the DICOM interpolation.
        method = "linear"

        self.n_channels_per_volume = []
        self.ROI_array = None
        self.metadata_ROI = None
        self.channels_ROI = None
        self.ROI_names_per_segmentation = None

        # Is only really needed if a Siemens .csv file is loaded as a ROI
        # volume.
        roi_names_per_modality = []

        # Separate the volume files into DICOM- and non-DICOM-files.
        dicom_nifti_paths = []
        non_dicom_paths = []

        for volume_idx, volume_path in enumerate(volume_files):
            if handler_dicom.is_DICOM_directory(path=volume_path) \
                    | handler_nifti.is_nifti(path=volume_path):
                dicom_nifti_paths.append(volume_path)
            # Make sure not to add empty or incompatible directories to the
            # list of paths to non-DICOM objects.
            elif os.path.isfile(volume_path):
                non_dicom_paths.append(volume_path)
            else:
                raise ValueError(F"The provided directory at {volume_path} "
                                 F"does not contain any valid DICOM files.")

        if ROI_file is not None:
            if handler_dicom.is_DICOM_directory(path=ROI_file) \
                    | handler_nifti.is_nifti(path=ROI_file) \
                    | is_siemens(path=ROI_file):
                # Update the interpolation details if coordinate ROIs are
                # identified, to properly display said ROIs.
                method = ["linear"] * len(dicom_nifti_paths) + ["nearest"]
                # BE CAUTIOUS: THE EXTRAPOLATION HAS BEEN ACTIVATED FOR ROIS,
                # BE WARY OF UNWANTED SIDE-EFFECTS OF THIS.
                fill_val_interpolation = [fill_val_interpolation] * len(
                    dicom_nifti_paths) + [None]

                dicom_nifti_paths.append(ROI_file)
            else:
                non_dicom_paths.append(ROI_file)

        # Merge all DICOM directories to the first DICOM directory specified.
        if len(dicom_nifti_paths) > 0:

            array, name, channels, voxel_size, acquisition_time, weight, \
                injected_dose, decay_factor, metadata, \
                n_channels_per_volume_dicom_nifti, affine, roi_names_per_modality = \
                handler_dicom.fuse_and_convert_DICOM_NIFTI(
                    volume_paths=dicom_nifti_paths,
                    output_path=None,
                    voxel_size=None,
                    method=method,
                    fill_value=fill_val_interpolation)

            # Make sure that the filler value that was inserted to avoid
            # excess extrapolation during the fusion is replaced with the
            # final filler value NaN. This temporary filler value will always
            # be -1, because ROI indices always start at 0 and get incremented.
            if ROI_file in dicom_nifti_paths:
                first_ROI_idx = int(
                    np.sum(n_channels_per_volume_dicom_nifti[:-1]))
                ROI_arr = array[..., first_ROI_idx:]
                ROI_arr[ROI_arr == -1] = np.nan

            voxel_arr = VoxelArray(source_voxel=array,
                                   metadata=metadata,
                                   name=name,
                                   channels=channels,
                                   voxel_size=voxel_size,
                                   acquisition_time=acquisition_time,
                                   weight=weight,
                                   injected_dose=injected_dose,
                                   decay_factor=decay_factor,
                                   affine=affine,
                                   volume_info_file=volume_info_file)
            voxel_array_collection.append(voxel_arr)

        for non_dicom_path in non_dicom_paths:
            voxel_arr = VoxelArray(
                source_voxel=non_dicom_path,
                volume_info_file=volume_info_file)
            voxel_array_collection.append(voxel_arr)
            n_channels_per_volume_non_dicom += [len(voxel_arr.get_channels())]

        # Identify the channels describing the ROIs.
        if ROI_file in dicom_nifti_paths:
            # This works because we know that the last volume will be the
            # ROI volume, as we added it to the end of the file list.
            # BUT DON'T FORGET TO ADD ALL OTHER NON-DICOM VOLUMES FIRST,
            # AS WELL.
            first_ROI_idx = int(np.sum(n_channels_per_volume_non_dicom)) + int(
                np.sum(n_channels_per_volume_dicom_nifti[:-1]))
            last_ROI_idx = first_ROI_idx + n_channels_per_volume_dicom_nifti[-1]
            ROI_idxs = np.arange(first_ROI_idx, last_ROI_idx)
            # Make sure to remove the ROI channels from the list of channel
            # counts for each volume.
            n_channels_per_volume_dicom_nifti = n_channels_per_volume_dicom_nifti[
                                                :-1]
        elif ROI_file in non_dicom_paths:
            # DON'T FORGET TO ADD ALL DICOM-VOLUMES FIRST.
            first_ROI_idx = int(
                np.sum(n_channels_per_volume_dicom_nifti)) + int(
                np.sum(n_channels_per_volume_non_dicom[:-1]))
            last_ROI_idx = first_ROI_idx + n_channels_per_volume_non_dicom[-1]
            ROI_idxs = np.arange(first_ROI_idx, last_ROI_idx)
            n_channels_per_volume_non_dicom = n_channels_per_volume_non_dicom[
                                              :-1]

        # Add up the counts for both the DICOM and non-DICOM volumes of the
        # fused voxel array.
        self.n_channels_per_volume = np.array(n_channels_per_volume_dicom_nifti
                                              + n_channels_per_volume_non_dicom)

        # Make sure that all voxel arrays contain the same coordinate
        # information.
        self.voxel_array = to_voxel_array_converter.pad_and_fuse_voxel_arrays(
            voxel_array_collection=voxel_array_collection)
        # Remove the channels belonging to the ROI volume from the voxel
        # array and store them as an extra array.
        if ROI_idxs is not None:
            # Keep the channels and metadata for the ROIs, in order to display
            # them separately later.
            self.channels_ROI = [channel for idx, channel in enumerate(
                self.voxel_array.get_channels()) if idx in ROI_idxs]
            self.ROI_names_per_segmentation = [ROI_names for idx, ROI_names in
                                               enumerate(
                                                   roi_names_per_modality) if
                                               idx in ROI_idxs]
            self.metadata_ROI = [metadata for idx, metadata in enumerate(
                self.voxel_array.get_metadata()) if idx in ROI_idxs]
            self.ROI_array = self.voxel_array.pop(ROI_idxs)

    def get_voxel_array(self):
        return self.voxel_array

    def get_ROI_array(self):
        return self.ROI_array

    def get_segmentation_bbox(self,
                              segmentation_idx: int,
                              ROI_id: int = None,
                              include_voxels_outside_mask: bool = True):
        """
        Function to get the contents of a ROI for a selected segmentation,
        but ONLY within a bounding box.
        """
        if ROI_id is not None:
            mask = np.zeros(self.ROI_array.shape[:3])
            mask[:] = np.nan
            mask[self.ROI_array[..., segmentation_idx] == ROI_id] = 1
        else:
            mask = self.ROI_array[..., segmentation_idx]

        ROI_contents,_ = self.voxel_array.get_array()
        ROI_contents = to_voxel_array_converter.get_mask_contents(
            volume=ROI_contents,
            mask=mask,
            include_voxels_outside_mask=include_voxels_outside_mask)

        return ROI_contents

    def get_channels_ROI(self):
        return self.channels_ROI

    def get_metadata_ROI(self):
        return self.metadata_ROI

    def get_ROI_names_per_segmentation(self):
        return self.ROI_names_per_segmentation

    def get_n_channels_per_volume(self):
        return self.n_channels_per_volume


# Just to trick the import system into working.
import to_voxel_array_converter
from handler_siemens import convert_siemens_to_volume, \
    is_siemens
from mat_handler import convert_mat_to_voxel_array
import handler_dicom
