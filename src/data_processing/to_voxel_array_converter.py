import os
from typing import List, Tuple, TypeVar, Optional

import h5py

import numpy as np
import pandas as pd
from PIL import Image
from pandas.errors import ParserError
import re
import ast
from functools import reduce

import voxelizer

################################################################################

default_voxel_size: float = 0.22
default_coordinate_columns: List[str] = ['x', 'y', 'z']

S1 = TypeVar('S1', str, pd.DataFrame)
S2 = TypeVar('S2', str, pd.DataFrame, voxelizer.VoxelArray)


################################################################################


def get_dataframe_generic(source: S1) -> pd.DataFrame:
    """
    Takes either a string or a dataframe as input and returns a dataframe.
    """
    if isinstance(source, str):
        try:
            result_df: pd.DataFrame = pd.read_csv(source)
        except ParserError:
            raise ValueError(F"Could not read file {source} as a dataframe.")
    elif isinstance(source, pd.DataFrame):
        result_df: pd.DataFrame = source.copy()
    else:
        raise ValueError("Dataframe source must either be a path string or a "
                         "dataframe itself.")
    return result_df


def convert_dataframe_into_4D_array(df: pd.DataFrame,
                                    coordinate_columns: list,
                                    pad_to_same_length: bool = False,
                                    reduce_to_minimal_size: bool = True):
    """
    Takes the entirety of data for a tumor, either in terms of input data or
    clustering results, and transforms it into a 4D-array with correct
    spatial representation.
    :param df: Input dataframe.
    :param coordinate_columns: The columns to be treated as the coordinates.
    :param pad_to_same_length: Specify whether the dimensions of the output
    array should all have the same length.
    :param reduce_to_minimal_size: Decide whether or not to return the
    smallest array containg all non-NaN values.
    :return: A 4D-array, where each 3D-entry represents all of the data for
    one voxel.
    """
    if (len(coordinate_columns) != 3) | (not set(coordinate_columns).issubset(
            set(df.columns))):
        raise ValueError("Badly shaped input for the conversion of a "
                         "dataframe into an array.")

    min_x: int = min(df[coordinate_columns[0]])
    max_x: int = max(df[coordinate_columns[0]])
    min_y: int = min(df[coordinate_columns[1]])
    max_y: int = max(df[coordinate_columns[1]])
    min_z: int = min(df[coordinate_columns[2]])
    max_z: int = max(df[coordinate_columns[2]])

    # In case the array is not supposed to be minimal, assume 0 as the
    # minimal coordinate, in case no negative coordinates are provided.
    if not reduce_to_minimal_size:
        min_x = min(min_x, 0)
        min_y = min(min_y, 0)
        min_z = min(min_z, 0)

    len_x_dim: int = int(max_x - min_x)
    len_y_dim: int = int(max_y - min_y)
    len_z_dim: int = int(max_z - min_z)

    if pad_to_same_length:

        max_len_dim: int = max(len_x_dim, len_y_dim, len_z_dim)

        pad_constant_x: int = int(
            + (max_len_dim - len_x_dim) / 2 + (
                    (max_len_dim - len_x_dim) % 2)) - min_x
        pad_constant_y: int = int(
            + (max_len_dim - len_y_dim) / 2 + (
                    (max_len_dim - len_y_dim) % 2)) - min_y
        pad_constant_z: int = int(
            + (max_len_dim - len_z_dim) / 2 + (
                    (max_len_dim - len_z_dim) % 2)) - min_z

        x_indices: np.ndarray = (df[coordinate_columns[
            0]].to_numpy() + pad_constant_x).astype(int)
        y_indices: np.ndarray = (df[coordinate_columns[
            1]].to_numpy() + pad_constant_y).astype(int)
        z_indices: np.ndarray = (df[coordinate_columns[
            2]].to_numpy() + pad_constant_z).astype(int)

        output_arr: np.ndarray = np.zeros((max_len_dim + 1, max_len_dim + 1,
                                           max_len_dim + 1,
                                           len(df.columns) - 3))

    else:
        x_indices: np.ndarray = (df[coordinate_columns[
            0]].to_numpy() - min_x).astype(int)
        y_indices: np.ndarray = (df[coordinate_columns[
            1]].to_numpy() - min_y).astype(int)
        z_indices: np.ndarray = (df[coordinate_columns[
            2]].to_numpy() - min_z).astype(int)

        output_arr: np.ndarray = np.zeros((len_z_dim + 1,
                                           len_y_dim + 1,
                                           len_x_dim + 1,
                                           len(df.columns) - 3))

    output_arr[:] = np.nan

    output_arr[(z_indices, y_indices, x_indices)] = df.drop(
        columns=coordinate_columns)

    return min_x, min_y, min_z, output_arr


def pad_array_to_cube_shape(input_arr: np.ndarray) -> np.ndarray:
    """
    Pad any array so that its first three dimensions have
    the same size (i.e. the array is cubic).
    :param input_arr: Input array, can have any number of dimensions
    :return: The padded input array, where the first three dimensions have
    the same size, padded with NaN.
    """
    # In case only a an array with less than THREE dimensions is provided.
    if input_arr.ndim < 3:
        worker_arr: np.ndarray = np.reshape(input_arr, (
                (1,) * (3 - input_arr.ndim)) + input_arr.shape)
    else:
        worker_arr: np.ndarray = np.copy(input_arr)

    max_dim_len: int = np.max(worker_arr.shape[:3])
    z_padding: int = int((max_dim_len - worker_arr.shape[0]) / 2) + (
            max_dim_len - worker_arr.shape[0]) % 2
    y_padding: int = int((max_dim_len - worker_arr.shape[1]) / 2) + (
            max_dim_len - worker_arr.shape[1]) % 2
    x_padding: int = int((max_dim_len - worker_arr.shape[2]) / 2) + (
            max_dim_len - worker_arr.shape[2]) % 2
    output_arr: np.ndarray = np.zeros(
        (max_dim_len,) * 3 + worker_arr.shape[3:])

    output_arr[:] = np.nan
    output_arr[z_padding:z_padding + worker_arr.shape[0],
    y_padding:y_padding + worker_arr.shape[1],
    x_padding:x_padding + worker_arr.shape[2]] = worker_arr

    return output_arr


def get_metadata_for_tumor(channel_columns: List[str],
                           source_metadata: S1,
                           run_regex: str = "Run_[0-9]+(\([0-9]+\))*") -> \
        List[str]:
    """
    Create an array containing a formatted string for each clustering run that
    was done on the given tumor.
    :param source_metadata: Dictionary containing the information in
    which file, given the clustering run results for a tumor, the metadata
    for said runs can be found.
    :param channel_columns: Names of the columns of the .csv file containing
    the results of the clustering runs for the given tumor.
    :param no_metadata_column_names: The subset of column names for which
    there is no metadata available.
    :return: An array containing the metadata per run as strings, ordered in
    the way the run column names were ordered.
    """
    # NOTE: THIS FUNCTION IS HIGHLY DEPENDENT ON THE CORRECT SHAPE OF THE
    # .CSV FILE CONTAINING THE METADATA.

    # In case no metadata dataframe is provided, simply return the column
    # names as metadata.
    if not source_metadata:
        return channel_columns

    # In case the source for the metadata is already a list, return the list,
    # fit to be the same length as the number of channel columns.
    if isinstance(source_metadata, list):
        return (source_metadata + [""] * (max(0, len(channel_columns) - len(
            source_metadata))))[:len(channel_columns)]

    metadata_list: list = []

    df_metadata: pd.DataFrame = get_dataframe_generic(source_metadata)

    # This dance is needed in order to find the metadata run column name
    # most matching with the run id selected by the user.
    run_ids: pd.DataFrame = df_metadata['Run ID'].astype(str)

    for channel in channel_columns:
        # In case the column in question doesn't fit the mold for a cluster
        # run ID, simply add the name of the column as metadata.
        if not re.search(run_regex, channel):
            metadata_list.append(channel)
        else:
            match_func: callable = lambda run_id: 0 if not re.search(channel,
                                                                     run_id) else len(
                re.search(channel, run_id).group(0))

            match_scores: pd.DataFrame = run_ids.apply(match_func)

            row_metadata_run = df_metadata.iloc[match_scores.argmax()]
            id_descriptor: str = "Run ID: "
            id_str: str = id_descriptor + str(row_metadata_run['Run ID'])

            tumor_name_descriptor: str = "Tumor Names: "
            tumor_str: str = tumor_name_descriptor + ",\n".join([
                ' ' * len(tumor_name_descriptor) + tumor_name for tumor_name
                in ast.literal_eval(row_metadata_run['Tumor Names'])])[len(
                tumor_name_descriptor):]

            ing_style_descriptor: str = "Ingestion Style: "
            ing_style_str: str = ing_style_descriptor + str(
                row_metadata_run['Ingestion Style'])

            passed_coords_descriptor: str = "Passed Coordinates: "
            # For backwards compatibility's sake.
            try:
                passed_coords_str: str = passed_coords_descriptor + str(
                    row_metadata_run['Passed Coordinates'])
            except KeyError:
                passed_coords_str: str = passed_coords_descriptor

            channel_name_descriptor: str = "Feature Names: "
            channel_str: str = channel_name_descriptor + ",\n".join([
                ' ' * len(channel_name_descriptor) + channel_name for
                channel_name
                in ast.literal_eval(
                    row_metadata_run['Feature Names'])])[len(
                channel_name_descriptor):]

            scaler_name_descriptor: str = "Scalers Used: "
            scaler_list: list = ast.literal_eval(row_metadata_run['Scalers '
                                                                  'Used'])
            if scaler_list is None:
                scalers_str: str = scaler_name_descriptor + " None"
            else:
                scalers_str: str = scaler_name_descriptor + ",\n".join([
                    ' ' * len(scaler_name_descriptor) + scaler_name for
                    scaler_name
                    in scaler_list])[len(scaler_name_descriptor):]

            algorithm_descriptor: str = "Algorithm Used: "
            algorithm_str: str = algorithm_descriptor \
                                 + format_algorithm_info_string_for_text_box(
                row_metadata_run['Algorithm Used'],
                len(algorithm_descriptor))

            metadata_for_column: str = id_str \
                                       + "\n\n" + tumor_str \
                                       + "\n\n" + ing_style_str \
                                       + "\n\n" + passed_coords_str \
                                       + "\n\n" + channel_str \
                                       + "\n\n" + scalers_str \
                                       + "\n\n" + algorithm_str
            metadata_list.append(metadata_for_column)

    return metadata_list


def format_algorithm_info_string_for_text_box(alg_str: str, offset: int):
    """
    Helper function outputting a formatted version of a string representation
    of a clustering algorithm instance.
    :param alg_str: The unformatted output of the __str__ method.
    :param offset: The number of white spaces to be inserted at the beginning
    of every line but the first.
    :return: The  formatted version of the string representation of the given
    algorithm instance.
    """
    open_bracket_count: int = 0
    close_bracket_count: int = 0

    alg_name: str = ""
    parameter_name_beginning_index: int = -1
    parameter_content_beginning_index: int = -1
    alg_parameter_name_beginning_index: int = -1
    alg_parameter_content_beginning_index: int = -1

    nested_alg_limits: list = []

    for i, char in enumerate(alg_str):
        if (char == '(') & (open_bracket_count == 0):
            # The very first parameter is encountered.
            alg_name: str = alg_str[:i]
            open_bracket_count += 1
            parameter_name_beginning_index = i + 1
        elif (char == '(') & (open_bracket_count > 0) & (
                open_bracket_count - 1 == close_bracket_count):
            open_bracket_count += 1
            alg_parameter_name_beginning_index = parameter_name_beginning_index
            alg_parameter_content_beginning_index = parameter_content_beginning_index
        elif (char == ')') & (close_bracket_count == open_bracket_count -
                              2) & (i != len(alg_str) - 1):
            close_bracket_count += 1
            nested_alg_limits.append((alg_parameter_name_beginning_index,
                                      alg_parameter_content_beginning_index,
                                      i + 1))
        elif (char == '('):
            open_bracket_count += 1
        elif (char == ')'):
            close_bracket_count += 1
        elif (char == ','):
            parameter_name_beginning_index = i + 1
        elif (char == '='):
            parameter_content_beginning_index = i + 1

    beginning_of_flat_section_index: int = len(alg_name) + 1
    formated_output_str: str = "Name=" + alg_name + ",\n"
    for i in range(len(nested_alg_limits)):
        info_triplet: tuple = nested_alg_limits[i]
        if i == 0:
            parameter_section: str = alg_str[
                                     beginning_of_flat_section_index:
                                     info_triplet[
                                         1]]
        else:
            parameter_section: str = alg_str[nested_alg_limits[i - 1][
                                                 2] + 1:info_triplet[1]]
        formated_output_str += (",\n" if i != 0 else "") + ",\n".join(
            [" " * offset + param_details for param_details in
             parameter_section.split(",")])
        formated_output_str += format_algorithm_info_string_for_text_box(
            alg_str[info_triplet[1]:
                    info_triplet[2]],
            offset + (info_triplet[
                          1] -
                      info_triplet[
                          0]))
    else:
        if nested_alg_limits:
            formated_output_str += ",\n" + ",\n".join(
                [" " * offset + param_details for param_details in
                 alg_str[nested_alg_limits[-1][2]:].split(")")[0].split(",")[
                 1:]])
        else:
            formated_output_str += ",\n".join(
                [" " * offset + param_details for param_details in
                 alg_str[beginning_of_flat_section_index:].split(")")[0].split(
                     ",")])

    return condense_square_brackets(formated_output_str)


def condense_square_brackets(input_str: str):
    """
    Post-process the formatted metadata-string so that the contents of any
    pair of square brackets is displayed in one line.
    :param input_str: The metadata-string, completely processed with the
    exception of the square brackets.
    :return: The metadata-string, adjusted for the present square brackets.
    """
    open_bracket_locs: list = [i for i, ltr in enumerate(input_str) if
                               ltr == '[']
    close_bracket_locs: list = [i for i, ltr in enumerate(input_str) if
                                ltr == ']']
    bracket_pairs: list = list(zip(open_bracket_locs, close_bracket_locs))
    result_str: str = ""
    current_bracket_index: int = 0
    for bracket_pair in bracket_pairs:
        result_str += input_str[current_bracket_index:bracket_pair[0]]
        result_str += "".join(input_str[bracket_pair[0]:bracket_pair[1] +
                                                        1].split())
        current_bracket_index = bracket_pair[1] + 1
    else:
        result_str += input_str[current_bracket_index:]

    return result_str


def extract_channel_columns(df: pd.DataFrame, coordinate_columns: List[
    str]) -> List[str]:
    """
    Removes the coordinate (xyz) columns from the list of columns of a
    dataframe.
    :param df: The input dataframe.
    :param coordinate_columns: The name of the dataframe columns describing
    the coordinates.
    :return: A list of all columns of the dataframe that do not describe the
    coordinates.
    """
    result_list: List[str] = list(df.columns)
    for coordinate in coordinate_columns:
        result_list.remove(coordinate)
    return result_list


def convert_input_to_hdf(source: S2,
                         output_name: str,
                         coordinate_columns: List[str] = None,
                         voxel_size: Optional[float] = None,
                         content_times: Optional[List[float]] = None,
                         name: Optional[str] = None) -> None:
    df: Optional[pd.DataFrame] = None
    if isinstance(source, str):
        try:
            df = pd.read_csv(source)
            tumor_name: str = source.split("\\")[-1].split("/")[-1].split(".")[
                0]
        except ParserError:
            raise ValueError(F"The source {source} cannot be read as a "
                             F"dataframe.")
    elif isinstance(source, pd.DataFrame):
        if name is None:
            raise ValueError("If a dataframe is provided as source for the "
                             ".hdf conversion, the name "
                             "parameter cannot be None.")
        df = source
        tumor_name: str = name

    elif isinstance(source, voxelizer.VoxelArray):
        tumor_array: np.ndarray = source.array
        tumor_name: str = source.get_name()
        tumor_channels: List[str] = source.get_channels()
        tumor_voxel_size: List[float] = source.get_voxel_size()
        tumor_content_times: List[float] = content_times
        tumor_metadata: list = source.get_metadata()

        min_x_value, min_y_value, min_z_value = source.get_min_values()
    else:
        raise ValueError("The source for the .hdf conversion must either "
                         "be a string path to a .csv file or a dataframe "
                         "or a VoxelArray object.")

    if df is not None:
        if coordinate_columns is None:
            raise ValueError("Please specify coordinate columns when "
                             "converting from a .csv file or dataframe into a "
                             ".hdf file.")
        tumor_array: np.ndarray = convert_dataframe_into_4D_array(df,
                                                                  coordinate_columns)
        tumor_channels: List[str] = extract_channel_columns(df,
                                                            coordinate_columns)
        tumor_voxel_size: float = voxel_size

        min_x_value = min(df[coordinate_columns[0]])
        min_y_value = min(df[coordinate_columns[1]])
        min_z_value = min(df[coordinate_columns[2]])

        tumor_content_times = content_times if content_times is not None \
            else [0.0] * len(tumor_channels)

    with h5py.File(output_name, 'w') as out_file:
        dataset_voxel: h5py.Dataset = out_file.create_dataset('voxel_array',
                                                              data=tumor_array)
        dataset_voxel.attrs['name'] = tumor_name
        dataset_voxel.attrs['channels'] = tumor_channels
        dataset_voxel.attrs['voxel_size'] = tumor_voxel_size
        dataset_voxel.attrs['content_times'] = tumor_content_times
        dataset_voxel.attrs['metadata'] = tumor_metadata

        dataset_voxel.attrs['min_x_value'] = min_x_value
        dataset_voxel.attrs['min_y_value'] = min_y_value
        dataset_voxel.attrs['min_z_value'] = min_z_value

    return


def convert_array_to_xyz_dataframe(input_arr: np.ndarray,
                                   column_names=None,
                                   min_values=(0, 0, 0),
                                   dtype=None) \
        -> pd.DataFrame:
    """
    Converts an array into a dataframe containing xyz information about the
    position of every single entry of the array.
    :param input_arr: The input array for the conversion.
    :param column_names: The names that are to be given to the output
    dataframe. If none are given or if their length doesn't match the number
    of columns of the ouput dataframe, the dataframe column names assume the
    standard values (0,1,2,...).
    :return: The ouput dataframe, containing all the information of the
    input array, with spatial information encoded as the first 3 columns
    of the output dataframe.
    """

    # Add dimensions to input array in case its dimensions aren't sufficient
    # for a conversion into an xyz dataframe.
    if input_arr.ndim < 4:
        worker_arr: np.ndarray = np.reshape(input_arr, (
                (1,) * (4 - input_arr.ndim)) + input_arr.shape)
    else:
        worker_arr: np.ndarray = np.copy(input_arr)

    data_locs: tuple = np.where(np.any(np.isnan(worker_arr) == False, axis=3))

    data_entries: np.ndarray = worker_arr[data_locs]
    locs_df: pd.DataFrame = pd.DataFrame(np.array(
        data_locs[::-1]).T + np.array(min_values)).astype(int)
    entries_df: pd.DataFrame = pd.DataFrame(data_entries).astype(dtype)

    result_df: pd.DataFrame = pd.concat([locs_df, entries_df], axis=1)

    # Apply the column names, in cas they were proviede.
    if column_names is not None:
        if len(result_df.columns) == len(column_names):
            result_df.columns = column_names

    result_df = result_df.sort_values(by=list(result_df.columns[:3]))
    result_df = result_df.reset_index(drop=True)

    return result_df


def pad_and_fuse_voxel_arrays(voxel_array_collection: list):
    """
    Create one fused voxel array out of several disparate ones, while adding
    up their additional information.
    """
    # Check for trivial voxel array collections.
    if len(voxel_array_collection) == 0:
        return None
    elif len(voxel_array_collection) == 1:
        return voxel_array_collection[0]

    min_values = np.array([np.infty, np.infty, np.infty])
    max_values = np.array([-np.infty, -np.infty, -np.infty])
    channels = []
    metadata = []
    weight = []
    injected_dose = []
    decay_factor = []
    acquisition_times = []
    # Choose the name of the target as the name of the fused voxel array.
    name = voxel_array_collection[0].get_name()
    # Just set the voxel size of the fused voxel array equal to
    # the first voxel size that is encountered.
    voxel_size = voxel_array_collection[0].get_voxel_size()
    coordinate_columns = voxel_array_collection[0].get_coordinate_columns()

    # Just take the affine of the first voxel array in the collection.
    affine = voxel_array_collection[0].get_affine()

    # First pass over the voxel arrays to determine the size of the fused
    # array, as well as the other class attributes.
    for voxel_arr in voxel_array_collection:
        va_min_values = np.array(voxel_arr.get_min_values()).astype(int)
        # Reverse the shape of the volume to adhere to the order of x-y-z.
        va_max_values = va_min_values + voxel_arr.array.shape[-2::-1]

        # Update the global min and max values in cases where the
        # loop voxel array surpasses the the current best values.
        min_values[va_min_values < min_values] = va_min_values[
            va_min_values < min_values]
        max_values[va_max_values > max_values] = va_max_values[
            va_max_values > max_values]

        # Make sure there are no duplicate channel names.
        for channel in voxel_arr.get_channels():
            if channel in channels:
                channels += [channel + ".1"]
            else:
                channels += [channel]
        # Convert to a list to avoid triggering numpy array element-wise
        # addition.
        metadata += list(voxel_arr.get_metadata())
        weight += list(voxel_arr.get_weight())
        injected_dose += list(voxel_arr.get_injected_dose())
        decay_factor += list(voxel_arr.get_decay_factor())
        acquisition_times += list(voxel_arr.get_acquisition_time())

    # Reverse the values again to now adhere to the order of z-y-x. Add one
    # to account for the fact that you always need an array whose length is
    # the maximum index plus 1.
    array_shape = tuple((max_values - min_values).astype(int))[::-1] + (
        len(channels),)
    array = np.zeros(array_shape)
    array[:] = np.nan
    channel_idx = 0
    # Second pass over the voxel array to save the volume information of all
    # voxel arrays in one united array.
    for voxel_arr in voxel_array_collection:
        va_min_values = np.array(voxel_arr.get_min_values())
        va_shape = voxel_arr.array.shape
        va_n_channels = len(voxel_arr.get_channels())
        x_offset, y_offset, z_offset = (va_min_values - min_values).astype(int)
        array[z_offset:z_offset + va_shape[0], y_offset:y_offset + va_shape[1],
        x_offset:x_offset + va_shape[2],
        channel_idx:channel_idx + va_n_channels] = voxel_arr.array

        channel_idx += va_n_channels

    # The fused voxel array, encompassing the volume information of all
    # passed voxel arrays.
    fused_voxel_arr = voxelizer.VoxelArray(source_voxel=array,
                                           metadata=metadata,
                                           name=name,
                                           channels=channels,
                                           voxel_size=voxel_size,
                                           coordinate_columns=coordinate_columns,
                                           min_values=list(min_values),
                                           affine=affine,
                                           weight=weight,
                                           injected_dose=injected_dose,
                                           decay_factor=decay_factor,
                                           acquisition_time=acquisition_times)

    return fused_voxel_arr


def pad_voxel_arrays_to_each_other(
        voxel_array_collection: List[voxelizer.VoxelArray]):
    """
    Ensures that all dataframes of voxel arrays which contribute to one
    voxel array collection have the same coordinate rows available,
    with the channel columns being padded with None. It then updates the
    affected parameters of each voxel array.
    """
    min_values_per_voxel_array: np.ndarray = np.array(
        [voxel_arr.get_min_values() for voxel_arr in
         voxel_array_collection])
    shape_per_voxel_array: np.ndarray = np.array(
        [voxel_arr.array.shape
         for voxel_arr in
         voxel_array_collection])
    # In case either the dimensions of the arrays or the minimum values per
    # direction are not identical for each element of the voxel array
    # collection, adjust them to one another. Else, return.
    if np.any(min_values_per_voxel_array != min_values_per_voxel_array[0],
              axis=(0, 1)) | \
            np.any(shape_per_voxel_array != shape_per_voxel_array[0],
                   axis=(0, 1)):
        df_list: list = [voxel_arr.to_pandas() for voxel_arr in
                         voxel_array_collection]
        reference_xyz_df: pd.DataFrame = pd.DataFrame(
            columns=default_coordinate_columns)
        for df in df_list:
            reference_xyz_df = \
                reference_xyz_df.merge(df, on=default_coordinate_columns,
                                       how="outer")[
                    default_coordinate_columns]

        for df, voxel_arr in zip(df_list, voxel_array_collection):
            df = df.merge(reference_xyz_df,
                          on=default_coordinate_columns,
                          how="outer")
            min_x_value, min_y_value, min_z_value, array = \
                convert_dataframe_into_4D_array(
                    df=df,
                    coordinate_columns=default_coordinate_columns,
                    pad_to_same_length=False,
                    reduce_to_minimal_size=True)

            voxel_arr.min_x_value = min_x_value
            voxel_arr.min_y_value = min_y_value
            voxel_arr.min_z_value = min_z_value
            voxel_arr.array = array

    return


def get_voxel_size_from_affine(affine: np.ndarray,
                               encoded: bool = False):
    """
    Compute the voxel size in all three directions from an affine contained
    within a nifti file. IMPORTANT: This function assumes the affine matrix
    to be encoded into NIFTI format.
    """
    if encoded:
        row_direction = affine[:-1, 1]
        column_direction = affine[:-1, 0]
    else:
        row_direction = affine[:-1, 0]
        column_direction = affine[:-1, 1]

    slice_direction = affine[:-1, 2]

    voxel_size = [np.linalg.norm(row_direction), np.linalg.norm(
        column_direction), np.linalg.norm(slice_direction)]

    return voxel_size


def crop_volume(volume: np.ndarray,
                filler_val: float = np.nan,
                inplace: bool = True):
    """
    Crop a volume to only the minimal area containing all non-NaN values.
    """
    if np.isnan(filler_val):
        valid_idxs = np.where(~np.isnan(volume))
    else:
        valid_idxs = np.where(volume != filler_val)
    bounding_box = np.array([np.min(valid_idxs, axis=1),
                             np.max(valid_idxs, axis=1) + 1]).T

    bbox_volume = volume if inplace else np.copy(volume)
    bbox_volume = bbox_volume[bounding_box[0, 0]:bounding_box[0, 1],
                  bounding_box[1, 0]:bounding_box[1, 1],
                  bounding_box[2, 0]:bounding_box[2, 1]]

    return bbox_volume


def replace_values(volume: np.ndarray,
                   old_val: float = np.nan,
                   new_val: float = 0.0,
                   inplace: bool = True):
    """
    Replace all occurences of a value within an array with another value.
    """
    if np.isnan(old_val):
        filler_idxs = np.isnan(volume)
    else:
        filler_idxs = volume == old_val

    replaced_volume = volume if inplace else np.copy(volume)
    replaced_volume[filler_idxs] = new_val

    return replaced_volume


def get_mask_contents(volume: np.ndarray,
                      mask: np.ndarray,
                      include_voxels_outside_mask: bool = True):
    """
    Get the bounding box of a volume, where the limits are defined by the
    bounding box of a mask.
    """

    if include_voxels_outside_mask:
        valid_idxs = np.where(~np.isnan(mask))
        bounding_box = np.array([np.min(valid_idxs, axis=1),
                                 np.max(valid_idxs, axis=1) + 1]).T

        masked_volume = np.copy(volume[bounding_box[0, 0]:bounding_box[0, 1],
                                bounding_box[1, 0]:bounding_box[1, 1],
                                bounding_box[2, 0]:bounding_box[2, 1]])
    else:
        masked_volume = np.copy(volume)
        masked_volume[np.isnan(mask)] = np.nan
        masked_volume = crop_volume(volume=masked_volume,
                                    filler_val=np.nan,
                                    inplace=True)
    return masked_volume


def extract_pet_info(info_file: str,
                     scan_name: str):
    """
    Extract the activity information from a .csv file.
    """
    if (info_file is None) | (scan_name is None):
        return np.repeat(np.nan, 5)

    if isinstance(info_file, str):
        normalization_df = pd.read_csv(info_file)
    elif isinstance(info_file, pd.Series):
        normalization_df = pd.DataFrame(info_file).T
    else:
        normalization_df = pd.DataFrame(info_file)
    scan_names = normalization_df['scan'].tolist()
    scan_row_idxs = [i for i in range(normalization_df.shape[0]) if (
            scan_names[i] in scan_name)]
    if len(scan_row_idxs) == 0:
        return np.repeat(np.nan, 5)

    # In case multiple rows were found, just consider the first one.
    scan_row = normalization_df.iloc[scan_row_idxs[0]]

    # MAKE SURE THE WEIGHT IS SPECIFIED IN kg AND NOT IN g.
    weight = cast_nan(scan_row.get('weight')) / 1000
    dose_pre = cast_nan(scan_row.get('dose_pre'))
    dose_pst = cast_nan(scan_row.get('dose_pst'))
    measure_time_pre = cast_nan(scan_row.get('measure_time_pre'))
    measure_time_pst = cast_nan(scan_row.get('measure_time_pst'))
    injection_time = cast_nan(scan_row.get('injection_time'))
    half_life = cast_nan(scan_row.get('half_life'))
    acquisition_time = recover_list_from_csv(csv_entry=cast_nan(scan_row.get(
        'timepoints')), dtype=float)
    pet_channels = recover_list_from_csv(csv_entry=cast_nan(scan_row.get(
        'pet_channels')), dtype=int)

    corrected_dose_before_injection = dose_pre * np.exp(-np.log(2) * (
            injection_time - measure_time_pre) / half_life)
    corrected_dose_after_injection = dose_pst * np.exp(np.log(2) * (
            measure_time_pst - injection_time) / half_life)

    corrected_dose_micro_curie = corrected_dose_before_injection - corrected_dose_after_injection

    # Change the units. MAKE SURE THE UNIT IS Bq AND NOT MBq!!!
    corrected_dose_Bq = corrected_dose_micro_curie * 0.037 * 1000000

    # Note: Just return one for the decay factor, as the dose was already
    # decay corrected.
    return weight, corrected_dose_Bq, 1, acquisition_time, pet_channels


def save_volume_as_images(volume: np.ndarray,
                          volume_name: str,
                          dir: str,
                          channel_idxs=None,
                          slide_orientation: str = "sagittal"):
    # Make sure to make no changes to the original volume.
    volume = np.copy(volume)
    # Make sure the volume always has 4 dimensions, for compatibility.
    if volume.ndim == 3:
        volume = volume[..., None]
    # Make sure to replace NaN.
    volume[np.isnan(volume)] = 0
    # Make sure the volume is 4-dimensional.
    if volume.ndim == 3:
        volume = volume[..., None]
    # AXIAL
    if slide_orientation.lower() == "axial":
        orientation_arr = volume
        orientation_str = "AXI"
    # CORONAL
    elif slide_orientation.lower() == "coronal":
        orientation_arr = np.einsum('zyxf->yzxf', volume)
        orientation_str = "COR"
    # SAGITTAL
    else:
        orientation_arr = np.einsum('zyxf->xzyf', volume)
        orientation_str = "SAG"

    # Normalize the volume to 0-1-range, for each channel.
    min_vals = np.nanmin(orientation_arr, axis=(0, 1, 2))
    max_vals = np.nanmax(orientation_arr, axis=(0, 1, 2))
    # Avoid division by zero.
    max_vals[max_vals == min_vals] = min_vals[max_vals == min_vals] + 1
    orientation_arr = (orientation_arr - min_vals) / (max_vals - min_vals)

    if channel_idxs is None:
        channel_idxs = range(orientation_arr.shape[-1])
    elif isinstance(channel_idxs, int):
        channel_idxs = [channel_idxs]

    files_output = []

    for channel_idx in channel_idxs:
        # Make sure the channel IDs are all correctly formatted.
        if channel_idx < 10:
            channel_str = F"00{channel_idx}"
        elif channel_idx < 100:
            channel_str = F"0{channel_idx}"
        elif channel_idx < 1000:
            channel_str = str(channel_idx)
        else:
            raise ValueError("Channel indices are only supported up to index "
                             "999.")
        for slice_idx in range(orientation_arr.shape[0]):
            # Make sure the slice IDs are all correctly formatted.
            if slice_idx < 10:
                slice_str = F"00{slice_idx}"
            elif slice_idx < 100:
                slice_str = F"0{slice_idx}"
            elif slice_idx < 1000:
                slice_str = str(slice_idx)
            else:
                raise ValueError(
                    "Slice indices are only supported up to index "
                    "999.")

            slice = orientation_arr[slice_idx, ..., channel_idx]
            img = Image.fromarray(np.uint8(slice * 255), mode="L")
            path_img = os.path.join(dir,
                                    F"{volume_name}_CHAN={channel_str}_{orientation_str}={slice_str}.bmp")
            img.save(path_img)

            files_output += [path_img]

    print(F"FINISHED SAVING THE VOLUME {volume_name} AT {dir} AS .BMP FILES.")

    return files_output


def recover_list_from_csv(csv_entry: str,
                          dtype: callable):
    """
    Turns a .csv string representation into an actual list.
    """
    # Catch the case of NaN being passed as the .csv entry.
    if isinstance(csv_entry, float):
        return csv_entry
    csv_entry = csv_entry \
        .replace("[", "").replace("]", "")
    csv_entry = list(map(dtype, csv_entry.split(" ")))

    return csv_entry


def cast_nan(val):
    """
    Makes sure to convert None to NaN.
    """
    if val is None:
        return np.nan
    else:
        return val
