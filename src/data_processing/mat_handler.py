import numpy as np
import pandas as pd
import scipy.io

from functools import reduce


def convert_mat_to_voxel_array(mat_file: str):
    mat_dict = scipy.io.loadmat(mat_file)
    metadata = ""
    arr_names = []
    for key in mat_dict.keys():
        if not isinstance(mat_dict[key], np.ndarray):
            metadata += F"{key}: {str(mat_dict[key])}\n"
        else:
            arr_names += [key]

    if len(arr_names) == 0:
        raise ValueError("The .mat file contains no array data.")

    # Show the shapes of all arrays contained within a .mat file,
    # up to dimension 4. Padding arrays with less than 4 dimensions.
    arr_shapes = [mat_dict[arr_name].shape[:4] + (1,) * max(4 - mat_dict[
        arr_name].ndim, 0) for arr_name in arr_names]
    array = np.zeros(np.max(arr_shapes, axis=0))
    features = []

    feature_idx = 0
    for arr_idx, arr_name in enumerate(arr_names):
        reshaped_arr = mat_dict[arr_name].reshape(arr_shapes[arr_idx])
        array[..., feature_idx:] = reshaped_arr

        n_features = reshaped_arr.shape[-1]
        features += [arr_name] if (n_features == 1) else [
            arr_name + F"_{i + 1}" for i in range(n_features)]
        feature_idx += n_features

    # Remove the last newline escape sequence.
    metadata = metadata[:-1]
    # Make sure the metadata is repeated for every feature.
    metadata = [metadata for _ in range(array.shape[-1])]

    return array, features, metadata


def generate_mat_dataframe_single_file(mat_file_name: str, mat_key_name: str,
                                       coordinate_columns: list = ['x', 'y',
                                                                   'z']):
    """
    Given a single .mat file, generate a single dataframe containing all the
    information of the .mat array, with its geographical properties kept
    intact due to coordinate columns in the output dataframe.
    :param mat_file_name: The file name of the .mat file that is to be
    processed.
    :param mat_key_name: The key name for the array in the .mat dictionary.
    :param coordinate_columns: The names given to the coordinate columns in
    the output dataframe.
    :return: The output dataframe, containing all the information from the
    .mat array.
    """
    # A check to guarantee reliable behavior of the function.
    if len(coordinate_columns) != 3:
        return ValueError(
            "Exactly three names for coordinate columns must be provided.")
    mat_arr: np.ndarray = scipy.io.loadmat(mat_file_name)[mat_key_name]
    df_columns = coordinate_columns + [
        mat_key_name] if mat_arr.ndim != 4 else coordinate_columns + [
        mat_key_name + F" {i + 1}" for i in range(mat_arr.shape[3])]

    rows: list = []

    for z in range(mat_arr.shape[0]):
        for y in range(mat_arr.shape[1]):
            for x in range(mat_arr.shape[2]):
                value = mat_arr[z, y, x]
                if isinstance(value, np.ndarray):
                    # Check that the 4th dimension isn't just NaNs.
                    if sum([not np.isnan(one_dim_value) for one_dim_value in
                            value]):
                        row = [x, y, z] + list(value)
                        rows.append(row)
                elif not np.isnan(value):
                    row = [x, y, z] + [value]
                    rows.append(row)

    result_df: pd.DataFrame = pd.DataFrame(rows, columns=df_columns)

    return result_df


def generate_mat_dataframe_single_animal(mat_file_names: list,
                                         mat_key_names: list,
                                         coordinate_columns: list = ['x',
                                                                     'y', 'z']):
    """
    Given several .mat files, all describing the same animal, generate a single
    dataframe containing all the
    information of the .mat arrays, with their geographical properties kept
    intact due to coordinate columns in the output dataframe.
    :param mat_file_names: The file names of the .mat files that are to be
    processed.
    :param mat_key_names: The key names for the arrays in the .mat dictionaries.
    :param coordinate_columns: The names given to the coordinate columns in
    the output dataframe.
    :return: The output dataframe, containing all the information from the
    .mat arrays.
    """
    # Some checks to guarantee reliable behavior of the function.
    if len(mat_file_names) != len(mat_key_names):
        raise ValueError(
            "The number of .mat file names is not equal to the number of key "
            "names.")
    elif len(set(mat_file_names)) != len(mat_file_names):
        raise ValueError("Duplicate .mat file names provided.")
    elif len(set(mat_key_names)) != len(mat_key_names):
        raise ValueError("Duplicate key names provided.")

    df_bundle_arr: list = []

    for mat_file_name, mat_key_name in zip(mat_file_names, mat_key_names):
        df_bundle_arr.append(generate_mat_dataframe_single_file(
            mat_file_name=mat_file_name, mat_key_name=mat_key_name))

    return reduce(lambda first_df, rest_df: pd.merge(first_df, rest_df,
                                                     on=coordinate_columns,
                                                     how='outer'),
                  df_bundle_arr)


def generate_mat_dataframe_multiple_animals(mat_file_names_collection: list,
                                            mat_key_names: list,
                                            coordinate_columns: list = ['x',
                                                                        'y',
                                                                        'z'],
                                            output_file_names=None):
    """
    Given a collection of several .mat files, each element describing one
    animal, generate several dataframes containing all the
    information of the .mat arrays, one dataframe per animal, with their
    geographical properties kept
    intact due to coordinate columns in the output dataframes. In addition to
    being returned, the dataframes are saved as .csv files under a given path.
    :param mat_file_names_collection: List of lists of the file names of the
    .mat files that are to be processed.
    :param mat_key_names: The key names for the arrays in the .mat
    dictionaries, to be used universally for each animal.
    :param coordinate_columns: The names given to the coordinate columns in
    the output dataframe, to be used universally for each animal.
    :param output_file_names: The names for the .csv file names. If none are
    provided, do not save the dataframes as .csv files and instead only
    return them.
    :return: The dataframes as a list of dataframes, one dataframe per each
    animal. If file names were provided, also save each dataframe as a .csv
    file.
    """
    # Some checks to guarantee reliable behavior of the function.
    if len(set(map(len, mat_file_names_collection))) > 1:
        raise ValueError("All animals must provide the same amount of .mat "
                         "file names.")
    elif len(mat_file_names_collection[0]) != len(mat_key_names):
        raise ValueError(
            "The number of .mat file names per animal is not equal to "
            "the number of key names.")

    # Collector for the final output.
    animal_df_collection: list = []

    for animal_index, mat_file_names in enumerate(
            mat_file_names_collection):
        animal_df: pd.DataFrame = generate_mat_dataframe_single_animal(
            mat_file_names=mat_file_names, mat_key_names=mat_key_names,
            coordinate_columns=coordinate_columns)
        if output_file_names is not None:
            animal_df.to_csv(output_file_names[animal_index], index=False)
        animal_df_collection.append(animal_df)

    return animal_df_collection
