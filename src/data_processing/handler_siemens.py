import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from functools import reduce
import warnings

from scipy.interpolate import griddata

import handler_nifti

"""
def convert_df_to_volume(df: pd.DataFrame,
                         voxel_size=None,
                         mask_padding: int = 10):
    eps = 1e-7
    n_decimals = 4
    n_samples = 10

    dist_scores = []
    directions_samples = []

    samples = df[np.random.choice(a=df.shape[0],
                                  size=min(n_samples, df.shape[0]),
                                  replace=False)]

    for sample in samples:
        distances = df - sample
        distances = np.sum(distances * distances, axis=1)
        order = np.argsort(distances)

        distances = distances[order]
        neighbors = df[order]

        # Make sure that the sample doesn't consider itself for the minimal
        # neighbors.
        neighbors = neighbors[distances > 0]

        direction_1 = neighbors[0] - sample

        for neighbor in neighbors[1:]:
            direction_2 = neighbor - sample
            if np.abs(direction_1 @ direction_2) < eps:
                dist_scores.append(np.linalg.norm(
                    direction_1) + np.linalg.norm(direction_2))
                directions_samples.append([direction_1, direction_2])
                break
        else:
            # Make sure to ignore the direction cosines of this sample if the
            # cosines are not orthogonal.
            dist_scores.append(np.infty)
            directions_samples.append([direction_1, direction_2])

    # Determine the sample with the biggest density, i.e. that has the
    # smallest distance to its two closest neighbors in two directions.
    densest_idx = np.argmin(dist_scores)
    direction_1, direction_2 = directions_samples[densest_idx]
    print(directions_samples)
    print(direction_1)
    print(direction_2)
    direction_3 = np.cross(direction_1, direction_2)

    direction_1 = direction_1 / np.linalg.norm(direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    direction_3 = direction_3 / np.linalg.norm(direction_3)

    directions_123 = [direction_1, direction_2, direction_3]

    # The canonical basis vectors.
    e_x = [1, 0, 0]
    e_y = [0, 1, 0]
    e_z = [0, 0, 1]
    canon = [e_x, e_y, e_z]

    # Helper function to get the angle between two vectors.
    vector_angle = lambda v1: lambda v2: np.arccos(np.abs(v1 @ v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Estimate which of the direction cosines corresponds to which anatomical
    # direction.
    x_idx = np.argmin(list(map(vector_angle(e_x), directions_123)))
    direction_x = directions_123[x_idx]
    directions_123.pop(x_idx)

    y_idx = np.argmin(list(map(vector_angle(e_y), directions_123)))
    direction_y = directions_123[y_idx]
    directions_123.pop(y_idx)

    direction_z = directions_123[0]
    directions_xyz = [direction_x, direction_y, direction_z]
    # Make sure all direction cosines point in the correct direction.
    directions_xyz = np.array([(np.sign(e @ direction)) * direction
                               for e, direction in zip(canon, directions_xyz)])

    # Execute a basis change from basis B1 (basis of physical
    # coordinates) to basis B2 (basis of modality array).
    A = directions_xyz.T
    basis_change_matrix = np.linalg.inv(A)
    offset = df[0]

    affine_coords = np.dot(basis_change_matrix, (df - offset).T).T

    if voxel_size is None:
        sorted_coords = np.round(np.sort(affine_coords, axis=1), n_decimals)
        sorted_coords = [np.unique(sorted_coords[:, i]) for i in range(
            sorted_coords.shape[1])]
        sorted_coords = [dim_coords - np.roll(dim_coords, shift=1) for
                         dim_coords in
                         sorted_coords]
        voxel_size = np.round([np.min(np.abs(dim_coords), axis=0) for \
                               dim_coords in sorted_coords], n_decimals)
        # Prevent Division-By-Zero errors.
        voxel_size[voxel_size == 0] = 1

    # Make sure the coordinates and direction vectors are compatible with the
    # array indices.
    affine_coords = np.round(affine_coords / voxel_size, n_decimals)
    directions_xyz = np.round(directions_xyz * voxel_size, n_decimals)
    A = directions_xyz.T

    min_values = np.min(affine_coords, axis=0)
    max_values = np.max(affine_coords, axis=0)

    # NOTE: The padding is done in order to not create a minimal volume
    # containing all the ROIs, but one with some filler values on all ends,
    # to ensure that during interpolation the edge ROI voxels will not be
    # considered as out of the grid. Please be aware that the filler value
    # MUST be a numeric value and not NaN, as otherwise interpolation of
    # edge voxels will always result in NAN.
    dim_cuboid = (max_values - min_values + 1 + mask_padding).astype(int)
    # Return the dimensions from xyz to the numpy-typical ordering of zyx.
    dim_cuboid[[0, 1, 2]] = dim_cuboid[[2, 1, 0]]

    # With this step, the coordinates are ordered z-y-x.
    cuboid_coords = np.vstack(np.where(np.ones(dim_cuboid))).T
    # Make sure the coordinates are transformed to the DICOM ordering of x-y-z.
    cuboid_coords[:, [0, 1, 2]] = cuboid_coords[:, [2, 1, 0]] + min_values - \
                                  (mask_padding // 2)
    cuboid_coords = np.dot(A, cuboid_coords.T).T + offset

    # Transform the result into a dataframe.
    cuboid_df = pd.DataFrame(cuboid_coords, columns=df.columns)

    # Compute an array, with the same shape as the cuboid, storing the
    # information about which voxels are occupied by which ROI. Note that if
    # two ROIs share voxels, then those voxels will be labelled as belonging
    # to the latter ROI. Make it four-dimensional in order to comply with the
    # DICOM interpolation.
    mask_arr = np.zeros(tuple(dim_cuboid) + (1,))

    # Get the coordinates for each ROI voxel in terms of array indexing.
    arr_coords = affine_coords - min_values + (mask_padding // 2)
    arr_coords = arr_coords[..., [2, 1, 0]].astype(int)

    # The coordinates of the voxels of an individual ROI. Note that the
    # coordinates have to be read in column-wise, to fit the format of
    # np.where.
    mask_arr[arr_coords[..., 0],
             arr_coords[..., 1],
             arr_coords[..., 2]] = 1

    return cuboid_df, mask_arr, voxel_size, dim_cuboid, directions_xyz
"""

COORDINATE_COLUMNS = ['x', 'y', 'z']


def is_siemens(path: str):
    """
    Very rough check to see whether a file is a Siemens ROI .csv file.
    """
    try:
        roi_file = open(path).read()
        lines = roi_file.split("\n")
        for line in lines:
            if "ROI Name:" in line:
                return True
        else:
            return False

    # UnicodeDecodeError for non-text files and PermissionError for directories.
    except (UnicodeDecodeError, PermissionError, IsADirectoryError):
        return False


def convert_siemens_to_df(siemens_file: str):
    """
    Convert a single Siemens .csv file into a usable voxel dataframe.
    """
    roi_identifier = "ROI Name: "
    frame_identifier = "Frame:, "
    units_identifier = "Units:,"
    coordinate_columns = ['x', 'y', 'z']

    feature_name = "MR"
    roi_df = None
    acquisition_times = []
    roi_df_list = []
    roi_name_list = []

    roi_idx = -1
    frame_idx = -1

    frame_rows = []

    # If a UnicodeDecodeError is being raised here, the file cannot be read
    # as a Siemens file.
    file_str = open(siemens_file).read()
    lines = file_str.split("\n")

    # This whole dance with i has to be done so that the processing of the
    # last roi will truly only happen after every single line has been read in.
    for i in range(len(lines) + 1):
        # Is being called either if a new ROI is found or if the end of
        # the file is reached.
        try:
            line = lines[i]
        except IndexError:
            line = ""
        if units_identifier in line:
            if "Bq/ml" in line:
                feature_name = "PT"
        elif (roi_identifier in line) | (i == len(lines)):
            roi_idx += 1

            # Add the name of the newly found ROI to a list of all ROI names.
            # But ONLY if the end of the document was not reached.
            if i < len(lines):
                roi_name = line.split(roi_identifier)[-1]
                roi_name_list.append(roi_name)

            if roi_idx > 0:

                # In case only one frame was found, create the roi dataframe
                # here.
                if frame_idx == -1:
                    roi_df = create_rounded_coordinates_df(frame_rows,
                                                           coordinate_columns,
                                                           feature_name)
                else:
                    roi_df.columns = COORDINATE_COLUMNS + [F"Channel_"
                                                           F"{0 if i < 9 else ''}"
                                                           F"{i + 1}" for i in
                                                           range(
                                                               frame_idx)]

                # Append the results of different ROIs, to create one
                # list containing the dataframes for all the ROIs of a patient.
                roi_df_list += [roi_df.astype(float)]

            frame_idx = -1
            roi_df = None

            # Delete list of current rows.
            frame_rows = []

        # Is being called when the beginning of a new frame is found.
        elif frame_identifier in line:
            frame_idx += 1

            # Determine the acquisition times of the individual frames.
            if roi_idx == 0:
                acquisition_time_str = \
                    line.split("Frame Time:, ")[-1].split(",")[0]
                acquisition_time_h = int(acquisition_time_str.split(":")[0])
                acquisition_time_min = int(acquisition_time_str.split(":")[1])
                acquisition_time_s = int(acquisition_time_str.split(":")[2])

                acquisition_time = acquisition_time_h * 3600 \
                                   + acquisition_time_min * 60 \
                                   + acquisition_time_s * 1

                acquisition_times.append(acquisition_time)

            # Only convert the list of frame rows into a df if there are
            # more than zero rows, that is, it is not the very first
            # frame of an ROI.
            if frame_idx > 0:
                if roi_df is None:
                    roi_df = pd.DataFrame(frame_rows)
                else:
                    roi_df = pd.concat([roi_df, pd.DataFrame(
                        frame_rows).iloc[:, 3:]], axis=1)

                # Delete list of current rows.
                frame_rows = []

        # Is being called when a content row
        # describing a voxel is recognized.
        # Assume that there will always be at least 4 columns to be read
        # in from the source .csv file.
        elif len(line.split(",")) > 3:
            frame_rows.append(line.split(","))

    # Make sure that acquisition times is always a non-empty array.
    if len(acquisition_times) == 0:
        acquisition_times = np.array([0])

    return roi_df_list, roi_name_list, acquisition_times


def convert_df_to_volume(df: pd.DataFrame,
                         padding=None):
    """
    Extend a ROI to be of the minimal cuboid around the original ROI.
    """
    EPS = 1e-7
    N_DECIMALS = 5
    N_SAMPLES = 10
    padding = 0 if (padding is None) else padding

    coords = df[COORDINATE_COLUMNS].to_numpy()
    vals = df.drop(labels=COORDINATE_COLUMNS, inplace=False, axis=1).to_numpy()
    # Insert filler values, in case only the coordinates were specified.
    if vals.shape[0] == 0:
        vals = np.ones(coords.shape[0])

    dist_scores = []
    directions_samples = []

    samples = coords[np.random.choice(
        a=coords.shape[0],
        size=min(N_SAMPLES, coords.shape[0]),
        replace=False)]

    for sample in samples:
        distances = coords - sample
        distances = np.sum(distances * distances, axis=1)
        order = np.argsort(distances)

        distances = distances[order]
        neighbors = coords[order]

        # Make sure that the sample doesn't consider itself for the minimal
        # neighbors.
        neighbors = neighbors[distances > 0]

        direction_1 = neighbors[0] - sample

        for neighbor in neighbors[1:]:
            direction_2 = neighbor - sample
            if np.abs(direction_1 @ direction_2) < EPS:
                dist_scores.append(np.linalg.norm(
                    direction_1) + np.linalg.norm(direction_2))
                directions_samples.append([direction_1, direction_2])
                break
        else:
            # Make sure to ignore the direction cosines of this sample if the
            # cosines are not orthogonal.
            dist_scores.append(np.infty)
            directions_samples.append([direction_1, direction_2])

    # Determine the sample with the biggest density, i.e. that has the
    # smallest distance to its two closest neighbors in two directions.
    densest_idx = np.argmin(dist_scores)
    direction_1, direction_2 = directions_samples[densest_idx]
    direction_3 = np.cross(direction_1, direction_2)

    direction_1 = direction_1 / np.linalg.norm(direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    direction_3 = direction_3 / np.linalg.norm(direction_3)

    directions_123 = [direction_1, direction_2, direction_3]

    # The canonical basis vectors.
    e_x = [1, 0, 0]
    e_y = [0, 1, 0]
    e_z = [0, 0, 1]
    canon = [e_x, e_y, e_z]

    # Helper function to get the angle between two vectors.
    vector_angle = lambda v1: lambda v2: np.arccos(np.abs(v1 @ v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Estimate which of the direction cosines corresponds to which anatomical
    # direction.
    x_idx = np.argmin(list(map(vector_angle(e_x), directions_123)))
    direction_x = directions_123[x_idx]
    directions_123.pop(x_idx)

    y_idx = np.argmin(list(map(vector_angle(e_y), directions_123)))
    direction_y = directions_123[y_idx]
    directions_123.pop(y_idx)

    direction_z = directions_123[0]
    directions_xyz = [direction_x, direction_y, direction_z]
    # Make sure all direction cosines point in the correct direction and have
    # unit length.
    directions_xyz = np.array([(np.sign(e @ direction)) *
                               direction / np.linalg.norm(direction)
                               for e, direction in zip(canon, directions_xyz)])

    # Execute a basis change from basis B1 (basis of physical
    # coordinates) to basis B2 (basis of modality array).
    A = directions_xyz.T
    basis_change_matrix = np.linalg.inv(A)
    offset = coords[0]

    affine_coords = np.round(np.dot(basis_change_matrix,
                                    (coords - offset).T).T,
                             N_DECIMALS)

    # Determine the voxel size of the volume.
    unique_coords = [np.sort(np.unique(affine_coords[:, i])) for i in
                     range(affine_coords.shape[1])]
    voxel_size = [np.min(np.abs(u - np.roll(u, 1))) for u in unique_coords]

    # Make sure the coordinates and direction vectors are compatible with the
    # array indices.
    affine_coords = np.round(affine_coords / voxel_size, 0)
    directions_xyz = directions_xyz * voxel_size
    A = directions_xyz.T

    # NOTE: The padding is done in order to not create a minimal volume
    # containing all the ROIs, but one with some filler values on all ends,
    # to ensure that during interpolation the edge ROI voxels will not be
    # considered as out of the grid. Please be aware that the filler value
    # MUST be a numeric value and not NaN, as otherwise interpolation of
    # edge voxels will always result in NAN.
    min_values = np.min(affine_coords, axis=0)
    max_values = np.max(affine_coords, axis=0)

    # Making sure that zero is the smallest coordinate in any direction.
    affine_coords = (affine_coords - min_values).astype(int)

    # Defining the shapes of the volumes.
    shape_volume = tuple((max_values - min_values + 1).astype(int)) \
                   + (vals.shape[1],)
    shape_volume_padded = tuple(
        np.array(shape_volume[:3]) + padding) + shape_volume[3:]

    # The minimal volume.
    volume = np.zeros(shape_volume)
    volume[:] = np.nan
    volume[affine_coords[:, 0], affine_coords[:, 1], affine_coords[:, 2]] = vals

    # The padded volume.
    volume_padded = np.zeros(shape_volume_padded)
    volume_padded[:] = np.nan
    volume_padded[
    padding // 2:padding // 2 + volume.shape[0],
    padding // 2:padding // 2 + volume.shape[1],
    padding // 2:padding // 2 + volume.shape[2]] = volume

    # Make sure to transform the volume to the default numpy orientation.
    volume_padded = np.einsum("xyz...->zyx...", volume_padded)
    image_position_patient = (A @ (min_values - (padding // 2))) + offset
    image_orientation_patient = np.array(directions_xyz[:2]).flatten()

    affine = handler_nifti.get_affine(
        image_orientation_patient=image_orientation_patient,
        image_position_patient=image_position_patient,
        pixel_spacing=voxel_size[:2],
        slice_thickness=voxel_size[2])

    return volume_padded, affine


def convert_siemens_to_volume(siemens_file: str,
                              mask_only: bool = False,
                              padding=None):
    roi_df_list, roi_name_list, acquisition_times = convert_siemens_to_df(
        siemens_file=siemens_file)

    # Assign each ROI an incrementing number as the only voxel of each ROI to
    # represent a mask of the ROI.
    if mask_only:
        roi_df_list = [roi_df_list[i][COORDINATE_COLUMNS]
                       .assign(v=np.repeat(i + 1, roi_df_list[i].shape[0]))
                       for i in range(len(roi_df_list))]

    merged_df = pd.concat(roi_df_list)
    volume, affine = convert_df_to_volume(df=merged_df, padding=padding)

    return volume, affine, roi_name_list, acquisition_times


def convert_siemens_to_volume_old(siemens_file: str,
                                  padding=None):
    """
    Convert a single Siemens .csv file into a usable voxel dataframe.
    """
    roi_identifier = "ROI Name: "
    frame_identifier = "Frame:, "
    units_identifier = "Units:,"
    coordinate_columns = ['x', 'y', 'z']

    feature_name = "MR"
    master_df = None
    roi_df = None
    acquisition_times = []
    roi_names = []

    roi_idx = -1
    frame_idx = -1

    frame_rows = []

    # If a UnicodeDecodeError is being raised here, the file cannot be read
    # as a Siemens file.
    file_str = open(siemens_file).read()
    lines = file_str.split("\n")

    # This whole dance with i has to be done so that the processing of the
    # last roi will truly only happen after every single line has been read in.
    for i in range(len(lines) + 1):
        # Is being called either if a new ROI is found or if the end of
        # the file is reached.
        try:
            line = lines[i]
        except IndexError:
            line = ""
        if units_identifier in line:
            if "Bq/ml" in line:
                feature_name = "PT"
        elif (roi_identifier in line) | (i == len(lines)):
            roi_idx += 1

            # Add the name of the newly found ROI to a list of all ROI names.
            roi_name = line.split(roi_identifier)[-1]
            roi_names.append(roi_name)

            if roi_idx > 0:

                # In case only one frame was found, create the roi dataframe
                # here.
                if frame_idx == -1:
                    roi_df = create_rounded_coordinates_df(frame_rows,
                                                           coordinate_columns,
                                                           feature_name)

                # Concatenate the results of different ROIs, to create one
                # big dataframe for all the ROIs of a patient.
                if master_df is None:
                    master_df = roi_df
                else:
                    master_df = pd.concat([master_df, roi_df], axis=0)

            frame_idx = -1
            roi_df = None

            # Delete list of current rows.
            frame_rows = []

        # Is being called when the beginning of a new frame is found.
        elif frame_identifier in line:
            frame_idx += 1

            # Determine the acquisition times of the individual frames.
            if roi_idx == 0:
                acquisition_time_str = \
                    line.split("Frame Time:, ")[-1].split(",")[0]
                acquisition_time_h = int(acquisition_time_str.split(":")[0])
                acquisition_time_min = int(
                    acquisition_time_str.split(":")[1])
                acquisition_time_s = int(acquisition_time_str.split(":")[2])

                acquisition_time = acquisition_time_h * 3600 \
                                   + acquisition_time_min * 60 \
                                   + acquisition_time_s * 1

                acquisition_times.append(acquisition_time)

            # Only convert the list of frame rows into a df if there are
            # more than zero rows, that is, it is not the very first
            # frame of an ROI.
            if frame_idx > 0:
                if roi_df is None:
                    roi_df = create_rounded_coordinates_df(frame_rows,
                                                           coordinate_columns,
                                                           feature_name +
                                                           F"_Frame_"
                                                           F"{frame_idx}")
                else:
                    roi_df = pd.concat([roi_df, pd.DataFrame(
                        frame_rows).iloc[:, 3:]], axis=1)

                # Delete list of current rows.
                frame_rows = []

        # Is being called when a content row
        # describing a voxel is recognized.
        # Assume that there will always be at least 4 columns to be read
        # in from the source .csv file.
        elif len(line.split(",")) > 3:
            frame_rows.append(line.split(","))

    if master_df.shape[1] <= len(coordinate_columns) + 1:
        master_df.columns = coordinate_columns + [feature_name]
    else:
        master_df.columns = coordinate_columns \
                            + [feature_name + F"_Frame_{i + 1}" for i in
                               range(master_df.shape[1] - len(
                                   coordinate_columns))]

    # Delete the index column, in place.
    master_df.reset_index(drop=True, inplace=True)

    # Computing a volume from the dataframe of coordinates.
    volume, affine = convert_df_to_volume(df=master_df,
                                          padding=padding)

    # The name assigned to the Siemens file.
    name = siemens_file.split("\\")[-1].split("/")[-1].split(".")[0]

    # Make sure that acquisition times is always a non-empty array.
    if len(acquisition_times) == 0:
        acquisition_times = np.array([0])

    return volume, affine, acquisition_times, name


def extract_cuboid_ROI(siemens_file: str,
                       voxel_size=None,
                       output_dir=None,
                       merge_ROIs: bool = False):
    """
   Convert a single Siemens .csv file into one or several cuboid ROI .csv files.
   """
    n_decimals = 10

    roi_collection = []

    roi_identifier = "ROI Name: "
    frame_identifier = "Frame:, "
    coordinate_columns = ['x', 'y', 'z']

    roi_df = None

    roi_idx = -1
    frame_idx = -1

    frame_rows = []

    # If a UnicodeDecodeError is being raised here, the file cannot be read
    # as a Siemens file.
    file_str = open(siemens_file).read()
    lines = file_str.split("\n")

    # This whole dance with i has to be done so that the processing of the
    # last roi will truly only happen after every single line has been read in.
    for i in range(len(lines) + 1):
        # Is being called either if a new ROI is found or if the end of
        # the file is reached.
        try:
            line = lines[i]
        except IndexError:
            line = ""
        if (roi_identifier in line) | (i == len(lines)):
            roi_idx += 1

            if roi_idx > 0:

                # In case only one frame was found, create the roi dataframe
                # here.
                if frame_idx == -1:
                    roi_df = create_rounded_coordinates_df(frame_rows,
                                                           coordinate_columns,
                                                           n_decimals=n_decimals)
                roi_collection.append((roi_df, roi_name))

            # Obviously, when the end of the file was reached, the ROI name
            # does not need to be updated.
            if i != len(lines):
                try:
                    roi_name = line.split(roi_identifier)[-1]
                except IndexError:
                    roi_name = ""

                if len(roi_name) == 0:
                    roi_name = str(roi_idx + 1)

            frame_idx = -1
            roi_df = None

            # Delete list of current rows.
            frame_rows = []

        # Is being called when the beginning of a new frame is found.
        elif frame_identifier in line:
            frame_idx += 1

            # Only convert the list of frame rows into a df if there are
            # more than zero rows, that is, it is not the very first
            # frame of an ROI.
            if frame_idx > 0:
                if roi_df is None:
                    roi_df = create_rounded_coordinates_df(frame_rows,
                                                           coordinate_columns,
                                                           n_decimals=n_decimals)
                else:
                    frame_df = create_rounded_coordinates_df(frame_rows,
                                                             coordinate_columns,
                                                             n_decimals=n_decimals)
                    roi_df = roi_df.merge(frame_df,
                                          how="outer",
                                          on=coordinate_columns)

                # Delete list of current rows.
                frame_rows = []

        # Is being called when a content row
        # describing a voxel is recognized.
        # Assume that there will always be at least 4 columns to be read
        # in from the source .csv file.
        elif len(line.split(",")) > 3:
            frame_rows.append(line.split(",")[:3])

    # For the creation of whole-slice masks, pass all ROI dataframes at once
    # to the function creating the mask arrays.
    if merge_ROIs:
        roi_names = [roi_name for _, roi_name in roi_collection]
        roi_collection = [roi_df for roi_df, _ in roi_collection]
        roi_collection = [extend_ROI_to_cuboid(roi_collection, voxel_size)
                          + (roi_names,)]
    else:
        roi_collection = [
            extend_ROI_to_cuboid(roi_df, voxel_size) + ([roi_name],)
            for roi_df, roi_name in roi_collection]

    # In case the ROI coordinates are supposed to be saved.
    if output_dir is not None:
        base_name = siemens_file.split("\\")[-1].split("/")[-1].split(".")[
            0]
        for roi_df, _a, _b, _c, _d, roi_name in roi_collection:
            file = os.path.join(output_dir,
                                base_name + "_" + roi_name + ".csv")
            roi_df.to_csv(file, index=False)

    return roi_collection


def extend_ROI_to_cuboid(roi_dfs,
                         voxel_size=None,
                         mask_padding: int = 10):
    """
    Extend a ROI to be of the minimal cuboid around the original ROI.
    """
    eps = 1e-7
    n_decimals = 5
    n_samples = 10

    # Make sure that the function also works if just a single dataframe
    # without a surrounding list is passed.
    if isinstance(roi_dfs, pd.DataFrame):
        roi_dfs = [roi_dfs]

    # The length of the individual roi dataframes, in order to separate them
    # later on, to get the roi masks.
    len_per_roi = [roi_df.shape[0] for roi_df in roi_dfs]

    # Create one big array containing the coordinates of all ROIs.
    roi_coords = pd.concat(roi_dfs, axis=0).to_numpy()

    dist_scores = []
    directions_samples = []

    samples = roi_coords[np.random.choice(a=roi_coords.shape[0],
                                          size=min(n_samples,
                                                   roi_coords.shape[0]),
                                          replace=False)]

    for sample in samples:
        distances = roi_coords - sample
        distances = np.sum(distances * distances, axis=1)
        order = np.argsort(distances)

        distances = distances[order]
        neighbors = roi_coords[order]

        # Make sure that the sample doesn't consider itself for the minimal
        # neighbors.
        neighbors = neighbors[distances > 0]

        direction_1 = neighbors[0] - sample

        for neighbor in neighbors[1:]:
            direction_2 = neighbor - sample
            if np.abs(direction_1 @ direction_2) < eps:
                dist_scores.append(np.linalg.norm(
                    direction_1) + np.linalg.norm(direction_2))
                directions_samples.append([direction_1, direction_2])
                break
        else:
            # Make sure to ignore the direction cosines of this sample if the
            # cosines are not orthogonal.
            dist_scores.append(np.infty)
            directions_samples.append([direction_1, direction_2])

    # Determine the sample with the biggest density, i.e. that has the
    # smallest distance to its two closest neighbors in two directions.
    densest_idx = np.argmin(dist_scores)
    direction_1, direction_2 = directions_samples[densest_idx]
    direction_3 = np.cross(direction_1, direction_2)

    direction_1 = direction_1 / np.linalg.norm(direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    direction_3 = direction_3 / np.linalg.norm(direction_3)

    directions_123 = [direction_1, direction_2, direction_3]

    # The canonical basis vectors.
    e_x = [1, 0, 0]
    e_y = [0, 1, 0]
    e_z = [0, 0, 1]
    canon = [e_x, e_y, e_z]

    # Helper function to get the angle between two vectors.
    vector_angle = lambda v1: lambda v2: np.arccos(np.abs(v1 @ v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Estimate which of the direction cosines corresponds to which anatomical
    # direction.
    x_idx = np.argmin(list(map(vector_angle(e_x), directions_123)))
    direction_x = directions_123[x_idx]
    directions_123.pop(x_idx)

    y_idx = np.argmin(list(map(vector_angle(e_y), directions_123)))
    direction_y = directions_123[y_idx]
    directions_123.pop(y_idx)

    direction_z = directions_123[0]
    directions_xyz = [direction_x, direction_y, direction_z]
    # Make sure all direction cosines point in the correct direction.
    directions_xyz = np.array([(np.sign(e @ direction)) * direction
                               for e, direction in
                               zip(canon, directions_xyz)])

    # Execute a basis change from basis B1 (basis of physical
    # coordinates) to basis B2 (basis of modality array).
    A = directions_xyz.T
    basis_change_matrix = np.linalg.inv(A)
    offset = roi_coords[0]

    affine_coords = np.dot(basis_change_matrix, (roi_coords - offset).T).T

    if voxel_size is None:
        unique_coords = np.round(np.sort(affine_coords, axis=1), n_decimals)
        unique_coords = [np.unique(unique_coords[:, i]) for i in range(
            unique_coords.shape[1])]
        delta_coords = [column_coords - np.roll(column_coords, shift=1) for
                        column_coords in unique_coords]
        # Throw out the voxel offsets where a rounding error occurred.
        delta_coords = [np.round(column_coords[column_coords > 10 ** (
                -n_decimals + 1)], n_decimals)
                        for column_coords in delta_coords]
        voxel_size = np.zeros((3,))
        for dim_idx, dim_coords in enumerate(delta_coords):
            values, counts = np.unique(dim_coords, return_counts=True)
            voxel_size[dim_idx] = values[np.argmax(counts)]
            voxel_size[dim_idx] = np.mean(dim_coords)
        # Prevent Division-By-Zero errors.
        voxel_size[voxel_size == 0] = 1
        voxel_size = np.array([0.2504054054, 0.2504054054, 0.2504054054])

    # Make sure the coordinates and direction vectors are compatible with the
    # array indices.
    print("AFFINE COORDS")
    print(affine_coords)
    affine_coords = np.round(affine_coords / voxel_size, 0)
    print(affine_coords)
    print(directions_xyz)
    directions_xyz = directions_xyz * voxel_size
    print(directions_xyz)
    A = directions_xyz.T

    min_values = np.min(affine_coords, axis=0)
    max_values = np.max(affine_coords, axis=0)

    # NOTE: The padding is done in order to not create a minimal volume
    # containing all the ROIs, but one with some filler values on all ends,
    # to ensure that during interpolation the edge ROI voxels will not be
    # considered as out of the grid. Please be aware that the filler value
    # MUST be a numeric value and not NaN, as otherwise interpolation of
    # edge voxels will always result in NAN.
    dim_cuboid = (max_values - min_values + 1 + mask_padding).astype(int)
    # Return the dimensions from xyz to the numpy-typical ordering of zyx.
    dim_cuboid[[0, 1, 2]] = dim_cuboid[[2, 1, 0]]
    print(max_values)
    print(min_values)
    print(dim_cuboid)
    print(voxel_size)

    # With this step, the coordinates are ordered z-y-x.
    cuboid_coords = np.vstack(np.where(np.ones(dim_cuboid))).T
    # Make sure the coordinates are transformed to the DICOM ordering of x-y-z.
    cuboid_coords[:, [0, 1, 2]] = cuboid_coords[:, [2, 1, 0]] + min_values - \
                                  (mask_padding // 2)
    cuboid_coords = np.dot(A, cuboid_coords.T).T + offset

    # Transform the result into a dataframe.
    cuboid_df = pd.DataFrame(cuboid_coords, columns=roi_dfs[0].columns)

    # Compute an array, with the same shape as the cuboid, storing the
    # information about which voxels are occupied by which ROI. Note that if
    # two ROIs share voxels, then those voxels will be labelled as belonging
    # to the latter ROI. Make it four-dimensional in order to comply with the
    # DICOM interpolation.
    mask_arr = np.zeros(tuple(dim_cuboid) + (1,))
    # Note that -1 can always be assumed as a non-conflicting filler value,
    # as the ROI indices always start at 0 and get incremented.
    mask_arr[:] = -1

    # Get the coordinates for each ROI voxel in terms of array indexing.
    arr_coords = affine_coords - min_values + (mask_padding // 2)
    arr_coords = arr_coords[..., [2, 1, 0]].astype(int)

    cumulative = 0
    for roi_id, len_roi in enumerate(len_per_roi):
        # The coordinates of the voxels of an individual ROI. Note that the
        # coordinates have to be read in column-wise, to fit the format of
        # np.where.
        roi_arr_coords = arr_coords[cumulative:cumulative + len_roi]
        mask_arr[roi_arr_coords[..., 0],
        roi_arr_coords[..., 1],
        roi_arr_coords[..., 2]] = roi_id

        cumulative += len_roi

    # TODO: Fix this.

    return cuboid_df, mask_arr, voxel_size, dim_cuboid, directions_xyz


def generate_master_csv(file_couples: List[Tuple[str, str]],
                        output_directories: List[str],
                        new_output_names: List[str],
                        suv_info_path: Optional[str],
                        ordinalize_coords: bool = True,
                        roi_identifier: str = "ROI Name: ",
                        roi_common_denominator: str = "umor",
                        frame_identifier: str = "Frame:, ",
                        file_name_suffix_MR: str = " - MR",
                        file_name_suffix_PET: str = " - PET",
                        roi_signifier: str = " Tumor",
                        column_name_mr: str = "T2",
                        idcc_column_name_suffix: str = "_IDCC",
                        suv_column_name_suffix: str = "_SUV",
                        name_signifier: str = " - Features",
                        coordinate_columns: list = ['x', 'y', 'z']):
    """
    Reads in couples of MR/PET .csv data from the Siemens software
    and writes it into one shared .csv file that can natively be read in
    as a dataframe.
    :param file_couples: List of tuples, containing first the name of the MR
    .csv file, followed by the name of the PET .csv file.
    :param output_directory: Name of directory where the information is
    supposed to be saved.
    :param roi_identifier: String by which to recognize a line as the
    beginning of a new ROI.
    :param roi_common_denominator: String by which to split the ROI name in
    order to extract the unique part of the name.
    :param frame_identifier: String by which to recognize a line as the
    beginning of a new frame.
    :param file_name_suffix_MR: String by which to split the MR file name
    in order to extract the animal name.
    :param file_name_suffix_PET: String by which to split the PET file name
    in order to extract the animal name.
    :param roi_signifier: Suffix to add to the animal name to signal the
    beginning of the ROI name.
    :param column_name_mr: Name to be given to the MR column in the output
    .csv file.
    :param coordinate_columns: Names of the columns representing the
    coordinates of the entries.
    :return:
    """
    # This really only works if all the PET files are in one directory and
    # all the MR files are in one directory.
    for i, file_couple in enumerate(file_couples):
        mr_file, pet_file = file_couple
        mr_dfs, csv_names_mr = generate_MR_dataframe(file_name=mr_file,
                                                     output_directory=
                                                     output_directories[i],
                                                     new_output_name=
                                                     new_output_names[i],
                                                     save_to_csv=False,
                                                     ordinalize_coords=False,
                                                     roi_identifier=roi_identifier,
                                                     roi_common_denominator=roi_common_denominator,
                                                     file_name_suffix=file_name_suffix_MR,
                                                     roi_signifier=roi_signifier,
                                                     name_signifier="",
                                                     column_name=column_name_mr,
                                                     coordinate_columns=coordinate_columns)
        pet_dfs, csv_names_pet = generate_activity_dataframe(
            file_name=pet_file,
            output_directory=
            output_directories[
                i],
            new_output_name=
            new_output_names[
                i],
            suv_info_path=suv_info_path,
            save_to_csv=False,
            ordinalize_coords=False,
            roi_identifier=roi_identifier,
            roi_common_denominator=roi_common_denominator,
            frame_identifier=frame_identifier,
            file_name_suffix=file_name_suffix_PET,
            roi_signifier=roi_signifier,
            name_signifier="",
            idcc_column_name_suffix=idcc_column_name_suffix,
            suv_column_name_suffix=suv_column_name_suffix,
            coordinate_columns=coordinate_columns)

        if len(mr_dfs) != len(pet_dfs):
            file_names_mr: List[str] = [
                csv_name.split("/")[-1].split(".")[0] \
                    if len(
                    csv_name.split("/")[-1]) <= len(
                    csv_name.split(
                        "\\")[-1]) else "".join(
                    csv_name.split("\\")[0:-1]) for csv_name in
                csv_names_mr]
            file_names_pet: List[str] = [
                csv_name.split("/")[-1].split(".")[0] \
                    if len(
                    csv_name.split("/")[-1]) <= len(
                    csv_name.split(
                        "\\")[-1]) else "".join(
                    csv_name.split("\\")[0:-1]) for csv_name in
                csv_names_pet]

            new_mr_dfs: List[pd.DataFrame] = []
            new_csv_names_mr: List[str] = []
            new_pet_dfs: List[pd.DataFrame] = []
            new_csv_names_pet: List[str] = []

            missing_in_mr: List[str] = []
            missing_in_pet: List[str] = []

            # Note: The dummy dfs are not empty but have the shape of the
            # other filled df so as to avoid the second
            # warning about the df couples not having the same length,
            # which would be redundant.

            for unique_file_name in set(file_names_mr + file_names_pet):
                if unique_file_name not in file_names_mr:
                    old_index: int = file_names_pet.index(unique_file_name)

                    dummy_df_mr = pet_dfs[old_index][
                        coordinate_columns].copy()
                    dummy_df_mr[[F"placeholder{i}" for i in range(len(
                        mr_dfs[0].columns) - len(
                        coordinate_columns))]] = None
                    dummy_df_mr.columns = mr_dfs[0].columns
                    new_mr_dfs.append(dummy_df_mr)
                    new_csv_names_mr.append(csv_names_pet[old_index])

                    new_pet_dfs.append(pet_dfs[old_index])
                    new_csv_names_pet.append(csv_names_pet[old_index])

                    missing_in_mr.append(unique_file_name)
                elif unique_file_name not in file_names_pet:
                    old_index: int = file_names_mr.index(unique_file_name)

                    new_mr_dfs.append(mr_dfs[old_index])
                    new_csv_names_mr.append(csv_names_mr[old_index])

                    dummy_df_pet = mr_dfs[old_index][
                        coordinate_columns].copy()
                    dummy_df_pet[[F"placeholder{i}" for i in range(len(
                        pet_dfs[0].columns) - len(coordinate_columns))]] = \
                        None
                    dummy_df_pet.columns = pet_dfs[0].columns
                    new_pet_dfs.append(dummy_df_pet)
                    new_csv_names_pet.append(csv_names_mr[old_index])

                    missing_in_pet.append(unique_file_name)
                else:
                    old_index_mr: int = file_names_mr.index(
                        unique_file_name)
                    old_index_pet: int = file_names_pet.index(
                        unique_file_name)

                    new_mr_dfs.append(mr_dfs[old_index_mr])
                    new_csv_names_mr.append(csv_names_mr[old_index_mr])

                    new_pet_dfs.append(pet_dfs[old_index_pet])
                    new_csv_names_pet.append(csv_names_pet[old_index_pet])

            warnings.warn(F"There aren\'t equal amounts of MR ("
                          F"={len(mr_dfs)}) and "
                          F"PET (={len(pet_dfs)}) "
                          F"dataframes provided. Check"
                          F" whether the correct .csv files were paired.\nThe "
                          F"following dataframes are missing for MR:"
                          F" {missing_in_mr}\n"
                          F"The following dataframes are missing for PET: "
                          F"{missing_in_pet}.\nThe missing dataframes were "
                          F"replaced with dataframes containing only None as "
                          F"features.")

            # Has to come after the warning so as to not display the already
            # adjusted dataframe length.
            mr_dfs, csv_names_mr, pet_dfs, csv_names_pet = new_mr_dfs, \
                new_csv_names_mr, \
                new_pet_dfs, \
                new_csv_names_pet

        for j in range(len(mr_dfs)):
            if len(mr_dfs[j]) != len(pet_dfs[j]):
                warnings.warn(F"The PET/MR dataframes for files"
                              F" {csv_names_mr[j]} and {csv_names_pet[j]} do "
                              F"not have the same amount"
                              F" of rows. (Number of Rows MR={len(mr_dfs[j])},"
                              F" Number of Rows PET={len(pet_dfs[j])})")
            master_df = merge_dfs_outer([mr_dfs[j], pet_dfs[j]],
                                        coordinate_columns)
            if ordinalize_coords:
                master_df, _ = ordinalize_coordinates(master_df,
                                                      coordinate_columns)
            final_csv_name: str = csv_names_mr[j].split(".")[
                                      0] + name_signifier + "." + \
                                  csv_names_mr[j].split(
                                      ".")[-1]
            try:
                master_df.to_csv(final_csv_name, index=False)
            except FileNotFoundError:
                # A long-winded way of saying: chose the longest string
                # including all directories but not the file name.
                csv_name = csv_names_mr[j].replace("/", "\\")
                new_directory_path: str = "\\".join(
                    csv_name.split("\\")[0:-1])
                os.mkdir(new_directory_path)

                master_df.to_csv(final_csv_name, index=False)


def generate_activity_dataframe(file_name: str,
                                output_directory: str = "",
                                new_output_name: Optional[str] = None,
                                suv_info_path: Optional[str] = None,
                                save_to_csv: bool = False,
                                ordinalize_coords: bool = False,
                                roi_identifier: str = "ROI Name: ",
                                roi_common_denominator: str = "umor",
                                frame_identifier: str = "Frame:, ",
                                file_name_suffix: str = " - PET",
                                roi_signifier: str = " Tumor",
                                name_signifier: str = " - Activity",
                                idcc_column_name_suffix: str = "_IDCC",
                                suv_column_name_suffix: str = "_SUV",
                                coordinate_columns: list = ['x', 'y', 'z']):
    """
    Create .csv files containing the activity information across all frames
    for a given ROI.
    :param file_name: Name of the file from which to draw the activity
    information.
    :param output_directory: Name of directory where the information is
    supposed to be saved.
    :param new_output_name: Optional new base name for the ROIs of the source
    file.
    :param suv_info_path: Path to the information necessary to compute the
    SUVs, if desired.
    :param save_to_csv: Boolean to decide whether or not to save the
    dataframes to corresponding .csv files.
    :param ordinalize_coords: Boolean to decide whether or not to ordinalize
    the coordinates of the input data. In the default case,
    this is not being done.
    :param roi_identifier: String by which to recognize a line as the
    beginning of a new ROI.
    :param roi_common_denominator: String by which to split the ROI name in
    order to extract the unique part of the name.
    :param frame_identifier: String by which to recognize a line as the
    beginning of a new frame.
    :param file_name_suffix: String by which to split the source file name
    in order to extract the animal name.
    :param roi_signifier: Suffix to add to the animal name to signal the
    beginning of the ROI name.
    :param name_signifier: Suffix to add to the finished .csv file.
    :param idcc_column_name_suffix: Suffix added to the IDCC column names,
    in case SUV columns are supposed to be added.
    :param suv_column_name_suffix: Suffix added to the SUV column names,
    in case SUV columns are supposed to be added.
    :param coordinate_columns: Names of the columns representing the
    coordinates of the entries.
    :return: An array of all the master dfs. It also saves all read in .csv
    files as one .csv file per ROI,
    containing all frame information for that ROI.
    """
    first_roi_found = False
    frame_df_collection = []
    roi_name = ""
    frame_name = ""
    frame_rows = []

    # The output lists.
    result_dfs = []
    output_names = []

    file_str = open(file_name).read()
    lines = file_str.split("\n")
    # This whole dance with i has to be done so that the processing of the
    # last roi will truly only happen after every single line has been read in.
    for i in range(len(lines) + 1):
        # Is being called either if a new ROI is found or if the end of
        # the file is reached.
        try:
            line = lines[i]
        except IndexError:
            line = ""
        if (roi_identifier in line) | (i == len(lines)):
            if first_roi_found:

                frame_df_collection.append(create_rounded_coordinates_df(
                    frame_rows, coordinate_columns, frame_name))
                # Reset the list of rows for the current frame.
                frame_rows = []
                master_df: pd.DataFrame = merge_dfs_outer(
                    frame_df_collection, coordinate_columns)

                if new_output_name is not None:
                    new_file_name: str = new_output_name
                else:
                    new_file_name: str = \
                        file_name.split("/")[-1].split("\\")[-1].split(
                            file_name_suffix)[0].split(".")[0]

                # Check whether or not a common denominator was provided. If
                # not, just name the .csv file using the entire name of the
                # ROI.
                if not roi_common_denominator:
                    csv_name = output_directory + "/" + \
                               new_file_name + roi_name \
                               + name_signifier + ".csv"
                else:
                    # In case no specific tumor name is provided,
                    # give it label "1".
                    roi_suffix = "1" if (len(
                        roi_name.split(roi_common_denominator)[-1].strip())
                                         == 0) \
                        else roi_name.split(roi_common_denominator)[
                        -1].strip()
                    csv_name = output_directory + "/" + \
                               new_file_name + roi_signifier + roi_suffix \
                               + name_signifier + ".csv"

                # Ordinalize the coordinates of the vectors.
                if ordinalize_coords:
                    master_df, _ = ordinalize_coordinates(master_df,
                                                          coordinate_columns)

                if suv_info_path is not None:
                    master_df = add_normalized_columns(df=master_df,
                                                       activity_columns=
                                                       [column_name
                                                        for column_name in
                                                        master_df.columns
                                                        if column_name
                                                        not in coordinate_columns],
                                                       file_name=new_file_name,
                                                       normalization_info_path=suv_info_path,
                                                       idcc_column_name_suffix=idcc_column_name_suffix,
                                                       suv_column_name_suffix=suv_column_name_suffix)

                # Update the output lists.
                result_dfs.append(master_df)
                output_names.append(csv_name)

                # Save the dataframe to a .csv file, if wanted.
                if save_to_csv:
                    try:
                        master_df.to_csv(csv_name, index=False)
                    except FileNotFoundError:
                        os.mkdir(output_directory)
                        master_df.to_csv(csv_name, index=False)

            try:
                roi_name = line.split(roi_identifier)[-1]
            except IndexError:
                pass

            first_roi_found = True
            frame_df_collection = []

        # Is being called when the beginning of a new frame is found.
        elif frame_identifier in line:
            # Only convert the list of frame rows into a df if there are
            # more than zero rows, that is, it is not the very first
            # frame of an ROI.
            if len(frame_rows) > 0:
                frame_df_collection.append(create_rounded_coordinates_df(
                    frame_rows, coordinate_columns, frame_name))

            frame_name = "Frame_" + \
                         line.split(frame_identifier)[1].split(",")[0]
            frame_rows = []

        # Is being called when a content row
        # describing a voxel is recognized.
        # Assume that there will always be at least 4 columns to be read
        # in from the source .csv file.
        elif len(line.split(",")) > 3:
            frame_rows.append(line.split(","))

    return result_dfs, output_names


def generate_MR_dataframe(file_name: str,
                          output_directory: str = "",
                          new_output_name: Optional[str] = None,
                          save_to_csv: bool = False,
                          ordinalize_coords: bool = False,
                          roi_identifier: str = "ROI Name: ",
                          roi_common_denominator: str = "umor",
                          file_name_suffix: str = " - MR",
                          roi_signifier: str = " Tumor",
                          name_signifier: str = " - T2",
                          column_name: str = "T2",
                          coordinate_columns: list = ['x', 'y', 'z']):
    """
    Create .csv files containing the activity information across all frames
    for a given ROI.
    """
    first_roi_found = False
    roi_rows = []
    roi_name = ""

    result_dfs = []
    output_names = []

    file_str = open(file_name).read()
    lines = file_str.split("\n")
    # This whole dance with i has to be done so that the processing of the
    # last roi will truly only happen after every single line has been read in.
    for i in range(len(lines) + 1):
        # Is being called either if a new ROI is found or if the end of
        # the file is reached.
        try:
            line = lines[i]
        except IndexError:
            line = ""
        if (roi_identifier in line) | (i == len(lines)):
            if first_roi_found:

                master_df: pd.DataFrame = create_rounded_coordinates_df(
                    roi_rows, coordinate_columns, column_name)

                if new_output_name is not None:
                    new_file_name: str = new_output_name
                else:
                    new_file_name: str = \
                        file_name.split("/")[-1].split("\\")[-1].split(
                            file_name_suffix)[
                            0].split(".")[0]

                # Check whether or not a common denominator was provided. If
                # not, just name the .csv file using the entire name of the
                # ROI.
                if not roi_common_denominator:
                    csv_name = output_directory + "/" + \
                               new_file_name + roi_name \
                               + name_signifier + ".csv"
                else:
                    # In case no specific tumor name is provided,
                    # give it label "1".
                    roi_suffix = "1" if (len(
                        roi_name.split(roi_common_denominator)[
                            -1].strip()) == 0) \
                        else roi_name.split(roi_common_denominator)[
                        -1].strip()
                    csv_name = output_directory + "/" + \
                               new_file_name + roi_signifier + roi_suffix \
                               + name_signifier + ".csv"

                # Ordinalize the coordinates of the vectors.
                if ordinalize_coords:
                    master_df, _ = ordinalize_coordinates(master_df,
                                                          coordinate_columns)

                # Update the output lists.
                result_dfs.append(master_df)
                output_names.append(csv_name)

                # Save the dataframe to a .csv file, if wanted.
                if save_to_csv:
                    try:
                        master_df.to_csv(csv_name, index=False)
                    except FileNotFoundError:
                        os.mkdir(output_directory)
                        master_df.to_csv(csv_name, index=False)

            try:
                roi_name = line.split(roi_identifier)[-1]
            except IndexError:
                pass

            first_roi_found = True
            roi_rows = []

        # Is being called when a content row
        # describing a voxel is recognized.
        # Assume that there will always be at least 4 columns to be read
        # in from the source .csv file.
        elif len(line.split(",")) > 3:
            roi_rows.append(line.split(","))

    return result_dfs, output_names


def create_rounded_coordinates_df(arr: np.ndarray,
                                  coordinate_columns: list,
                                  feature_name=None,
                                  n_decimals: int = 5,
                                  dtype: str = 'float32'):
    """
    Creates a dataframe containing the
    coordinate information of the input as rounded values.
    :param arr: List or array of rows of the current frame.
    :param feature_name: Name of the current feature(i.e. frame number,
    in the case of PET .csv files), to be used as a column name.
    :param n_decimals: Number of decimals to round down to.
    :param coordinate_columns: Name of columns representing the coordinates.
    :return: A dataframe containing all information of the current frame.
    """
    n_dims = len(coordinate_columns)

    result_arr = np.copy(arr).astype(dtype)

    # COMMENTED OUT FOR NOW.
    # if result_arr.shape[1] <= len(coordinate_columns):
    #    feature_name = []

    if feature_name is None:
        feature_name = ['0']
    else:
        feature_name = [feature_name]

    # In case an empty array was provided, simply return an empty dataframe.
    # BUT pass over the coordinate column and feature column names.
    if result_arr.shape[0] == 0:
        return pd.DataFrame(columns=coordinate_columns + feature_name)

    # A check to catch empty arrays, that is, ROIs without a single voxel.
    if result_arr.shape[0] > 0:
        result_arr[:, :n_dims] = np.round(result_arr[:, :n_dims],
                                          n_decimals)
        result_df = pd.DataFrame(result_arr,
                                 columns=coordinate_columns + feature_name)
    else:
        result_df = pd.DataFrame(columns=coordinate_columns + feature_name)

    return result_df


def merge_dfs_outer(dfs: list, coordinate_columns: list):
    """
    Create a dataframe containing all information
    of the individual frames of a ROI.
    :param dfs: A list containing all the dataframes to be merged.
    :param coordinate_columns: The columns on which the dataframes are being
    merged.
    :return: Dataframe containing all merged frames, accounting for missing
    frame information by filling it with NaN.
    """
    loop_dfs: List[pd.DataFrame] = []
    # Kick out all the rows for which the coordinates are not completely
    # provided.
    for df in dfs:
        loop_dfs.append(df.loc[df[coordinate_columns].isnull().sum(
            axis=1) == 0])

    return reduce(lambda first_df, rest_df: pd.merge(first_df, rest_df,
                                                     on=coordinate_columns,
                                                     how='outer'), loop_dfs)


def ordinalize_coordinates(input_df: pd.DataFrame,
                           coordinate_columns: list,
                           epsilon: float = 0.01,
                           n_decimals: int = 4):
    """
    Ordinalize the coordinates of a dataframe so that they all start at zero
    and take only integer values.
    :param input_df: Input dataframe.
    :param coordinate_columns: The names of the columns of the dataframe
    representing the coordinates that are to be ordinalized.
    :param epsilon: Error threshold for assigning float coordinates to their
    corresponding integer coordinates.
    :param n_decimals: Number of decimals the reconstructed values and the
    minimal entry distance are to be rounded down to.
    :return: The input array, with the original coordinates replaced by the
    ordinal ones.
    """
    # Get the minimum distance between two voxels in all dimensions
    # that isn't zero.
    df_arr = input_df[coordinate_columns].to_numpy()

    # A check to see whether a dataframe only contains None-entries.
    if not np.any(df_arr):
        return input_df

    result_arr = np.zeros(df_arr.shape)
    result_arr[:] = -1
    min_distances = []
    for dimension_name in coordinate_columns:
        values_dim = np.sort(np.unique(input_df[dimension_name].to_numpy()))
        # In case an ROI exists only in two or fewer dimensions.
        if len(values_dim) < 2:
            min_distances.append(0)
            continue
        min_distance_dim = np.min(
            np.abs(values_dim - np.roll(values_dim, 1)))
        min_distances.append(min_distance_dim)

    min_distances = np.round(min_distances, n_decimals)

    # In case a dimension has length 1, this check is irrelevant.
    if (epsilon >= np.min(min_distances) / 2) & (np.min(min_distances) > 0):
        raise ValueError(F"Epsilon (={epsilon}) was chosen too big (bigger "
                         F"than half of the min voxel distance={np.min(min_distances)}),"
                         F" resulting in possible "
                         F"overlap of the ordinalization assignment.")

    min_values = np.min(df_arr, axis=0)
    max_values = np.max(df_arr, axis=0)

    for dim_idx, dimension_name in enumerate(coordinate_columns):
        # In case a dimension only has length 1, all voxels shall have
        # coordinate 0 in that dimension.
        if min_distances[dim_idx] <= 0:
            result_arr[:, dim_idx] = 0
            continue
        # Note: The 2 serves as a buffer to truly include all voxels of a
        # Siemens .csv file.
        for new_coordinate in range(int(
                (max_values - min_values)[dim_idx] / min_distances[
                    dim_idx]) + 2):
            loop_column = result_arr[:, dim_idx]
            roundabout_value = np.round(min_values[
                                            dim_idx] + new_coordinate *
                                        min_distances[dim_idx], n_decimals)
            loop_column[np.where((df_arr[:, dim_idx] < (
                    roundabout_value + epsilon))
                                 & (df_arr[:, dim_idx] > (
                    roundabout_value - epsilon)))] \
                = new_coordinate

    print(min_values, max_values)
    print(min_distances)
    print(pd.DataFrame(df_arr))

    if np.sum(result_arr < 0):
        raise ValueError(
            F"Ordinalization failed. The following values could "
            F"not be ordinalized: "
            F"{df_arr[np.where(result_arr < 0)]}")

    output_df = pd.DataFrame(result_arr, columns=coordinate_columns).astype(
        {
            coordinate_column: 'int' for coordinate_column in
            coordinate_columns})
    output_df[input_df.columns.drop(coordinate_columns)] = input_df[
        input_df.columns.drop(coordinate_columns)]

    return output_df, min_distances


def add_normalized_columns(df: pd.DataFrame,
                           activity_columns: List[str],
                           file_name: str,
                           normalization_info_path: str,
                           idcc_column_name_suffix: str,
                           suv_column_name_suffix: str) -> pd.DataFrame:
    """
    Converts the activity information of a dataframe into normalized data (
    IDCC and SUV) and adds this information to a copy of the input dataframe,
    given a .csv file containing the necessary information for the conversion.
    """
    f18_half_life: float = 6586.2

    idcc_columns: List[str] = [column_name + idcc_column_name_suffix for
                               column_name in activity_columns]
    suv_columns: List[str] = [column_name + suv_column_name_suffix for
                              column_name in activity_columns]

    normalization_info_df: pd.DataFrame = pd.read_csv(
        normalization_info_path)
    normalized_info_row: pd.DataFrame = \
        normalization_info_df[normalization_info_df['scan_name'] ==
                              file_name].reset_index(
            drop=True).loc[0]

    result_df: pd.DataFrame = df.copy()

    # Try if there's a injected dose column directly available. If not,
    # calculate the dose based off of other values.
    try:
        injected_dose_mega_becquerel = normalized_info_row['dose_injected']
        body_weight: float = normalized_info_row['body_weight']

        idcc_df: pd.DataFrame = result_df[activity_columns] / (
                injected_dose_mega_becquerel * 1000000) * 100

        if np.isnan(body_weight):
            suv_df: pd.DataFrame = pd.DataFrame(columns=suv_columns)
        else:
            suv_df: pd.DataFrame = result_df[activity_columns] / (
                    injected_dose_mega_becquerel * 1000000) * body_weight
    except KeyError:
        body_weight: float = normalized_info_row['body_weight']
        dose_pre: float = normalized_info_row['dose_pre']
        dose_pst: float = normalized_info_row['dose_pst']
        measure_time_pre: float = normalized_info_row['measure_time_pre']
        measure_time_pst: float = normalized_info_row['measure_time_pst']
        acquisition_time: float = normalized_info_row['acquisition_time']
        # Is here for completion's sake.
        injection_time: float = normalized_info_row['injection_time']

        # So as it turns out this check isn't necessary, because NaN thankfully
        # simply propagates through the entire calculation, but it is still left
        # here for posterity.
        if not ((np.isnan(dose_pre)) | (np.isnan(
                dose_pst)) | (
                        np.isnan(measure_time_pre)) | (
                        np.isnan(measure_time_pst)) | (
                        np.isnan(acquisition_time))):
            injected_dose_micro_curie: float = dose_pre * np.exp(
                -np.log(2) * (
                        acquisition_time - measure_time_pre) / f18_half_life) - dose_pst * np \
                                                   .exp(
                np.log(2) * (
                        measure_time_pst - acquisition_time) / f18_half_life)

            injected_dose_mega_becquerel = injected_dose_micro_curie * 0.037

            idcc_df: pd.DataFrame = result_df[activity_columns] / (
                    injected_dose_mega_becquerel * 1000000) * 100

            if np.isnan(body_weight):
                suv_df: pd.DataFrame = pd.DataFrame(columns=suv_columns)
            else:
                suv_df: pd.DataFrame = result_df[activity_columns] / (
                        injected_dose_mega_becquerel * 1000000) * body_weight
        else:
            idcc_df: pd.DataFrame = pd.DataFrame(columns=suv_columns)
            suv_df: pd.DataFrame = pd.DataFrame(columns=suv_columns)

    result_df[idcc_columns] = idcc_df
    result_df[suv_columns] = suv_df

    return result_df


def merge_and_sort_activity_values(paths: List[str],
                                   activity_columns: List[str],
                                   new_path: str):
    """
    Merge the activity data of several .csv files and sort each column in
    ascending order, independently from the other columns, for the sorted
    activity value tables.
    """
    result_df: pd.DataFrame = pd.DataFrame(columns=activity_columns)
    df_list: List[pd.DataFrame] = [pd.read_csv(path)[activity_columns] for
                                   path
                                   in paths]

    # If the df list is empty, avoid the pandas read pandas concat function call
    # which would cause an error.
    if df_list:
        master_df: pd.DataFrame = pd.concat(df_list).reset_index(drop=True)
        for column_name in master_df.columns:
            result_df[column_name] = master_df[column_name].sort_values(
                na_position='first').reset_index(drop=True)

    result_df.to_csv(new_path, index=False)


def interpolate_ordinalized_coordinates(path_input: str,
                                        path_output: str,
                                        coordinate_columns=["x", "y", "z"]):
    """
    Interpolate ordinalized coordinates for a dataframe which does not have
    the same voxel size in all of its dimensions. That is, for the most part,
    its z-size is bigger than its x- and y-size.
    """
    input_df: pd.DataFrame = pd.read_csv(path_input)
    min_values_per_coordinate: np.ndarray = np.min(
        input_df[coordinate_columns],
        axis=0).astype(int) - 1
    max_values_per_coordinate: np.ndarray = np.max(
        input_df[coordinate_columns],
        axis=0).astype(int) + 1

    ranges: list = [np.arange(min_values_per_coordinate[i],
                              max_values_per_coordinate[i] + 1)
                    for i
                    in range(len(coordinate_columns))]
    # All the coordinates contained in the interpolated dataframe.
    ordinalized_coordinates: np.ndarray = np.sort(np.array(np.meshgrid(
        *ranges)).T.reshape(-1, len(coordinate_columns)))

    interpolated_features = griddata(
        points=input_df[coordinate_columns],
        values=input_df[input_df.columns.drop(coordinate_columns)],
        xi=ordinalized_coordinates,
        method="linear")

    output_df: pd.DataFrame = pd.DataFrame(
        np.append(ordinalized_coordinates,
                  interpolated_features,
                  axis=1),
        columns=input_df.columns).astype({
        coordinate_column: 'int' for coordinate_column in
        coordinate_columns})

    output_df.to_csv(path_output, index=False)


def ordinalize_per_slice(path_input: str,
                         path_output: str,
                         xy_columns: list = ["x", "y"],
                         z_column: str = "z",
                         epsilon: float = 0.001,
                         n_decimals: int = 4):
    df: pd.DataFrame = pd.read_csv(path_input)

    # A check to see whether a dataframe only contains None-entries.
    if not np.any(df):
        return df

    result_arr = np.zeros(df.shape)
    result_arr[:] = -1
    min_distances_xy = []
    for dimension_name in xy_columns:
        sorted_values = np.sort(np.unique(df[dimension_name]))
        sorted_values = sorted_values - np.roll(sorted_values, 1)
        print(
            "_______________________________________________________________")
        print(sorted_values)
        values_dim = np.sort(np.unique(df[dimension_name].to_numpy()))
        # In case an ROI exists only in two or less dimensions.
        if len(values_dim) < 2:
            continue
        min_distance_dim = np.min(
            np.abs(values_dim - np.roll(values_dim, 1)))
        min_distances_xy.append(min_distance_dim)
    print(
        "------------------------------------------------------------------")
    sorted_z_values = np.sort(np.unique(df[z_column]))
    sorted_z_values = sorted_z_values - np.roll(sorted_z_values, 1)
    print(sorted_z_values)

    if np.allclose(min_distances_xy, np.roll(min_distances_xy, 1)):
        min_distance_xy = min_distances_xy[0]
    else:
        raise ValueError("The x-y-coordinates do not share the same step "
                         "sizes.")

    if epsilon >= min_distance_xy / 2:
        raise ValueError(F"Epsilon (={epsilon}) was chosen too big (bigger "
                         F"than half of the min voxel distance="
                         F"{min_distance_xy}),"
                         F" resulting in possible "
                         F"overlap of the ordinalization assignment.")

    min_values_xy = np.min(df[xy_columns], axis=0)
    max_values_xy = np.max(df[xy_columns], axis=0)

    print(min_values_xy[1], max_values_xy[1], min_distances_xy)

    z_values = np.sort(np.unique(df[z_column]))
    print(z_values)

    for z_index, z_value in enumerate(z_values):
        for n_dimension, dimension_name in enumerate(xy_columns):
            print("now its ", dimension_name)
            for new_coordinate in range(
                    int((max_values_xy - min_values_xy)[n_dimension] /
                        min_distance_xy) + 3):
                roundabout_value = np.round(min_values_xy[
                                                n_dimension] + new_coordinate *
                                            min_distance_xy, n_decimals)
                df.loc[(df[dimension_name] < (
                        roundabout_value + epsilon)) & (
                               df[dimension_name] > (
                               roundabout_value - epsilon))
                       & (df[z_column] == z_value), dimension_name] = \
                    new_coordinate
        df.loc[df[z_column] == z_value, z_column] = z_index
        print("z_value done!")

    df = df.astype({
        coordinate_column: 'int' for coordinate_column in
        xy_columns + [z_column]})
    df.to_csv(path_output, index=False)
