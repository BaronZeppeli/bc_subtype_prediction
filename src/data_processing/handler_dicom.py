import os

import warnings
from warnings import warn

from typing import List, Tuple
from collections.abc import Iterable

import pydicom
import pydicom.multival
import pydicom.uid
import pydicom.valuerep
import pydicom.sequence
from nibabel.affines import apply_affine
from pydicom.errors import InvalidDicomError

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.interpolate import griddata, RegularGridInterpolator

import handler_nifti
import handler_siemens

DEFAULT_NAME = "PATIENT"
DEFAULT_MODALITY = "MODALITY"

NL = "\n"
COMMA = ", "
EPS = 1e-10
FILLER_ORIENTATION = np.repeat(np.nan, 6)


def is_empty_string(string):
    """
    Helper function to catch weird behavior when an attribute
    can either be a string or None.
    """
    if string is None:
        return True
    else:
        return len(string) == 0


def custom_formatwarning(msg, *args, **kwargs):
    """
    Custom warning message to not show the source code when disiplaying a
    warning.
    """
    # ignore everything except the message
    return str(msg) + '\n'


# To suppress the link to the line of code where the warning was raised.
# warnings.formatwarning = custom_formatwarning


def get_dicom_attribute(slice: pydicom.dataset.FileDataset,
                        address: list,
                        default,
                        dtype=None):
    """
    Helper function to savely retrieve attributes from DICOM slices.
    """
    attribute = slice.get(address)
    if attribute is not None:
        attribute = attribute.value
        # Check again for the attribute, in case the content of the attribute
        # was None.
        if (dtype is not None) & (attribute is not None):
            attribute = dtype(attribute)
    else:
        attribute = default

    return attribute


def time_decoder(dicom_time,
                 format: str,
                 unit="seconds"):
    """
    Helper function to get the timestamp from a time as encoded in a DICOM file.
    """
    # In case None is provided, immediately return a default value.
    if dicom_time is None:
        return None

    # Determine the conversion factor, depending on the selected output unit.
    scaler: int = 1 if unit == "seconds" \
        else 60 if unit == "minutes" \
        else 3600
    # The output time, decoded as a float
    decoded_time: float = 0

    dicom_time = str(dicom_time)
    # The string representing the (possibly year, month, day) hours, minutes
    # and  seconds.
    datetime_str: str = dicom_time.split(".")[0]
    decimal_str: str = dicom_time.split(".")[-1] if ("." in dicom_time) else ""

    # Handling for when the date is supposed to be considered.
    if ((format == "datetime") & (13 <= len(datetime_str) <= 14)) \
            | ((format == "date") & (len(datetime_str) == 8)):
        years = int(datetime_str[:4])
        months = int(datetime_str[4:6])
        days = int(datetime_str[6:8])

        decoded_time += (years * (3600 * 24 * 365) + months * (
                3600 * 24 * 30) + days * (3600 * 24)) / scaler

        hms_str = datetime_str[8:]
        datetime = True
    else:
        hms_str = datetime_str[-6:]
        datetime = False

    # Always try to convert the time.
    # This code accounts for the hour being stored as just one digit.
    if 5 <= len(hms_str) <= 6:
        # Either 0 or 1.
        h_offset: int = 6 - len(hms_str)

        hours: int = int(hms_str[0:2 - h_offset])
        minutes: int = int(hms_str[2 - h_offset:4 - h_offset])
        seconds: float = float(hms_str[4 - h_offset:] + "." + decimal_str)

        decoded_time += (hours * 3600 + minutes * 60 + seconds * 1) / scaler

    elif not datetime:
        # In case everything else fails, return a default value.
        warnings.warn(F"Failed to convert the string {dicom_time} to a "
                      F"date or time.")
        decoded_time = 0.0

    return decoded_time


def get_acquisition_time(slice: pydicom.dataset.FileDataset):
    """
    The helper function to get the acquisition time in any context.
    """
    # Attribute named "Acquisition Time"
    acquisition_time = get_dicom_attribute(slice=slice,
                                           address=[0x0008,
                                                    0x0032],
                                           default=None,
                                           dtype=str)
    # In case no acquisition time is provided, try acquisition datetime.
    if acquisition_time is None:
        # Attribute named "Acquisition DateTime"
        acquisition_time = get_dicom_attribute(slice=slice,
                                               address=[0x0008,
                                                        0x002A],
                                               default=None,
                                               dtype=str)

    # Apply the time decoding to the encoded string.
    acquisition_time = time_decoder(dicom_time=acquisition_time, format="time")

    return acquisition_time


def get_injected_dose(slice: pydicom.dataset.FileDataset):
    """
    The helper function to get the injected dose in any context.
    """
    # Attribute named "Radiopharmaceutical Information Sequence"
    radpharm_sequence = get_dicom_attribute(slice=slice, address=[0x0054,
                                                                  0x0016],
                                            dtype=list,
                                            default=None)

    if radpharm_sequence is not None:
        # Attribute named "Radionuclide Total Dose"
        injected_dose = get_dicom_attribute(slice=radpharm_sequence[0],
                                            address=[0x0018,
                                                     0x1074],
                                            dtype=float,
                                            default=None)
        return injected_dose
    else:
        return None


def get_decay_factor(slice: pydicom.dataset.FileDataset):
    """
    The helper function to get the injected dose in any context. NOTE: This
    does not refer to the DICOM attribute "Decay Factor" (0054, 1321),
    but the factor needed to correct for decay in terms of SUV calculation.
    """
    # Attribute named "Decay Correction"
    decay_correction = get_dicom_attribute(slice=slice, address=[0x0054,
                                                                 0x1102],
                                           default="", dtype=str)
    # In case the the series has already been corrected to the
    # radiopharmacon administration time instead of the scan start time,
    # there is no additional correction to be done.
    if decay_correction.lower() == "admin":
        return 1
    # TODO: Do the decay correction to the first timepoint by hand if
    #  none was provided.
    elif decay_correction.lower() == "none":
        warnings.warn("The selected series was not decay corrected.")

    # Attribute named "Radiopharmaceutical Information Sequence"
    radpharm_sequence = get_dicom_attribute(slice=slice, address=[0x0054,
                                                                  0x0016],
                                            default=None, dtype=list)
    if radpharm_sequence is None:
        return

    # Attribute named "Series Time"
    series_time = get_dicom_attribute(slice=slice,
                                      address=[0x0008, 0x0031],
                                      default=None,
                                      dtype=str)
    # Apply the time decoding to the encoded string. NOTE: This means that if
    # a float is passed for series time, it is always interpreted as a
    # decoded time representation.
    if isinstance(series_time, str):
        series_time = time_decoder(dicom_time=series_time,
                                   format="time")

    # Attribute named "Radionuclide Half Life"
    half_life = get_dicom_attribute(slice=radpharm_sequence[0],
                                    address=[0x0018, 0x1075],
                                    default=None,
                                    dtype=float)

    # In case no series time or half life is provided, return.
    if (series_time is None) | (half_life is None):
        return None

    # Attribute named "Series Date"
    series_date = get_dicom_attribute(slice=slice,
                                      address=[0x0008, 0x0021],
                                      default=None,
                                      dtype=str)
    # Attribute named "Radiopharmaceutical Start DateTime"
    radpharm_startdatetime = get_dicom_attribute(slice=radpharm_sequence[0],
                                                 address=[0x0018, 0x1078],
                                                 default=None,
                                                 dtype=str)
    # Attribute named "Radiopharmaceutical Start Time"
    radpharm_starttime = get_dicom_attribute(slice=radpharm_sequence[0],
                                             address=[0x0018, 0x1072],
                                             default=None,
                                             dtype=str)
    # In case there is datetime information available, use it to account
    # for the injection and the scan being done on separate days.
    if not (is_empty_string(series_date)
            | is_empty_string(radpharm_startdatetime)):
        series_date = time_decoder(dicom_time=series_date,
                                   format="date")
        radpharm_startdatetime = time_decoder(
            dicom_time=radpharm_startdatetime,
            format="datetime")
        delta_scan_start = series_time + series_date - \
                           radpharm_startdatetime
    elif not is_empty_string(radpharm_starttime):
        radpharm_starttime = time_decoder(dicom_time=radpharm_starttime,
                                          format="time")
        delta_scan_start = series_time - radpharm_starttime
    else:
        delta_scan_start = -1

    # This case is true either if the necessary attributes were not found or
    # if the scan time comes before the injection start time.
    if delta_scan_start < 0:
        return None
    else:
        return np.power(2, -delta_scan_start / half_life)


def load_dicom_file(file_path: str):
    """
    Loads in a single DICOM file and returns the values
    necessary to create a voxel array instance.
    :param file_path: The path to the directory containing the DICOM
    files that are to be read in.
    :return: The array, name, features voxel size,
    and metadata for the voxel array object.
    """
    slice: pydicom.dataset.FileDataset = pydicom.dcmread(file_path)

    # DO NOT FORGET: RESCALE THE VALUES USING THE PROVIDED
    # INFORMATION!!!
    # Attribute named "Rescale Intercept"
    b = get_dicom_attribute(slice=slice, address=[0x0028, 0x1052],
                            default=0.0, dtype=float)
    # Attribute named "Rescale Slope"
    m = get_dicom_attribute(slice=slice, address=[0x0028, 0x1053],
                            default=1.0, dtype=float)

    array: np.ndarray = (m * np.expand_dims([slice.pixel_array], axis=3)) + b

    # Attribute named "Patient's Name"
    name: str = get_dicom_attribute(slice=slice, address=[0x0010, 0x0010],
                                    default=DEFAULT_NAME, dtype=str)

    channels: List[str] = [get_dicom_attribute(slice=slice, address=[0x0008,
                                                                     0x0060],
                                               default=DEFAULT_MODALITY,
                                               dtype=str)]
    # Attribute named "Pixel Spacing"
    pixel_spacing: List[float] = get_dicom_attribute(slice=slice,
                                                     address=[0x0028, 0x0030],
                                                     default=[1.0, 1.0],
                                                     dtype=list)
    # Attribute named "Slice Thickness" (0018,0050)
    slice_thickness = get_dicom_attribute(slice=slice, address=[0x0018, 0x0050],
                                          default=1.0, dtype=float)
    # The pixel spacing has the order Row Spacing - Column Spacing.
    voxel_size: List[float] = pixel_spacing + [slice_thickness]
    metadata: List[str] = [str(slice)]

    acquisition_time = [get_acquisition_time(slice=slice)]
    # Attribute named "Patient's Weight"
    weight = [get_dicom_attribute(slice=slice, address=[0x0010, 0x1030],
                                  default=1.0, dtype=float)]

    injected_dose = [get_injected_dose(slice=slice)]

    decay_factor = [get_decay_factor(slice=slice)]

    affine = handler_nifti.get_affine_from_DICOM(slice=slice)

    return array, name, channels, voxel_size, acquisition_time, weight, \
        injected_dose, decay_factor, metadata, affine


def load_and_order_dicom_slices(directory_path: str,
                                method: str = "linear",
                                fill_value=None):
    """
    Handles the loading and sorting of DICOM slices contained in one directory.
    """
    # Load the DICOM files.
    file_paths = np.array([
        directory_path + "/" + file_name for file_name in
        os.listdir(directory_path) if
        os.path.isfile(directory_path + "/" + file_name)])
    # Workaround to ensure that the individual attributes are not parsed as
    # single array elements.
    slices = np.zeros((len(file_paths, )), dtype=object)
    for i, file_path in enumerate(file_paths):
        slices[i] = pydicom.dcmread(file_path, force=True)

    # Check to ensure all DICOM slices have the same orientation.
    # Attribute named "Image Orientation (Patient)"
    orientation_vectors = np.array([get_dicom_attribute(slice=slice,
                                                        address=[0x0020,
                                                                 0x0037],
                                                        default=FILLER_ORIENTATION)
                                    for slice in slices]).astype(float)
    # Set the orientation to be used for alignment equal to the most frequent
    # ImageOrientationPatient in the entire directory.
    valid_orientations, n_orientations = np.unique(orientation_vectors[np.all(
        ~np.isnan(orientation_vectors), axis=1)], axis=0, return_counts=True)
    if len(valid_orientations) > 0:
        reference_orientation = valid_orientations[np.argmax(n_orientations)]
    else:
        warnings.warn("No single valid ImageOrientationPatient attribute was "
                      "found in the specified directory. Failed to merge the "
                      "DICOM slices.")
        reference_orientation = FILLER_ORIENTATION

    # All the slices not conforming to the most frequent image orientation
    # within the specified directory of DICOM files.
    outlier_idxs = np.where(np.any(orientation_vectors != reference_orientation,
                                   axis=1))[0]

    # Single out the outliers.
    outlier_slices = []
    load_success_outliers = []
    load_fail_outliers = []

    for outlier_idx in outlier_idxs:
        try:
            # In case the outlier slice has no pixel array, discard it
            # completely, as it probably contains only a metadata dataset.
            _ = slices[outlier_idx].pixel_array
            load_success_outliers.append(file_paths[outlier_idx])
            outlier_slices.append(slices[outlier_idx])
        except AttributeError:
            load_fail_outliers.append(file_paths[outlier_idx])

    # Throw out all outlier slices from the main slice array.
    slices = slices[[i for i in range(len(slices)) if i not in outlier_idxs]]
    if len(outlier_idxs) > 0:
        if len(load_success_outliers) > 0:
            warn_text = "The image orientation is not the same for each of " \
                        "the DICOM files that are to be fused.\nThe following" \
                        " file(s) do not have the same image orientation as" \
                        " the rest of the DICOM files in the directory and " \
                        "were discarded from the volume:"
            warn(warn_text + "\n" + ",\n".join(load_success_outliers) + "\n")

        if len(load_fail_outliers) > 0:
            warn("The following file(s) could not be read as a DICOM file:\n"
                 + ",\n".join(load_fail_outliers) + "\n")

    # The direction along the different slices.
    z_direction = np.cross(reference_orientation[:3], reference_orientation[
        3:])
    z_direction /= np.linalg.norm(z_direction)
    # Hotfix to ensure that the canonical direction vectors always are
    # directed towards positive space.
    if z_direction[np.argmax(np.abs(z_direction))] < 0:
        z_direction = -z_direction

    # The position of every single slice, in the same order as the slices
    # themselves.
    all_positions = np.array([get_dicom_attribute(slice=slice,
                                                  address=[0x0020,
                                                           0x0032],
                                                  default=np.repeat(
                                                      np.nan, 3))
                              for slice in slices]).astype(float)
    no_position_idxs = np.any(np.isnan(all_positions), axis=1)
    if np.sum(no_position_idxs) > 0:
        warn(F"The following DICOM files have no position information and "
             F"will be discarded from the volume:\n"
             F"{NL.join(file_paths[no_position_idxs])}")

    # Discard the slices without any position information.
    outlier_slices += slices[no_position_idxs].tolist()
    slices = slices[~no_position_idxs]
    all_positions = all_positions[~no_position_idxs]

    # Order the slices and positions by their location along the z-direction.
    position_ranking = [np.dot(position, z_direction) for position in
                        all_positions]
    slices = slices[np.argsort(position_ranking)]
    # Don't forget to sort the position vectors as well, as they're the basis
    # for indexing the slice array!
    all_positions = all_positions[np.argsort(position_ranking)]

    # Get all positions, but maintain all NaN-positions as well.
    unique_positions, n_modalities_per_position = np.unique(
        all_positions, axis=0, return_counts=True)
    unique_positions = unique_positions[
        np.all(~np.isnan(unique_positions), axis=1)]
    n_positions = len(unique_positions)
    n_modalities = np.max(n_modalities_per_position)

    # Select the start position as the first position in the array of ordered
    # positions.
    start_position = all_positions[np.all(~np.isnan(all_positions), axis=1)][0]
    # Order the positions again, as np.unique does not enforce ordering.
    unique_position_order = np.argsort(
        [np.linalg.norm(pos_vec - start_position) for
         pos_vec in
         unique_positions])
    unique_positions = unique_positions[unique_position_order]

    ############################################################################
    # CREATE A 4D ARRAY FROM THE LOADED DICOM SLICES.

    pixel_arr_dims = np.array([slice.pixel_array.shape for slice in slices])
    if np.any(pixel_arr_dims != pixel_arr_dims[0]):
        warn("Not all slices have the same shape for their pixel array. The "
             "different shapes of the pixel arrays are:\n" +
             ", ".join(map(str, np.unique(pixel_arr_dims, axis=0)))
             + F"\nPadding the missing pixels with {fill_value}.\n")

    max_height, max_width = np.max(pixel_arr_dims, axis=0)
    volume_arr = np.zeros((n_positions, max_height, max_width, n_modalities))
    info_missing_idxs = []
    modalities_missing_idxs = []
    for position_idx, unique_position_vector in enumerate(
            unique_positions):
        pos_idxs = np.all(all_positions == unique_position_vector,
                          axis=1)
        # The third output can be named modalities_missing, as we know that
        # there will never be too many modalities in this case.
        ordered_slices, position_arr, info_missing, modalities_missing = \
            combine_slices(
                slices=slices[pos_idxs],
                shape=volume_arr.shape[1:],
                fill_value=fill_value)
        volume_arr[position_idx] = position_arr
        # Order the slices according to the order of the modalities
        # identified with combine_slices().
        slices[pos_idxs] = ordered_slices

        # Make sure to store the slices for which errors were encountered
        # under the list of outliers.
        if info_missing | modalities_missing:
            outlier_slices += slices[pos_idxs].tolist()

        # Make sure to keep track for which slice positions problems were
        # encountered.
        info_missing_idxs += [position_idx] if info_missing else []
        modalities_missing_idxs += [position_idx] if modalities_missing else []

    if len(info_missing_idxs) > 0:
        warnings.warn(F"For the following DICOM slices there was not enough "
                      F"information in their header available to properly "
                      F"order their modalities:\n"
                      F"{NL.join(file_paths[info_missing_idxs])}. They will "
                      F"be re-interpolated using their neighbors.")
    if len(modalities_missing_idxs) > 0:
        warnings.warn(F"For the following DICOM slices did not have enough "
                      F"available modalities (expected {n_modalities}):\n"
                      F"{NL.join(file_paths[modalities_missing_idxs])}. They "
                      F"will be re-interpolated using their neighbors.")

    # Discard the slices without any ordering information or with too few
    # modalities.
    merge_fail_idxs = list(
        set(info_missing_idxs).union(modalities_missing_idxs))
    merge_success_idxs = list(set(np.arange(n_positions)).difference(
        merge_fail_idxs))
    merge_fail_idxs.sort()
    merge_success_idxs.sort()

    # Remove the slices with modality problems from the list of slices and
    # positions, to enforce their re-interpolation!!!
    slices_new = []
    for position_idx in merge_success_idxs:
        slices_new += list(slices[position_idx * n_modalities
                             :(position_idx + 1) * n_modalities])
    slices = np.array(slices_new)
    unique_positions = unique_positions[merge_success_idxs]

    ############################################################################
    # DETECT MISSING OR MISPLACED SLICES.

    # Store the distance of each position vector to the starting vector.
    norms = np.linalg.norm(unique_positions - start_position, axis=1)
    norms -= np.min(norms)

    # The distance between neighboring slices.
    slice_dists = np.round(np.diff(norms), 2)
    if not np.allclose(slice_dists, slice_dists[0]):
        unique_dists, counts = np.unique(slice_dists, return_counts=True)
        # Set the standard distance by which the slice distance is determined
        # equal to the most frequent distance between slices.
        standard_dist = unique_dists[np.argmax(counts)]

        # Figure out what exactly went wrong.
        placement_check = norms / standard_dist
        placement_check = np.round(placement_check - np.fix(placement_check),
                                   5) % 1.0
        placement_check = placement_check < EPS
        # CASE 1: There are simply some slides missing.
        if np.all(placement_check):
            n_positions_true = int(np.linalg.norm(unique_positions[-1] -
                                                  unique_positions[0])
                                   / standard_dist) + 1
            new_positions = start_position + np.arange(
                n_positions_true)[..., None] * z_direction * standard_dist
            norms_true = np.arange(n_positions_true) * standard_dist
            missing_idxs = [i for i in range(n_positions_true) if not np.any(
                np.isclose(norms, norms_true[i], atol=EPS))]
            # Do not show the slice positions with failed modality
            # ordering/missing modalities under the list of missing
            # positions, because they were removed artificially!
            missing_idxs = list(set(missing_idxs).difference(merge_fail_idxs))
            missing_idxs.sort()
            warnings.warn(
                F"The slices for position(s) {missing_idxs} in the "
                F"DICOM volume are missing.\n"
                F"The missing information will be interpolated.")
        # CASE 2: A real subset of all slices is misplaced along the z-axis.
        else:
            # Determine the islands of slices that have the standard distance and
            # select the largest island as the new origin for the interpolated slice
            # positions.
            islands = np.concatenate([0, np.isclose(slice_dists,
                                                    standard_dist, atol=EPS),
                                      0],
                                     axis=None).astype(bool)
            water_idxs = np.where(~islands)[0]

            warnings.warn(F"The slices of indices "
                          F"[{COMMA.join(water_idxs.astype(str))}] "
                          F"are misplaced "
                          F"along the z-axis of the DICOM volume and will be "
                          F"re-interpolated.")
            island_sizes = np.diff(water_idxs) - 1
            biggest_island = np.argmax(island_sizes)
            island_start_idx = water_idxs[biggest_island]
            island_end_idx = island_start_idx + np.max(island_sizes)
            n_slices_before_island_start = np.ceil(np.linalg.norm(
                unique_positions[
                    island_start_idx] - start_position) / standard_dist)
            n_slices_after_island_start = np.ceil(np.linalg.norm(
                unique_positions[
                    island_start_idx] - unique_positions[
                    -1]) / standard_dist)
            n_new_slices = n_slices_before_island_start + \
                           n_slices_after_island_start + 1
            interpolation_start_position = unique_positions[island_start_idx] - \
                                           z_direction * standard_dist * n_slices_before_island_start
            new_positions = interpolation_start_position + np.arange(
                n_new_slices)[..., None] * z_direction * standard_dist

            # Don't forget to remove all the slices not within the biggest
            # island from the list of valid slices!
            outlier_slices += slices[:island_start_idx].tolist() + slices[
                island_end_idx:].tolist()
            slices = slices[
                island_start_idx:island_end_idx]

        # Interpolate the missing or misplaced slices and insert them into a
        # slice array with new dimensions.
        volume_arr = interpolate_orthogonal_slices(slice_arr=volume_arr,
                                                   new_positions=new_positions,
                                                   old_positions=unique_positions,
                                                   method=method,
                                                   fill_value=fill_value)
        # DO NOT FORGET TO UPDATE THE LIST OF ALL POSITIONS!
        unique_positions = new_positions

    ############################################################################
    # COLLECT ADDITIONAL META-INFORMATION.

    # Just take the first n_modalities slices to get the metadata from the
    # headers, as all slices that are left in the list of slices are all
    # correctly positioned, complete, and ordered in terms of their modalities.
    acquisition_time = [get_acquisition_time(slice) for slice in
                        slices[0:n_modalities]]

    # Attribute named "Patient's Weight"
    weight = [get_dicom_attribute(slice=slice, address=[0x0010, 0x1030],
                                  default=None, dtype=float) for slice
              in slices[0:n_modalities]]
    injected_dose = [get_injected_dose(slice=slice) for slice in slices[
        0:n_modalities]]
    # IMPORTANT: All modalities should, if everything works as intended,
    # share the same  decay factor, because in order
    # to compute SUV values the difference between the first timepoint and
    # injection have to be considered.
    decay_factor = [get_decay_factor(slice=slice) for slice in slices[
        0:n_modalities]]

    return slices, outlier_slices, volume_arr, unique_positions, \
        acquisition_time, weight, injected_dose, decay_factor


def combine_slices(slices: list,
                   shape: tuple = None,
                   fill_value: float = 0):
    """
    Takes several slices (ideally of the same position) and stacks them in
    the 3rd dimension.
    """
    info_missing = False
    modality_mismatch = False
    # Check whether the slice contains information about the
    # time it was recorded.
    # Attribute named Acquisition Time
    acq_times = np.array([get_dicom_attribute(slice=slice, address=[0x0008,
                                                                    0x0032],
                                              default=None) for slice in
                          slices]).astype(float)
    # Check whether the slice contains information about its
    # instance order.
    # Attribute named Instance Number
    inst_numbers = np.array([get_dicom_attribute(slice=slice, address=[0x0020,
                                                                       0x0013],
                                                 default=None) for slice in
                             slices]).astype(float)
    n_missing_acq_times = np.sum(np.isnan(acq_times))
    n_missing_inst_numbers = np.sum(np.isnan(inst_numbers))
    if n_missing_acq_times == 0:
        sort_basis = acq_times
    elif n_missing_inst_numbers == 0:
        sort_basis = inst_numbers
    else:
        # If only 1 modality is specified, there is inherently no ambiguity
        # in how to order the modalities.
        info_missing = len(slices) > 1
        if n_missing_acq_times < n_missing_inst_numbers:
            acq_times[np.isnan(acq_times)] = np.nanmax(acq_times) + 1
            sort_basis = acq_times
        else:
            inst_numbers[np.isnan(acq_times)] = np.nanmax(inst_numbers) + 1
            sort_basis = inst_numbers

    # Sort the slices according to either the acquisition time or the
    # instance number.
    slices = slices[np.argsort(sort_basis)]
    if shape is not None:
        if shape[-1] != len(slices):
            modality_mismatch = True
            # In case the user specified fewer modalities than are available,
            # just ignore the user's input.
            if shape[-1] < len(slices):
                shape = shape[:2] + (len(slices),)
    else:
        height, width = np.max([slice.pixel_array.shape for slice in slices],
                               axis=0).astype(int)
        shape = (height, width, len(slices))

    position_arr = np.zeros(shape)
    position_arr[:] = fill_value
    for modality_idx, slice in enumerate(slices):
        # DO NOT FORGET: RESCALE THE VALUES USING THE PROVIDED
        # INFORMATION!!!
        # Attribute named "Rescale Intercept"
        b = get_dicom_attribute(slice, address=[0x0028, 0x1052],
                                default=0.0, dtype=float)
        # Attribute named "Rescale Slope"
        m = get_dicom_attribute(slice, address=[0x0028, 0x1053],
                                default=1.0, dtype=float)
        slice_arr = (m * slice.pixel_array) + b

        pad_height = ((shape[0] - slice_arr.shape[0]) // 2) + (
                (shape[0] - slice_arr.shape[0]) % 2)
        pad_width = ((shape[1] - slice_arr.shape[1]) // 2) + (
                (shape[1] - slice_arr.shape[1]) % 2)

        position_arr[pad_height:pad_height + slice_arr.shape[0],
        pad_width:pad_width + slice_arr.shape[1],
        modality_idx] = slice_arr

    return slices, position_arr, info_missing, modality_mismatch


def transform_ordered_slices_to_4D_array(slices: np.ndarray,
                                         n_modalities_per_position: np.ndarray,
                                         position_idxs=None,
                                         pad_element: float = 0,
                                         dtype=np.float16,
                                         ignore_spacing: bool = True):
    """
    Handles the creation of the array representing the data contained within 
    an ordered list of DICOM slices.
    """
    n_modalities_per_position = np.array(n_modalities_per_position)
    if np.any(n_modalities_per_position != n_modalities_per_position[0]):
        warn("Not all slices have the same number of modalities provided. "
             "The number of modalities per slice are:\n"
             F"{n_modalities_per_position},\n"
             F"Padding the missing modalities with {pad_element}.\n")

    # In case only certain positions within the DICOM files are supposed to be
    # considered.
    if position_idxs is not None:
        if np.isscalar(position_idxs):
            position_idxs = [int(position_idxs)]
        relevant_slice_idxs = np.sort(np.array([np.arange(
            n_modalities_per_position[single_slice_idx]) + np.sum(
            n_modalities_per_position[:single_slice_idx]) for single_slice_idx
                                                in
                                                position_idxs]).flatten()).astype(
            int)
    else:
        position_idxs = np.arange(len(n_modalities_per_position))
        relevant_slice_idxs = np.arange(len(slices))

    pixel_arr_dims = np.array([slices[slice_idx].pixel_array.shape for \
                               slice_idx in
                               relevant_slice_idxs])

    if np.any(pixel_arr_dims != pixel_arr_dims[0]):
        warn("Not all slices have the same shape for their pixel array. The "
             "different shapes of the pixel arrays are:\n" +
             ", ".join(map(str, np.unique(pixel_arr_dims, axis=0)))
             + F"\nPadding the missing pixels with {pad_element}.\n")

    max_height, max_width = np.max(pixel_arr_dims, axis=0)
    # In case the array is supposed to be spacing-sensitive, create an array
    # that fills all positions between the min- and max- indices that are not
    # contained in the position_idx parameter with all-zero-slices.
    if ignore_spacing:
        result_arr: np.ndarray = np.zeros(
            (len(position_idxs),
             max_height,
             max_width,
             np.max(n_modalities_per_position)))
        loop_positions = position_idxs
    else:
        result_arr = np.zeros(
            (max(position_idxs) - min(position_idxs) + 1,
             max_height,
             max_width,
             np.max(n_modalities_per_position)))
        loop_positions = range(min(position_idxs), max(position_idxs) + 1)
    result_arr[:] = pad_element

    start_index: int = 0
    for position_number, single_position_idx in enumerate(loop_positions):
        # This check makes sure that pixel arrays are only accessed for
        # non-filler-positions. Only relevant if ignore_spacing is False.
        if single_position_idx in position_idxs:
            n_modalities = n_modalities_per_position[single_position_idx]
            for modality_number, slice in enumerate(slices[relevant_slice_idxs][
                                                        start_index:start_index +
                                                                    n_modalities]):
                # DO NOT FORGET: RESCALE THE VALUES USING THE PROVIDED
                # INFORMATION!!!
                try:
                    b = slice.RescaleIntercept
                except AttributeError:
                    b = 0
                try:
                    m = slice.RescaleSlope
                except AttributeError:
                    m = 1
                pixel_arr: np.ndarray = (m * slice.pixel_array) + b

                pad_height: int = ((max_height - pixel_arr.shape[0]) // 2) + (
                        (max_height - pixel_arr.shape[0]) % 2)
                pad_width: int = ((max_width - pixel_arr.shape[1]) // 2) + (
                        (max_width - pixel_arr.shape[1]) % 2)

                result_arr[position_number,
                pad_height:pad_height + pixel_arr.shape[0],
                pad_width:pad_width + pixel_arr.shape[1],
                modality_number] = pixel_arr

            start_index += n_modalities

    return result_arr.astype(dtype)


def fuse_dicom_GRIDDATA(directory_list: List[str]):
    """
    Takes two different sets of DICOM data and fuses them into one array,
    assuming that the two datasets share the same coordinate system.
    """
    # NOTE: Currently, only two directories can be fused. Change in future!
    directory_path_target: str = directory_list[0]
    directory_path_moving: str = directory_list[-1]

    slices_target, n_modalities_per_position_target, \
        acquisition_times_target, weight_target, injected_dose_target, \
        decay_factor_target \
        = load_and_order_dicom_slices(
        directory_path=directory_path_target,
        discard_outliers=True)
    slices_moving, n_modalities_per_position_moving, \
        acquisition_times_moving, weight_moving, injected_dose_moving, \
        decay_factor_moving \
        = load_and_order_dicom_slices(
        directory_path=directory_path_moving,
        discard_outliers=True)

    # Attribute named "Image Orientation (Patient)"
    image_orientation_patient_target: np.ndarray = np.array(list(
        slices_target[0][0x0020, 0x0037].value))
    image_orientation_patient_moving: np.ndarray = np.array(
        list(slices_moving[0][0x0020, 0x0037].value))

    # Attribute named "Image Position (Patient)"
    image_position_patient_1_target: np.ndarray = np.array(
        list(slices_target[0][0x0020, 0x0032].value))
    image_position_patient_1_moving: np.ndarray = np.array(
        list(slices_moving[0][0x0020, 0x0032].value))

    image_position_patient_N_target: np.ndarray = np.array(
        list(slices_target[-1][0x0020, 0x0032].value))
    image_position_patient_N_moving: np.ndarray = np.array(
        list(slices_moving[-1][0x0020, 0x0032].value))

    # Attribute named "Pixel Spacing"
    pixel_spacing_target: list = list(slices_target[0][0x0028, 0x0030].value)
    pixel_spacing_moving: list = list(slices_moving[0][0x0028, 0x0030].value)

    N_target: int = len(n_modalities_per_position_target)
    N_moving: int = len(n_modalities_per_position_moving)

    array_target: np.ndarray = transform_ordered_slices_to_4D_array(
        slices=slices_target,
        n_modalities_per_position=n_modalities_per_position_target)
    array_moving: np.ndarray = transform_ordered_slices_to_4D_array(
        slices=slices_moving,
        n_modalities_per_position=n_modalities_per_position_moving)

    voxel_positions_target: np.ndarray = get_affine_voxel_coordinates(
        array_shape=array_target.shape,
        image_orientation_patient=image_orientation_patient_target,
        image_position_patient_1=image_position_patient_1_target,
        image_position_patient_N=image_position_patient_N_target,
        pixel_spacing=pixel_spacing_target,
        N=N_target
    )
    voxel_positions_moving: np.ndarray = get_affine_voxel_coordinates(
        array_shape=array_moving.shape,
        image_orientation_patient=image_orientation_patient_moving,
        image_position_patient_1=image_position_patient_1_moving,
        image_position_patient_N=image_position_patient_N_moving,
        pixel_spacing=pixel_spacing_moving,
        N=N_moving
    )
    corrected_voxel_positions_moving = voxel_positions_moving

    # rgi: RegularGridInterpolator = RegularGridInterpolator(
    #    tuple(voxel_positions_moving.T), array_moving, method="nearest")

    interpolation_results: np.ndarray = griddata(
        points=corrected_voxel_positions_moving,
        values=array_moving.reshape((
            voxel_positions_moving.shape[0],
            array_moving.shape[-1])),
        xi=voxel_positions_target,
        method="nearest")

    # Combining the target array with the interpolated
    # data from the moving array.
    fusion_arr: np.ndarray = np.zeros(array_target.shape[:-1] + (
        array_target.shape[-1] + array_moving.shape[-1],))
    fusion_arr[..., :array_target.shape[-1]] = array_target
    fusion_arr[..., array_target.shape[-1]:] = interpolation_results.reshape(
        array_target.shape[:-1] + (array_moving.shape[-1],))

    ###### WATCH OUT: THIS SECTION HAS TO BE DRASTICALLY CHANGED ONCE THE
    # VOXEL ARRAY CLASS CAN ACCOUNT FOR MORE DYNAMICALLY SHAPED
    # METADATA!!!######
    if slices_target[0].Modality == 'PT':
        try:
            frame_count: int = int(slices_target[0][0x0054, 0x0101].value)
        except KeyError:
            frame_count = 1
        features_target: List[str] = [F"Frame {i + 1} - Target" for i in range(
            frame_count)]
        metadata_target: List[str] = [str(slices_target[i]) for i in range(
            frame_count)]
    else:
        features_target: List[str] = [str(slices_target[0].Modality)
                                      + " - Target"]
        metadata_target: List[str] = [str(slices_target[0])]

    if slices_moving[0].Modality == 'PT':
        try:
            frame_count: int = int(slices_moving[0][0x0054, 0x0101].value)
        except KeyError:
            frame_count = 1
        features_moving: List[str] = [F"Frame {i + 1} - Moving" for i in range(
            frame_count)]
        metadata_moving: List[str] = [str(slices_moving[i]) for i in range(
            frame_count)]
    else:
        features_moving: List[str] = [str(slices_moving[0].Modality)
                                      + " - Moving"]
        metadata_moving: List[str] = [str(slices_moving[0])]

    voxel_array_collection = []

    # Obviously, this line needs to be changed one more than two DICOM
    # directories can be merged.
    for i, slices in enumerate([slices_target, slices_moving]):
        # NOTE: Since the pixels of the moving object were adjusted to the
        # target object, its pixel spacing and slice thickness will be that
        # of the target object.
        if i == 0:
            arr = fusion_arr[..., :array_target.shape[-1]]
            metadata = metadata_target
            name = slices_target[0].PatientName
            acquisition_times = acquisition_times_target
            features = features_target
            weight = weight_target
            injected_dose = injected_dose_target
        else:
            arr = fusion_arr[..., array_target.shape[-1]:]
            metadata = metadata_moving
            name = slices_moving[0].PatientName
            acquisition_times = acquisition_times_moving
            features = features_moving
            weight = weight_moving
            injected_dose = injected_dose_moving

        # Attribute named "Slice Thickness" (0018,0050)
        slice_thickness: float = slices_target[0].SliceThickness
        voxel_size: List[float] = pixel_spacing_target + [slice_thickness]

        # voxel_array_collection.append(VoxelArray(source_voxel=arr,
        #                                         metadata=metadata,
        #                                         name=name,
        #                                         channels=features,
        #                                         voxel_size=voxel_size,
        #
        #                                         acquisition_time=acquisition_times,
        #                                         weight=weight,
        #                                         injected_dose=injected_dose))

    return voxel_array_collection


def is_DICOM_directory(path: str):
    """
    Check whether a provided path leads to
    a directory containing DICOM files or not.
    """
    # In case the path leads to a file and not a directory, return immediately.
    if os.path.isfile(path):
        return False
    file_paths = [os.path.join(path, file_name) for file_name in
                  os.listdir(path) if
                  os.path.isfile(os.path.join(path, file_name))]
    for file_path in file_paths:
        try:
            pydicom.dcmread(file_path)
            return True
        # The OSError accounts for files that are not DICOM, and the IndexError
        # accounts for empty directories.
        except (OSError, IndexError, InvalidDicomError):
            continue
    else:
        return False


def convert_DICOM_to_HDF(directory_path: str,
                         output_path: str):
    """
    Convert a DICOM directory to a .hdf file containing all the metadata of
    the first slice of the DICOM stack (FOR NOW).
    """
    array, name, features, voxel_size, acquisition_times, weight, \
        injected_dose, decay_factor, metadata, n_features_per_volume, affine, \
        roi_names_per_modality = \
        fuse_and_convert_DICOM_NIFTI(volume_paths=directory_path,
                                     output_path=None)

    first_slice = pydicom.dcmread(
        directory_path + "/" + os.listdir(directory_path)[0])
    header_df = dicom_header_to_dict(dataset=first_slice)

    with h5py.File(output_path, 'w') as out_file:
        dataset_voxel: h5py.Dataset = out_file.create_dataset('voxel_array',
                                                              data=array)
        for attribute in set(header_df.columns):
            item = header_df[attribute].iloc[0]
            if isinstance(item, pydicom.sequence.Sequence):
                item = dicom_header_to_dict(item).to_dict()
                item = list(item)
            elif isinstance(item, pydicom.multival.MultiValue):
                item = list(item)
            elif isinstance(item, pd.Series):
                new_item = []
                for index, value in item.items():
                    if isinstance(value, bytes):
                        if len(value) > 100:
                            continue
                        else:
                            new_item.append(str(value))
                    else:
                        new_item.append(value)

                item = new_item
            elif isinstance(item, pydicom.uid.UID):
                item = str(item)
            elif isinstance(item, pydicom.valuerep.PersonName):
                item = str(item)
            elif isinstance(item, bytes):
                item = str(item)

            dataset_voxel.attrs[attribute] = item if item is not None else ""

        dataset_voxel.attrs['name'] = str(name)
        dataset_voxel.attrs['features'] = list(features)
        dataset_voxel.attrs['voxel_size'] = list(voxel_size)
        dataset_voxel.attrs['acquisition_times'] = list(acquisition_times)
        dataset_voxel.attrs['metadata'] = list(metadata)

        dataset_voxel.attrs['min_x_value'] = 0
        dataset_voxel.attrs['min_y_value'] = 0
        dataset_voxel.attrs['min_z_value'] = 0

    print(F"Converted DICOM directory at {directory_path} to a .hdf file at "
          F"{output_path}.")

    return


def dicom_header_to_dict(dataset):
    """
    Helper function to convert all attributes of a DICOM slide into a dataframe.
    """
    if isinstance(dataset, pydicom.Dataset):
        attributes = [attribute for attribute in dataset.dir() if
                      (attribute != "Overlay Data") & (
                              attribute != "Pixel Data")]
        values = []
        invalid_idxs = []
        for i, attribute in enumerate(attributes):
            try:
                value = dataset[attribute].value
                values.append(value)
            except (ValueError, KeyError):
                invalid_idxs.append(i)
        # Delete all attributes that raised a value error, i.e. didn't have a
        # valid string name attached to them. MAYBE CHANGE THIS LATER?
        attributes = [attribute for i, attribute in enumerate(attributes) if i
                      not in invalid_idxs]
        dict_header = {}
        for attribute in attributes:
            val = dataset[attribute].value
            if isinstance(val, pydicom.dataelem.RawDataElement):
                val = pydicom.dataelem.DataElement_from_raw(val)
            dict_header[attribute] = val
    else:
        dict_header = dataset

    return dict_header


def crop_HDF_arrays(input_path: str,
                    output_path: str,
                    x_min: int, x_max: int,
                    y_min: int, y_max: int,
                    z_min: int, z_max: int):
    """
    Crops the actual array of a voxel array stored in a .hdf file
    and updates said file.
    """
    if (x_min < 0) | (x_max < 0) | \
            (y_min < 0) | (y_max < 0) | \
            (z_min < 0) | (z_max < 0):
        raise ValueError("Min values for each dimension cannot be smaller "
                         "than 0.")

    with h5py.File(input_path, 'r') as in_file:
        dataset_voxel_input: h5py.Dataset = in_file['voxel_array']

        new_voxel_array = dataset_voxel_input[z_min:z_max, y_min:y_max,
        x_min:x_max]

        try:
            dataset_coordinates_input: h5py.Dataset = in_file[
                'coordinates_array']
            new_coordinates_array = dataset_coordinates_input[z_min:z_max,
            y_min:y_max,
            x_min:x_max]
            coordinates_present = True
        except KeyError:
            coordinates_present = False

        with h5py.File(output_path, 'w') as out_file:
            dataset_voxel_output: h5py.Dataset = out_file.create_dataset(
                'voxel_array',
                data=new_voxel_array)

            for key in dataset_voxel_input.attrs.keys():
                dataset_voxel_output.attrs[key] = dataset_voxel_input.attrs.get(
                    key)

            if coordinates_present:
                dataset_coordinates_output: h5py.Dataset = \
                    out_file.create_dataset('coordinates_array',
                                            data=new_coordinates_array)

            out_file.close()

        in_file.close()
    return


def interpolate_voxels_from_HDF(input_path: str,
                                output_path: str,
                                voxel_size_x: float,
                                voxel_size_y: float,
                                voxel_size_z: float,
                                method="nearest"):
    with h5py.File(input_path, 'r') as in_file:
        dataset_voxel_input: h5py.Dataset = in_file['voxel_array']

        input_arr = dataset_voxel_input[...]

        try:
            dataset_coordinates_input: h5py.Dataset = in_file[
                'coordinates_array']
            coordinates_input = dataset_coordinates_input[...]

        except KeyError:
            return ValueError("The specified .hdf file does not contain a "
                              "coordinates dataset.")

        len_x = np.linalg.norm(coordinates_input[0, 0, -1] - \
                               coordinates_input[0, 0, 0])
        len_y = np.linalg.norm(coordinates_input[0, -1, 0] - \
                               coordinates_input[0, 0, 0])
        len_z = np.linalg.norm(coordinates_input[-1, 0, 0] - \
                               coordinates_input[0, 0, 0])

        dim_x = max(int(np.floor(len_x / voxel_size_x)), 1)
        dim_y = max(int(np.floor(len_y / voxel_size_y)), 1)
        dim_z = max(int(np.floor(len_z / voxel_size_z)), 1)

        array_shape: tuple = (dim_z, dim_y, dim_x)

        try:
            image_orientation_patient = np.array(
                list(dataset_voxel_input.attrs["Image " \
                                               "Orientation (" \
                                               "Patient)"]))
        except KeyError:
            image_orientation_patient = np.array(
                list(dataset_voxel_input.attrs["ImageOrientationPatient"]))

        image_position_patient_1 = coordinates_input[0, 0, 0]
        image_position_patient_N = image_position_patient_1 + (
                coordinates_input[-1, 0,
                0] - image_position_patient_1) / \
                                   len_z * voxel_size_z * dim_z
        N = dim_z
        pixel_spacing = [voxel_size_y, voxel_size_x]

        print(F"Shape of new array: {array_shape}.")

        coordinates_output = []
        interpolation_output = []

        for slice_idx in tqdm(range(array_shape[0]), desc="Interpolating "
                                                          "slices"):
            coordinates_slice = get_affine_voxel_coordinates(
                array_shape=array_shape,
                image_orientation_patient=image_orientation_patient,
                image_position_patient_1=
                image_position_patient_1,
                image_position_patient_N=image_position_patient_N,
                N=N,
                pixel_spacing=pixel_spacing,
                slice_idx=slice_idx)

            interpolation_slice: np.ndarray = griddata(
                points=coordinates_input.reshape((-1, 3)),
                values=input_arr.reshape((-1, input_arr.shape[-1])),
                xi=coordinates_output,
                method=method)

            if len(coordinates_output) == 0:
                coordinates_output = coordinates_slice
            else:
                coordinates_output = np.append(coordinates_output,
                                               coordinates_slice, axis=0)

            if len(interpolation_output) == 0:
                interpolation_output = interpolation_slice
            else:
                interpolation_output = np.append(interpolation_output,
                                                 interpolation_slice, axis=0)

        output_arr = interpolation_output.reshape(
            array_shape + (input_arr.shape[-1],))
        coordinates_output = coordinates_output.reshape(
            output_arr.shape[:-1] + (3,))

        with h5py.File(output_path, 'w') as out_file:
            dataset_voxel_output: h5py.Dataset = out_file.create_dataset(
                'voxel_array', data=output_arr)

            for key in dataset_voxel_input.attrs.keys():
                dataset_voxel_output.attrs[key] = dataset_voxel_input.attrs.get(
                    key)

            dataset_voxel_output.attrs["voxel_size"] = [voxel_size_x,
                                                        voxel_size_y,
                                                        voxel_size_z]

            dataset_coordinates_output: h5py.Dataset = out_file.create_dataset(
                'coordinates_array', data=coordinates_output)

            out_file.close()

        in_file.close()

    return


def fuse_HDF_files(input_path_list: list,
                   output_path: str,
                   method: str = "nearest"):
    """
    Fuse two or more hdf files togethers, using their coordinate information
    and interpolation.
    """
    # IMPORTANT SYNTAX: The first file in the file-list will always be
    # considered as the target for all other files.
    with h5py.File(input_path_list[0], 'r') as target_file:
        dataset_voxel_target: h5py.Dataset = target_file['voxel_array']
        fusion_arr: np.ndarray = dataset_voxel_target[...]
        target_shape: tuple = fusion_arr.shape[:3]

        features: list = list(dataset_voxel_target.attrs.get('features'))
        acquisition_times: list = list(
            dataset_voxel_target.attrs.get('acquisition_times'))
        metadata: list = list(dataset_voxel_target.attrs.get('metadata'))

        dataset_coordinates_target: h5py.Dataset = target_file[
            'coordinates_array']
        # Make the array 2-dimensional.
        coordinates_target: np.ndarray = dataset_coordinates_target[
            ...].reshape((-1, 3))

        for input_path in input_path_list[1:]:
            with h5py.File(input_path, 'r') as moving_file:
                dataset_voxel_moving: h5py.Dataset = moving_file['voxel_array']
                moving_arr: np.ndarray = dataset_voxel_moving[...]
                # Make the array 2-dimensional.
                moving_arr = moving_arr.reshape((-1, moving_arr.shape[-1]))

                dataset_coordinates_moving: h5py.Dataset = moving_file[
                    'coordinates_array']
                coordinates_moving: np.ndarray = dataset_coordinates_moving[
                    ...].reshape((-1, 3))

                interpolation_result: np.ndarray = griddata(
                    points=coordinates_moving,
                    values=moving_arr,
                    xi=coordinates_target,
                    method=method)
                interpolated_arr: np.ndarray = interpolation_result.reshape(
                    target_shape + (moving_arr.shape[-1],))

                # Update the results so far.
                fusion_arr = np.append(fusion_arr, interpolated_arr, axis=3)
                features += list(dataset_voxel_moving.attrs.get('features'))
                acquisition_times += list(dataset_voxel_moving.attrs.get(
                    'acquisition_times'))
                metadata += list(dataset_voxel_moving.attrs.get('metadata'))

                moving_file.close()

        with h5py.File(output_path, 'w') as out_file:
            dataset_voxel_fusion: h5py.Dataset = out_file.create_dataset(
                'voxel_array', data=fusion_arr)

            for key in dataset_voxel_target.attrs.keys():
                dataset_voxel_fusion.attrs[key] = \
                    dataset_voxel_target.attrs.get(
                        key)

            dataset_voxel_fusion.attrs["features"] = features
            dataset_voxel_fusion.attrs["acquisition_times"] = acquisition_times
            dataset_voxel_fusion.attrs["metadata"] = metadata

            dataset_coordinates_fusion: h5py.Dataset = \
                out_file.create_dataset(
                    'coordinates_array', data=coordinates_target.reshape(
                        fusion_arr.shape[:-1] + (3,)))

            out_file.close()

        target_file.close()

    return


################################################################################
# AFFINE TRANSFORMATION


def get_affine_voxel_coordinates(array_shape: tuple,
                                 image_orientation_patient: np.ndarray,
                                 image_position_patient_1: np.ndarray,
                                 image_position_patient_N: np.ndarray,
                                 pixel_spacing: list,
                                 N: int,
                                 slice_idx=None):
    """
    Operates the conversion from the affine array coordinates to the absolute
    physical coordinates from the scanner for each voxel.
    """
    dr: float = pixel_spacing[0]
    dc: float = pixel_spacing[1]

    # Make sure that a division by 0 is not encountered.
    N = max(N, 1)

    # Ensure unit length.
    F_1 = image_orientation_patient[3:] / np.linalg.norm(
        image_orientation_patient[3:])
    F_2 = image_orientation_patient[:3] / np.linalg.norm(
        image_orientation_patient[:3])

    column_A_1: np.ndarray = np.append(F_1 * dr, 0)
    column_A_2: np.ndarray = np.append(F_2 * dc, 0)
    column_A_3: np.ndarray = np.append((image_position_patient_N
                                        - image_position_patient_1) / (N - 1),
                                       0)
    column_A_4: np.ndarray = np.append(image_position_patient_1, 1)
    A: np.ndarray = np.stack([column_A_1, column_A_2, column_A_3,
                              column_A_4]).T.astype(np.float16)
    if slice_idx is None:
        arr_indices: np.ndarray = np.vstack(np.where(np.ones(array_shape[
                                                                 :3]))).T.astype(
            np.int16)
    elif np.isscalar(slice_idx):
        arr_indices: np.ndarray = np.vstack(np.where(np.ones(array_shape[
                                                                 1:3]))).T.astype(
            np.int16)
        arr_indices = np.append([[slice_idx]] * arr_indices.shape[0],
                                arr_indices,
                                axis=1)
    # In case it is a list.
    else:
        arr_indices = np.vstack(
            np.where(np.ones((len(slice_idx),) + array_shape[1:3]))).T
        arr_indices_reference = np.copy(arr_indices[:, 0])
        # Assign the correct z-value to the coordinates given the passed slice
        # indices.
        for i, single_slice_idx in enumerate(slice_idx):
            slice_locs = np.where(arr_indices_reference == i)
            arr_indices[slice_locs, 0] = single_slice_idx
        # Enforce the deletion of the now obsolete z-index reference array.
        del arr_indices_reference
    # arr_indices: np.ndarray = np.array(list(np.ndindex(array_shape[:3])))

    # Swapping the first and third column so that the dimension-order for the
    # coordinates is row-column-slice, instead of slice-row-column.
    arr_indices[:, [0, 1, 2]] = arr_indices[:, [1, 2, 0]]

    # Adding a constant 1 column to make the matrix multiplication work.
    input_vectors: np.ndarray = np.pad(arr_indices, ((0, 0), (0, 1)),
                                       'constant',
                                       constant_values=1).astype(np.int16)
    # Remove the filler 1 column.
    voxel_positions: np.ndarray = np.dot(A, input_vectors.T).T[:, :-1]

    return voxel_positions


################################################################################
# FUSION AND CONVERSION


def fuse_and_convert_DICOM_NIFTI(volume_paths,
                                 output_path=None,
                                 voxel_size=None,
                                 method="linear",
                                 fill_value=None,
                                 file_format: str = "hdf",
                                 only_3D_NIFTI: bool = False):
    """
    THE downscaling and fusion function. Ignore the rest.
    """
    DEFAULT_ORIENTATION = "axial"

    # Make sure the function is also working when only a single DICOM
    # directory is passed as a string.
    if isinstance(volume_paths, str):
        volume_paths = [volume_paths]

    # Make sure that the method parameter is a list and has the same length
    # as the dicom directories.
    if isinstance(method, str):
        method = [method] * len(volume_paths)
    elif 0 < len(method) < len(volume_paths):
        method = list(method) + (
                [method[-1]] * (len(volume_paths) - len(method)))
    elif len(method) < 1:
        # Failsafe if the length of method is zero.
        method = ["nearest"] * len(volume_paths)

    # Make sure that the fill_value parameter is a list and has the same length
    # as the dicom directories.
    if not isinstance(fill_value, Iterable):
        fill_value = [fill_value] * len(volume_paths)
    elif 0 < len(fill_value) < len(volume_paths):
        fill_value = list(fill_value) + [fill_value[-1]] * (
                len(volume_paths) - len(fill_value))
    elif len(fill_value) < 1:
        # Failsafe if the length of fill_value is zero.
        fill_value = [None] * len(volume_paths)

    # Ensure that the voxel size is iterable.
    # x-spacing/ y-spacing / z-spacing
    if isinstance(voxel_size, int) | isinstance(voxel_size, float):
        voxel_size = np.repeat(voxel_size, 3)
    elif isinstance(voxel_size, list):
        voxel_size = np.array(voxel_size)

    # This is only really relevant if a Siemens .csv file was loaded.
    roi_names_per_channel = []

    # TODO: Think about how to handle DICOM directories with inconsistent
    #  slice distances. This would also allow you to be more flexible with
    #  outliers.
    # Note that we do always discard outliers for now, but might change this
    # in the future.
    target_path = volume_paths[0]
    if is_DICOM_directory(target_path):
        slices, slices_outlier, slices_arr, positions, acquisition_times, \
            weight, injected_dose, decay_factor = load_and_order_dicom_slices(
            directory_path=target_path,
            method=method[0],
            fill_value=fill_value[0])

        # These two variables are needed only after the interpolation process is
        # finished.
        header_df = dicom_header_to_dict(dataset=slices[0])
        name = slices[0].PatientName

        # Attribute named "Image Orientation (Patient)"
        image_orientation_patient_target = np.array(list(
            slices[0][0x0020, 0x0037].value))
        # Attribute named "Image Position (Patient)"
        image_position_patient_1_target = positions[0]
        image_position_patient_N_target = positions[-1]
        # Attribute named "Pixel Spacing"
        pixel_spacing_target = list(slices[0][0x0028, 0x0030].value)
        slice_thickness_target = slices[0].SliceThickness
        N_target = slices_arr.shape[0]

        coordinates_first_slice = get_affine_voxel_coordinates(
            array_shape=(1,) + slices_arr.shape[1:3],
            image_orientation_patient=image_orientation_patient_target,
            image_position_patient_1=image_position_patient_1_target,
            image_position_patient_N=image_position_patient_N_target,
            pixel_spacing=pixel_spacing_target,
            N=N_target,
            slice_idx=0).reshape(slices_arr.shape[1:3] + (3,))

        len_x = np.linalg.norm(coordinates_first_slice[0, -1] - \
                               coordinates_first_slice[0, 0])
        len_y = np.linalg.norm(coordinates_first_slice[-1, 0] - \
                               coordinates_first_slice[0, 0])
        len_z = np.linalg.norm(image_position_patient_N_target - \
                               image_position_patient_1_target)

        # ALWAYS RETURN THE CORRESPONDING NIFTI AFFINE MATRIX.
        nifti_affine = handler_nifti.get_affine_from_DICOM(slices[0])
    elif handler_nifti.is_nifti(target_path):
        nifti_arr, nifti_affine, nifti_header = handler_nifti.load_NIFTI(
            path=target_path,
            decode=True)
        if nifti_arr.ndim == 3:
            nifti_arr = nifti_arr[..., None]

        name = os.path.basename(target_path)
        N_target = nifti_arr.shape[0]

        pixel_spacing_target, \
            slice_thickness_target, \
            image_orientation_patient_target, \
            image_position_patient_1_target, \
            image_position_patient_N_target \
            = handler_nifti.get_DICOM_attrs_from_affine(
            affine=nifti_affine, n_slices=N_target, decode=False)

        acquisition_times = [None] * nifti_arr.shape[3]
    elif handler_siemens.is_siemens(path=target_path):
        nifti_arr, nifti_affine, _, acquisition_times = \
            handler_siemens.convert_siemens_to_volume(
                siemens_file=target_path,
                mask_only=False,
                padding=0)
    else:
        raise ValueError("The first specified volume must either be a DICOM "
                         "directory, a NIFTI file, or a Siemens .csv file.")

    # Shared handling for NIFTI and Siemens.
    if handler_nifti.is_nifti(target_path) \
            | handler_siemens.is_siemens(target_path):
        weight = [None] * nifti_arr.shape[3]
        injected_dose = [None] * nifti_arr.shape[3]
        decay_factor = [None] * nifti_arr.shape[3]

        len_z = (nifti_arr.shape[0] - 1) * slice_thickness_target
        len_y, len_x = (np.array(
            nifti_arr.shape[1:3]) - 1) * pixel_spacing_target
        # Just work with an empty dataframe, in case the volume is to be
        # saved as a .hdf file.
        header_df = pd.DataFrame()

    # In case no voxel size was provided, just assume the voxel size of the
    # target.
    voxel_size_target = np.append(pixel_spacing_target,
                                  slice_thickness_target)

    if voxel_size is None:
        voxel_size = voxel_size_target
        skip_target_interpolation = True
    else:
        # Make sure to transform the voxel size, which was specified in
        # x/y/z-order, into row-spacing/column-spacing/slice-spacing-order,
        # to comply with the OLD code below.
        orientation, _ = handler_nifti.determine_slice_orientation(
            direction_cosine=image_orientation_patient_target,
            default_orientation=DEFAULT_ORIENTATION)
        # COMMENTED OUT FOR NOW.
        # Include back in, in order to return to the ordering of x-y-z for
        # voxel_size.
        # if orientation.lower() == "axial":
        #    voxel_size = voxel_size[[1, 0, 2]]
        # elif orientation.lower() == "coronal":
        #    voxel_size = voxel_size[[2, 0, 1]]
        # else:
        #    voxel_size = voxel_size[[2, 1, 0]]

        # Make sure to provide the correct voxel size for the affine matrix,
        # in case the user changed it during interpolation.
        nifti_affine = handler_nifti.update_affine_voxel_size(
            affine=nifti_affine,
            voxel_size=voxel_size)
        # Skip the target interpolation, in case the specified voxel size is
        # equal to the one already present in the target volume.
        skip_target_interpolation = np.all(voxel_size == voxel_size_target)

    dim_x = int(np.round(len_x / voxel_size[1], 0)) + 1
    dim_y = int(np.round(len_y / voxel_size[0], 0)) + 1
    dim_z = int(np.round(len_z / voxel_size[2], 0)) + 1

    interpolated_array_shape: tuple = (dim_z, dim_y, dim_x)

    interpolation_fused = None
    channels_per_dir = []
    metadata_fused = []
    acquisition_times_fused = []
    weight_fused = []
    injected_dose_fused = []
    decay_factor_fused = []

    for path_idx, volume_path in enumerate(volume_paths):
        # In case no voxel size was specified, we can skip interpolating the
        # target volume, as it will be replicated 1-to-1 by the chosen
        # interpolation settings.
        if (path_idx == 0) & skip_target_interpolation:
            if is_DICOM_directory(target_path):

                # Duplicate code from interpolate_modality. Currently, this is
                # the only way to get the attributes for the target volume,
                # if the interpolation for the target volume
                frame_count = slices_arr.shape[3]
                if frame_count > 1:
                    channels = [F"{slices[0].Modality}_Frame_{i + 1}" for i in
                                range(frame_count)]
                else:
                    channels = [slices[0].Modality]

                metadata = [str(slices[i]) for i in range(frame_count)]
                # To avoid errors from inconsistent slice distance information.
                if np.any(
                        np.array(interpolated_array_shape) != slices_arr.shape[
                            :3]):
                    print(interpolated_array_shape)
                    print(slices_arr.shape)
                    warnings.warn("There are some inconsistencies in slice "
                                  "distances in the target DICOM directory.")
                    interpolated_array_shape = slices_arr.shape[:3]
            elif handler_nifti.is_nifti(path=target_path):
                slices_arr = nifti_arr
                channels = [F"Channel_{0 if i < 9 else ''}{i + 1}" for i in
                            range(slices_arr.shape[-1])]
                metadata = [nifti_header] * slices_arr.shape[-1]
            else:
                slices_arr = nifti_arr
                channels = [F"Channel_{0 if i < 9 else ''}{i + 1}" for i in
                            range(slices_arr.shape[-1])]
                metadata = channels

            # The voxels need to be sorted in a 2D-list, in order for the
            # interpolation to work.
            interpolation_modality = slices_arr.reshape((-1, slices_arr.shape[
                -1]))

            # In order to keep the roi_names_per_channel list at the same
            # length as the total number of channels.
            roi_names = [None] * len(channels)
        else:
            desc = "Interpolating " + ("target" if (
                    path_idx == 0) else F"source {path_idx}")

            # THE INTERPOLATION STEP.
            interpolation_modality, acquisition_times, \
                weight, injected_dose, decay_factor, channels, \
                metadata, roi_names = interpolate_modality(
                modality_path=volume_path,
                interpolated_array_shape=interpolated_array_shape,
                voxel_size=voxel_size,
                image_orientation_patient_target=image_orientation_patient_target,
                image_position_patient_1_target=image_position_patient_1_target,
                image_position_patient_N_target=image_position_patient_N_target,
                method=method[path_idx],
                fill_value=fill_value[path_idx],
                desc=desc)

        # Add the names for each ROI in the ROI volume to the list of
        # ROI names per modality. Note that this will always be None unless a
        # Siemens .csv file is interpolated.
        roi_names_per_channel += roi_names

        # Make sure the interpolated voxels are always 2-dimensional!
        if interpolation_modality.ndim == 1:
            interpolation_modality = interpolation_modality[..., None]

        if interpolation_fused is None:
            interpolation_fused = interpolation_modality
        else:
            interpolation_fused = np.append(interpolation_fused,
                                            interpolation_modality,
                                            axis=1)

        channels_per_dir += [channels]
        metadata_fused += list(metadata)
        acquisition_times_fused += list(acquisition_times)
        weight_fused += list(weight)
        injected_dose_fused += list(injected_dose)
        decay_factor_fused += list(decay_factor)

    # Make sure that there are no duplicate feature names.
    channels_per_dir = np.array(channels_per_dir, dtype=object)
    channels_fused = []
    for path_idx, channels in enumerate(channels_per_dir):
        for feature in channels:
            other_idxs = [i for i in range(len(channels_per_dir)) if
                          i != path_idx]
            if feature in channels_per_dir[other_idxs].flatten():
                channels_fused.append(
                    feature + " - " + ("Target" if (path_idx == 0)
                                       else F"Source_{path_idx}"))
            else:
                channels_fused.append(feature)

    # Change the shape of the interpolation results to a 4D array, for saving.
    interpolation_fused = interpolation_fused.reshape(
        interpolated_array_shape[:3] + (len(channels_fused),))

    # TRANSFORM THE VOLUME INTO AXIAL ORIENTATION, IN CASE IT IS NOT ALREADY.
    interpolation_fused, nifti_affine = handler_nifti.standardize_orientation(
        volume=interpolation_fused,
        affine=nifti_affine,
        orientation=DEFAULT_ORIENTATION,
        decode=False)
    voxel_size = handler_nifti.get_voxel_size(affine=nifti_affine, decode=False)

    ############################################################################
    # In case no output path was provided, simply return the array and all
    # necessary additional information to instantiate a voxel array.
    if output_path is None:
        # This list is needed to separate the features into target and source.
        n_channels_per_volume = [len(channels) for channels in channels_per_dir]
        return interpolation_fused, name, channels_fused, voxel_size, \
            acquisition_times_fused, weight_fused, injected_dose_fused, \
            decay_factor_fused, metadata_fused, n_channels_per_volume, \
            nifti_affine, roi_names_per_channel
    ############################################################################

    if file_format.lower() == "hdf":
        with h5py.File(output_path, 'w') as out_file:
            dataset_voxel = out_file.create_dataset('voxel_array',
                                                    data=interpolation_fused)
            for attribute in set(header_df.columns):
                item = header_df[attribute].iloc[0]
                if isinstance(item, pydicom.sequence.Sequence):
                    item = dicom_header_to_dict(item).to_dict()
                    item = list(item)
                elif isinstance(item, pydicom.multival.MultiValue):
                    item = list(item)
                elif isinstance(item, pd.Series):
                    new_item = []
                    for index, value in item.items():
                        if isinstance(value, bytes):
                            if len(value) > 100:
                                continue
                            else:
                                new_item.append(str(value))
                        else:
                            new_item.append(value)

                    item = new_item
                elif isinstance(item, pydicom.uid.UID):
                    item = str(item)
                elif isinstance(item, pydicom.valuerep.PersonName):
                    item = str(item)
                elif isinstance(item, bytes):
                    item = str(item)

                dataset_voxel.attrs[
                    attribute] = item if item is not None else ""

            dataset_voxel.attrs['name'] = str(name)
            dataset_voxel.attrs['channels'] = list(channels_fused)
            dataset_voxel.attrs['voxel_size'] = list(voxel_size)
            dataset_voxel.attrs['acquisition_times'] = list(
                acquisition_times_fused)
            dataset_voxel.attrs['metadata'] = list(metadata_fused)

            dataset_voxel.attrs['min_x_value'] = 0
            dataset_voxel.attrs['min_y_value'] = 0
            dataset_voxel.attrs['min_z_value'] = 0

            dataset_voxel.attrs['weight'] = weight_fused
            dataset_voxel.attrs['injected_dose'] = injected_dose_fused
            dataset_voxel.attrs['decay_factor'] = decay_factor_fused
            dataset_voxel.attrs['affine'] = nifti_affine

        file_format = "HDF"
    elif file_format.lower() == "nifti":
        encoded_affine = handler_nifti.encode_affine(affine=nifti_affine,
                                                     decoded_array_shape=interpolation_fused.shape)
        # Convert the image data array to the shape found in NIFTI files.
        interpolation_fused = handler_nifti.encode_array(
            arr=interpolation_fused)

        # Make sure to return only 3-dimensional NIFTIs, in case multi-modal
        # DICOM series were provided.
        if only_3D_NIFTI:
            dir = os.path.dirname(os.path.realpath(output_path))
            base_name_forward = output_path.split("/")[-1].split(".")[0]
            base_name_back = output_path.split("\\")[-1].split(".")[0]
            base_name = base_name_forward if len(base_name_forward) < len(
                base_name_back) else base_name_back
            for modality_idx in range(interpolation_fused.shape[-1]):
                suffix = channels_fused[modality_idx]

                nifti_file_modality = nib.Nifti1Image(interpolation_fused[
                                                          ..., modality_idx],
                                                      affine=encoded_affine)
                nib.save(nifti_file_modality, os.path.join(dir,
                                                           base_name + "_" +
                                                           suffix + ".nii"))
        else:
            nifti_file = nib.Nifti1Image(interpolation_fused,
                                         affine=encoded_affine)
            nib.save(nifti_file, output_path)

        file_format = "NIFTI"

    # Otherwise, save as a .npy file.
    else:
        np.save(arr=interpolation_fused.astype(float), file=output_path)

        file_format = "NPY"

    nl = ",\n"
    print(F"Fused DICOM directories at\n"
          F"[{nl.join(volume_paths)}]\nto a {file_format} file at"
          F"\n{output_path}.")
    return


def interpolate_modality(modality_path: str,
                         interpolated_array_shape: tuple,
                         voxel_size: list,
                         image_orientation_patient_target: np.ndarray,
                         image_position_patient_1_target: np.ndarray,
                         image_position_patient_N_target: np.ndarray,
                         method="Linear",
                         fill_value=None,
                         desc: str = "Interpolation"):
    """
    Helper function to execute the actual interpolation step of the DICOM
    fusion process. Importantly, this function includes compatibility with
    Siemens ROI .csv files.
    """
    dim_z = interpolated_array_shape[0]
    interpolation_modality = None

    # This is only needed if a Siemens .csv file is loaded.
    roi_names = None

    # Check whether the volume is stored in a DICOM directory or a Siemens
    # ROI .csv file.
    if is_DICOM_directory(modality_path):
        slices, slices_outliers, dicom_arr, positions, acquisition_times, \
            weight, injected_dose, decay_factor = load_and_order_dicom_slices(
            directory_path=modality_path,
            method=method,
            fill_value=fill_value)

        # In case a directory contains a multi-modal volume, append the
        # suffix "Frame" plus the index of the modality, counting from 1.
        frame_count = dicom_arr.shape[-1]
        if frame_count > 1:
            channels = [F"{slices[0].Modality}_Frame_{i + 1}" for i in
                        range(frame_count)]
        else:
            channels = [slices[0].Modality]

        metadata = [str(slices[i]) for i in range(frame_count)]

        # Attribute named "Image Orientation (Patient)"
        image_orientation_patient = np.array(list(
            slices[0][0x0020, 0x0037].value))
        # Attribute named "Image Position (Patient)"
        image_position_patient_1 = positions[0]
        image_position_patient_N = positions[-1]
        # Attribute named "Pixel Spacing"
        pixel_spacing = list(slices[0][0x0028, 0x0030].value)
        N = dicom_arr.shape[0]

        direction_slice = (image_position_patient_N
                           - image_position_patient_1) / max(N - 1.0, 1.0)
        slice_thickness = np.linalg.norm(direction_slice)

        coordinates_modality = get_affine_voxel_coordinates(
            array_shape=dicom_arr.shape[:3],
            image_orientation_patient=image_orientation_patient,
            image_position_patient_1=image_position_patient_1,
            image_position_patient_N=image_position_patient_N,
            N=N,
            pixel_spacing=pixel_spacing,
            slice_idx=[0, 1] if N > 1 else [0])

        # Reshaping for easier indexing.
        coordinates_modality = coordinates_modality \
            .reshape((-1,) + dicom_arr.shape[1:3] + (3,))

    # SIEMENS CSV.
    elif handler_siemens.is_siemens(path=modality_path):
        # TODO: Enable the interpolation of Siemens files as anything but masks.
        nifti_arr, affine_source, roi_names, _ \
            = handler_siemens.convert_siemens_to_volume(
            siemens_file=modality_path,
            mask_only=True,
            padding=10)

        # This wrapping is needed, because from a Siemens .csv file, only ONE
        # ROI channel can be extracted.
        roi_names = [roi_names]

        # In case a Siemens ROI .csv file was passed, no further information
        # about the volume can be derived.
        acquisition_times, weight, \
            injected_dose, decay_factor = [None], [None], [None], [None]

        # NOTE: This needs to be changed in case the support for Siemens ROI
        # .csv files is supposed to be extended beyond the mere coordinates
        # and to include the specified volume values as well.
        channels = ["ROIs"]
        metadata = channels.copy()
    # NIFTI.
    elif handler_nifti.is_nifti(path=modality_path):
        nifti_arr, affine_source, nifti_header = handler_nifti.load_NIFTI(
            path=modality_path,
            decode=True)

        if nifti_arr.ndim == 3:
            nifti_arr = nifti_arr[..., None]

        n_channels = nifti_arr.shape[3]
        acquisition_times = [None] * n_channels
        metadata = [nifti_header] * n_channels
        roi_names = [None] * n_channels
    else:
        raise ValueError(F"The path {modality_path} does not contain any "
                         F"orientation information.")

    # Shared trunk of NIFTI and Siemens files.
    if handler_nifti.is_nifti(modality_path) \
            | handler_siemens.is_siemens(modality_path):

        # In case a NIFTI/Siemens file was passed, no further information
        # about the volume can be derived.
        N = nifti_arr.shape[0]
        n_channels = nifti_arr.shape[3]
        weight, injected_dose, decay_factor = [None] * n_channels, \
                                              [None] * n_channels, \
                                              [None] * n_channels
        channels = [F"Channel_{i + 1}" for i in range(n_channels)]

        # Check to maybe skip a lot of computation.
        affine_target = handler_nifti.get_affine(
            image_orientation_patient=image_orientation_patient_target,
            image_position_patient=image_position_patient_1_target,
            pixel_spacing=voxel_size[:2],
            slice_thickness=voxel_size[2],
            decoded_array_shape=None,
            encode=False)

        if (interpolated_array_shape[:3] == nifti_arr.shape[:3]) \
                & np.allclose(affine_source, affine_target):
            warnings.warn("Skipping interpolation, as the target and source "
                          "have the same shape and the same affines.")
            return nifti_arr.reshape((-1, nifti_arr.shape[3])), \
                acquisition_times, \
                weight, injected_dose, decay_factor, channels, metadata, roi_names

        pixel_spacing, \
            slice_thickness, \
            image_orientation_patient, \
            image_position_patient_1, \
            image_position_patient_N = handler_nifti.get_DICOM_attrs_from_affine(
            affine=affine_source, n_slices=N, decode=False)

        coordinates_modality = get_affine_voxel_coordinates(
            array_shape=nifti_arr.shape[:3],
            image_orientation_patient=image_orientation_patient,
            image_position_patient_1=image_position_patient_1,
            image_position_patient_N=image_position_patient_N,
            N=N,
            pixel_spacing=pixel_spacing,
            slice_idx=[0, 1] if N > 1 else [0])

        # Reshaping for easier indexing.
        coordinates_modality = coordinates_modality \
            .reshape((-1,) + nifti_arr.shape[1:3] + (3,))

        direction_slice = (image_position_patient_N
                           - image_position_patient_1) / max(N - 1.0, 1.0)

    # Execute a basis change from basis B1 (basis of physical
    # coordinates) to basis B2 (basis of modality array).

    e1_B2 = coordinates_modality[0, 0, 1] - coordinates_modality[
        0, 0, 0]
    e2_B2 = coordinates_modality[0, 1, 0] - coordinates_modality[
        0, 0, 0]
    if N > 1:
        e3_B2 = coordinates_modality[1, 0, 0] - coordinates_modality[
            0, 0, 0]
    else:
        e3_B2 = direction_slice

    A = np.array([e1_B2, e2_B2, e3_B2]).T
    basis_change_matrix = np.linalg.inv(A)

    for slice_idx in tqdm(range(dim_z), desc=desc):
        coordinates_interpolated = get_affine_voxel_coordinates(
            array_shape=interpolated_array_shape,
            image_orientation_patient=image_orientation_patient_target,
            image_position_patient_1=image_position_patient_1_target,
            image_position_patient_N=image_position_patient_N_target,
            N=dim_z,
            pixel_spacing=[voxel_size[0], voxel_size[1]],
            slice_idx=slice_idx)

        coordinates_interpolated = coordinates_interpolated.reshape(
            interpolated_array_shape[1:] + (3,))

        plane_points = [coordinates_interpolated[0, 0],
                        coordinates_interpolated[-1, 0],
                        coordinates_interpolated[0, -1],
                        coordinates_interpolated[-1, -1]]

        coordinates_interpolated = coordinates_interpolated.reshape((-1, 3))
        # The slices of the SOURCE that collide with the current loop target
        # slice.
        collision_slice_idxs = get_collision_slice_range(
            plane_points=plane_points,
            cuboid_point=coordinates_modality[0, 0, 0],
            cuboid_edge_direction=direction_slice,
            slice_distance=slice_thickness,
            n_slices=N)

        if is_DICOM_directory(modality_path):
            slices_arr = dicom_arr[collision_slice_idxs]
        else:
            # No elif check here, because if the file would not fit one of
            # the three accepted data formats, an error would have been
            # raised above.
            slices_arr = nifti_arr[collision_slice_idxs]

        affinity_B2 = coordinates_modality[0, 0, 0] + min(
            collision_slice_idxs) * direction_slice

        affine_coordinates_interpolated = np.dot(
            basis_change_matrix,
            (coordinates_interpolated - affinity_B2).T).T

        # Change from x-y-z to z-y-x to be compatible with
        # RegularGridInterpolator.
        affine_coordinates_interpolated[:, [0, 1, 2]] \
            = affine_coordinates_interpolated[:, [2, 1, 0]]
        affine_coordinates_interpolated = affine_coordinates_interpolated \
            .astype(np.float16)

        # THE ACTUAL INTERPOLATION PROCESS.
        with warnings.catch_warnings():
            # Catch some weird errors that do not impact the
            # interpolation results.
            warnings.simplefilter("ignore")
            interpolation_slice = RegularGridInterpolator(
                points=(np.arange(slices_arr.shape[0]), np.arange(
                    slices_arr.shape[1]),
                        np.arange(slices_arr.shape[2])),
                values=slices_arr,
                method=method,
                fill_value=fill_value,
                bounds_error=False)(affine_coordinates_interpolated)

        if interpolation_modality is None:
            interpolation_modality = interpolation_slice
        else:
            interpolation_modality = np.append(interpolation_modality,
                                               interpolation_slice,
                                               axis=0)

    # Delete the DICOM slices to free up RAM space.
    if is_DICOM_directory(modality_path):
        del slices

    # Return non-informative ROI names, in case anything but a Siemens .csv
    # file was loaded.
    if roi_names is None:
        roi_names = [None] * len(channels)

    return interpolation_modality, acquisition_times, \
        weight, injected_dose, decay_factor, channels, metadata, roi_names


def line_plane_collision(plane_point: np.ndarray,
                         plane_normal: np.ndarray,
                         line_point: np.ndarray,
                         n_dot_u: np.ndarray):
    """
    Determine the location at which a line and a plane collide.
    """
    w = line_point - plane_point
    r = -plane_normal.dot(w) / n_dot_u
    return r


def line_point_collision(line_point: np.ndarray,
                         line_direction: np.ndarray,
                         single_point: np.ndarray):
    """
    Determine the factor of a line at which the line and a point collide.
    """
    # EQUATION: LINE_DIRECTION x PLANE_POINT = a
    a = np.dot(single_point, line_direction)
    plane_point_factor = (a - np.dot(line_point, line_direction)) / np.dot(
        line_direction, line_direction)

    return plane_point_factor


def plane_point_distance(plane_point: np.ndarray,
                         plane_normal: np.ndarray,
                         single_point: np.ndarray):
    """
    Determine the distance between a plane and a point.
    """
    return np.dot(single_point - plane_point, plane_normal)


def get_collision_slice_range(plane_points: list,
                              cuboid_point: np.ndarray,
                              cuboid_edge_direction: np.ndarray,
                              slice_distance: float,
                              n_slices: int):
    """
    Determine which slices of the target are necessary to interpolate the
    selected slices of the source.
    """
    # Ensure that the directional vector of the cuboid edges has unit length.
    # But don't overwrite the original array!
    cuboid_edge_direction = np.copy(cuboid_edge_direction)
    cuboid_edge_direction /= np.linalg.norm(cuboid_edge_direction)
    plane_point_distances = []

    for plane_point in plane_points:
        plane_point_distances.append(
            plane_point_distance(plane_point=cuboid_point,
                                 plane_normal=cuboid_edge_direction,
                                 single_point=plane_point) / slice_distance)
    min_idx = int(np.floor(max(0, min(plane_point_distances + [n_slices - 1]))))
    max_idx = int(np.ceil(min(n_slices - 1, max(plane_point_distances + [0]))))

    # Make sure the minimum and maximum indices are never the same, to avoid
    # interpolation errors in a 2D grid!
    if max_idx == min_idx:
        max_idx = np.min([max_idx + 1, n_slices - 1])

    return np.arange(min_idx, max_idx + 1)


def interpolate_orthogonal_slices(slice_arr: np.ndarray,
                                  new_positions: np.ndarray,
                                  old_positions: np.ndarray,
                                  method: str,
                                  fill_value: float):
    """
    Interpolation for the case of missing slices inbetween two orthogonal
    slices.
    """

    # The array where the interpolation results are collectively stored.
    interpolated_slice_arr = np.zeros(
        (new_positions.shape[0],) + slice_arr.shape[1:])

    z_direction = old_positions[1] - old_positions[0]
    z_direction /= np.linalg.norm(z_direction)

    for new_pos_idx, new_position in enumerate(new_positions):

        position_dists = np.linalg.norm(old_positions - new_position, axis=1)
        # Identify the original slice that is closest to the new slice position.
        closest_position_idx = np.argmin(position_dists)
        # In case the new slice matches perfectly with an already existing
        # slide, just insert this existing slide into the final output array.
        if position_dists[closest_position_idx] < EPS:
            interpolated_slice_arr[new_pos_idx] = slice_arr[
                closest_position_idx]
        # Otherwise, interpolate a new slice.
        else:
            z_travel_new = np.linalg.norm(new_position - old_positions[0])
            z_travel_closest = np.linalg.norm(
                old_positions[closest_position_idx] - old_positions[0])
            # In case the new slice is closer to the start slice than its closest
            # neighbor, select the slice before that as the lower bound. Otherwise,
            # select the slice after that as the upper bound.
            if z_travel_new < z_travel_closest:
                lower_bound_idx, upper_bound_idx = closest_position_idx - 1, closest_position_idx
            else:
                lower_bound_idx, upper_bound_idx = closest_position_idx, \
                    closest_position_idx + 1

            grid_arr = np.vstack([slice_arr[lower_bound_idx],
                                  slice_arr[upper_bound_idx]]) \
                .reshape((2,) + slice_arr.shape[1:])
            # Determine the relative position of the new slice in terms of its 2
            # bounding slices.
            relative_travel = np.linalg.norm(new_position - old_positions[
                lower_bound_idx]) / np.linalg.norm(
                old_positions[upper_bound_idx] - \
                old_positions[lower_bound_idx])

            # Choose the new slice to have the z-coordinate be the relative
            # position everywhere, and for the y- and x- coordinate stick to
            # the bounding slices.
            interpolation_positions = (np.ones(
                slice_arr.shape[1] * slice_arr.shape[
                    2]) * relative_travel,) + np.where(
                np.ones(slice_arr.shape[1:3]))

            warnings.simplefilter("ignore")
            interp_arr = RegularGridInterpolator(
                points=(np.arange(grid_arr.shape[0]),
                        np.arange(grid_arr.shape[1]),
                        np.arange(grid_arr.shape[2])),
                values=grid_arr,
                method=method,
                fill_value=fill_value,
                bounds_error=False)(interpolation_positions)

            interpolated_slice_arr[new_pos_idx] = interp_arr.reshape(
                grid_arr.shape[1:])

    return interpolated_slice_arr


"""
################################################################################
# CODE TO CREATE A NEW DICOM SLICE

# Insert dummy slices into the slice array to ensure the correct
    # topographical positioning of the slices.
    for missing_position_idx in missing_position_idxs:
        # Copy a bunch of necessary settings from the first slice.
        filler_slice = pydicom.dataset.Dataset()
        filler_slice.file_meta = pydicom.dataset.Dataset()
        filler_slice.file_meta.TransferSyntaxUID = slices[
            0].file_meta.TransferSyntaxUID
        filler_arr = np.copy(slices[0].pixel_array)
        filler_arr[:] = 0
        filler_slice.PixelData = filler_arr.tobytes()
        filler_slice.add_new(0x00280100, 'BitsAllocated',
                             slices[0]['BitsAllocated'].value)
        filler_slice.add_new(0x00280101, 'BitsStored',
                             slices[0]['BitsStored'].value)
        filler_slice.add_new(0x00280010, 'Rows',
                             slices[0]['Rows'].value)
        filler_slice.add_new(0x00280011, 'Columns',
                             slices[0]['Columns'].value)
        filler_slice.add_new(0x00280103, 'PixelRepresentation',
                             slices[0]['PixelRepresentation'].value)
        filler_slice.add_new(0x00280002, 'SamplesPerPixel',
                             slices[0]['SamplesPerPixel'].value)
        filler_slice.add_new(0x00280004, 'PhotometricInterpretation',
                             slices[0]['PhotometricInterpretation'].value)
        filler_slice.add_new(0x00200032, 'ImagePositionPatient',
                             start_position - \
                             missing_position_idx * \
                             z_direction * slice_thickness)
        filler_slice.add_new(0x00200037, 'ImageOrientationPatient',
                             reference_orientation)
        # Add the dummy slice at the correct position.
        slices = np.concatenate([slices[:missing_position_idx],
                                 [filler_slice],
                                 slices[missing_position_idx:]])
        # Update the number of eligible positions as well.
        n_modalities_per_position = np.concatenate([n_modalities_per_position[
                                                    :missing_position_idx],
                                                    [1],
                                                    n_modalities_per_position[
                                                    missing_position_idx:]])


# FUSION AND ROI EXTRACTION

def fuse_and_extract_ROIs(dicom_dirs,
                          roi_file: str,
                          output_dir: str,
                          voxel_size=None,
                          method: str = "linear",
                          fill_value=None,
                          file_format: str = "nifti"):
    Takes one or several DICOM directories and one roi file as input and
    returns the content of the ROI as a 4D array.
    :param dicom_dirs: The list of DICOM directories to be used for the
    interpolation, the order of the directories will be maintained in the
    output.
    :param roi_file: The Siemens Inveon CSV ROI file that is supposed to be
    extended to cuboid shape.
    :param output_dir: The directory where the fused files are to be saved.
    :param voxel_size: Voxel size of the interpolated 4D array. If None is
    provided, assume the dimensions of the ROI voxels.
    :param method: Method of interpolation. Possible options: linear, nearest.
    :param fill_value: The constant value used for voxels that are outside the
    grid of the regular grid interpolator. If None is provided, voxel values
    are extrapolated.
    :param file_format: The format for the final 4D array. Possilbe options:
    hdf, nifti, npy.
    # Make sure the function is also working when only a single DICOM
    # direcory is passed as a string.
    if isinstance(dicom_dirs, str):
        dicom_dirs = [dicom_dirs]

    # Ensure that the voxel size is iterable.
    # Row Spacing / Column Spacing / Slice Spacing
    if isinstance(voxel_size, int) | isinstance(voxel_size, float):
        voxel_size = [voxel_size] * 3

    base_roi_name = roi_file.split("\\")[-1].split("/")[-1].split(".")[0]

    roi_collection = handler_siemens.extract_cuboid_ROI(
        siemens_file=roi_file, output_dir=None, voxel_size=voxel_size)

    for roi_df, mask_arr, roi_voxel_size, dim_roi, directions_roi, roi_name in \
            roi_collection:
        header_df = None
        name = None

        interpolation_fused = None
        features_per_dir = []
        metadata_fused = []
        acquisition_times_fused = []
        weight_fused = []
        injected_dose_fused = []

        roi_arr = roi_df.to_numpy().reshape(tuple(dim_roi) + (3,))

        for dir_idx, dicom_dir in enumerate(dicom_dirs):
            interpolation_modality = None

            slices, n_modalities_per_position, acquisition_times, \
            weight, injected_dose, decay_factor = load_and_order_dicom_slices(
                directory_path=dicom_dir,
                discard_outliers=True)

            # These two variables are needed only after the
            # interpolation process is  finished.
            if dir_idx == 0:
                header_df = dicom_header_to_df(dataset=slices[0])
                name = slices[0].PatientName

            # Attribute named "Image Orientation (Patient)"
            image_orientation_patient = np.array(list(
                slices[0][0x0020, 0x0037].value))
            # Attribute named "Image Position (Patient)"
            image_position_patient_1 = np.array(
                list(slices[0][0x0020, 0x0032].value))
            image_position_patient_N = np.array(
                list(slices[-1][0x0020, 0x0032].value))
            # Attribute named "Pixel Spacing"
            pixel_spacing = list(slices[0][0x0028, 0x0030].value)
            N = len(n_modalities_per_position)

            direction_slice = (image_position_patient_N
                               - image_position_patient_1) / (N - 1.0)
            slice_thickness = np.linalg.norm(direction_slice)

            coordinates_modality = get_affine_voxel_coordinates(
                array_shape=(N,) + slices[0].pixel_array.shape,
                image_orientation_patient=image_orientation_patient,
                image_position_patient_1=image_position_patient_1,
                image_position_patient_N=image_position_patient_N,
                N=N,
                pixel_spacing=pixel_spacing,
                slice_idx=[0, 1] if N > 1 else [0])

            # Execute a basis change from basis B1 (basis of physical
            # coordinates) to basis B2 (basis of modality array).

            # Reshaping for easier indexing.
            coordinates_modality = coordinates_modality \
                .reshape((-1,) + slices[0].pixel_array.shape + (3,))

            e1_B2 = coordinates_modality[0, 0, 1] \
                    - coordinates_modality[0, 0, 0]
            e2_B2 = coordinates_modality[0, 1, 0] \
                    - coordinates_modality[0, 0, 0]
            if N > 1:
                e3_B2 = coordinates_modality[1, 0, 0] \
                        - coordinates_modality[0, 0, 0]
            else:
                e3_B2 = direction_slice

            A = np.array([e1_B2, e2_B2, e3_B2]).T
            basis_change_matrix = np.linalg.inv(A)

            desc = "Interpolating " + ("target" if (
                    dir_idx == 0) else F"source {dir_idx}")

            for roi_slice_idx in tqdm(range(dim_roi[0]), desc=desc):
                roi_coords = roi_arr[roi_slice_idx]

                plane_points = [roi_coords[0, 0],
                                roi_coords[-1, 0],
                                roi_coords[0, -1],
                                roi_coords[-1, -1]]

                roi_coords = roi_coords.reshape((-1, 3))

                collision_slice_idxs = get_collision_slice_range(
                    plane_points=plane_points,
                    cuboid_point=coordinates_modality[0, 0, 0],
                    cuboid_edge_direction=direction_slice,
                    slice_distance=slice_thickness,
                    n_slices=N)

                slices_arr = transform_ordered_slices_to_4D_array(
                    slices=slices,
                    n_modalities_per_position=n_modalities_per_position,
                    position_idxs=collision_slice_idxs)

                affinity_B2 = coordinates_modality[0, 0, 0] + min(
                    collision_slice_idxs) * direction_slice

                affine_coordinates_interpolated = np.dot(
                    basis_change_matrix,
                    (roi_coords - affinity_B2).T).T

                # Change from x-y-z to z-y-x to be compatible with
                # RegularGridInterpolator.
                affine_coordinates_interpolated[:, [0, 1, 2]] = \
                    affine_coordinates_interpolated[:, [2, 1, 0]]

                affine_coordinates_interpolated = affine_coordinates_interpolated \
                    .astype(np.float16)

                # THE ACTUAL INTERPOLATION PROCESS.
                with warnings.catch_warnings():
                    # Catch some weird errors that do not impact the
                    # interpolation results.
                    warnings.simplefilter("ignore")
                    interpolation_slice: np.ndarray = RegularGridInterpolator(
                        points=(np.arange(slices_arr.shape[0]),
                                np.arange(slices_arr.shape[1]),
                                np.arange(slices_arr.shape[2])),
                        values=slices_arr,
                        method=method,
                        fill_value=fill_value,
                        bounds_error=False)(affine_coordinates_interpolated)

                # Add the interpolated slices to the interpolation results of
                # the modality.
                if interpolation_modality is None:
                    interpolation_modality = interpolation_slice
                else:
                    interpolation_modality = np.append(interpolation_modality,
                                                       interpolation_slice,
                                                       axis=0)
            # Add each new modality as an extra column to the interpolation
            # results of all the directories.
            if interpolation_fused is None:
                interpolation_fused = interpolation_modality
            else:
                interpolation_fused = np.append(interpolation_fused,
                                                interpolation_modality,
                                                axis=1)

            if slices[0].Modality == 'PT':
                try:
                    frame_count: int = int(slices[0][0x0054, 0x0101].value)
                except KeyError:
                    # In case the PET data is not dynamic, just assume 1 as the
                    # number of frames.
                    frame_count = 1
                if frame_count > 1:
                    features = [F"PT_Frame_{i + 1}" for i in range(
                        frame_count)]
                else:
                    features = ["PT"]
                metadata = [str(slices[i]) for i in range(frame_count)]
            else:
                features = [slices[0].Modality]
                metadata = [str(slices[0])]

            features_per_dir += [features]
            metadata_fused += list(metadata)
            acquisition_times_fused += list(acquisition_times)
            weight_fused += list(acquisition_times)
            injected_dose_fused += list(acquisition_times)
            # Make sure to get rid of the big slice collection.
            del slices

        # Make sure that there are no duplicate feature names.
        features_per_dir = np.array(features_per_dir, dtype=object)
        features_fused = []
        for dir_idx, features in enumerate(features_per_dir):
            for feature in features:
                other_idxs = [i for i in range(len(features_per_dir)) if
                              i != dir_idx]
                if feature in features_per_dir[other_idxs].flatten():
                    features_fused.append(
                        feature + " - " + ("Target" if (dir_idx == 0)
                                           else F"Source_{dir_idx}"))
                else:
                    features_fused.append(feature)

        # Change the shape of the interpolation results to a 4D array,
        # for saving.
        interpolation_fused = interpolation_fused.reshape(
            tuple(dim_roi) + (len(features_fused),))

        output_path = os.path.join(output_dir, base_roi_name + "_" + roi_name)

        if file_format.lower() == "hdf":
            output_path += ".hdf"
            file_format = "HDF"

            with h5py.File(output_path, 'w') as out_file:
                dataset_voxel = out_file.create_dataset('voxel_array',
                                                        data=interpolation_fused)
                for attribute in set(header_df.columns):
                    item = header_df[attribute].iloc[0]
                    if isinstance(item, pydicom.sequence.Sequence):
                        item = dicom_header_to_df(item).to_dict()
                        item = list(item)
                    elif isinstance(item, pydicom.multival.MultiValue):
                        item = list(item)
                    elif isinstance(item, pd.Series):
                        new_item = []
                        for index, value in item.items():
                            if isinstance(value, bytes):
                                if len(value) > 100:
                                    continue
                                else:
                                    new_item.append(str(value))
                            else:
                                new_item.append(value)

                        item = new_item
                    elif isinstance(item, pydicom.uid.UID):
                        item = str(item)
                    elif isinstance(item, pydicom.valuerep.PersonName):
                        item = str(item)
                    elif isinstance(item, bytes):
                        item = str(item)

                    dataset_voxel.attrs[
                        attribute] = item if item is not None else ""

                dataset_voxel.attrs['name'] = str(name)
                dataset_voxel.attrs['features'] = list(features_fused)
                dataset_voxel.attrs['voxel_size'] = list(roi_voxel_size)
                dataset_voxel.attrs['acquisition_times'] = list(
                    acquisition_times_fused)
                dataset_voxel.attrs['metadata'] = list(metadata_fused)

                dataset_voxel.attrs['min_x_value'] = 0
                dataset_voxel.attrs['min_y_value'] = 0
                dataset_voxel.attrs['min_z_value'] = 0
        elif file_format.lower() == "nifti":
            output_path += ".nii.gz"
            file_format = "NIFTI"

            # Construct the affine, in the same fashion
            # get_affine_voxel_coordinates does it.
            dr, dc = roi_voxel_size[:2]
            F_1 = directions_roi[1] / np.linalg.norm(directions_roi[1])
            F_2 = directions_roi[0] / np.linalg.norm(directions_roi[0])

            column_A_1 = np.append(F_1 * dr, 0)
            column_A_2 = np.append(F_2 * dc, 0)
            column_A_3 = np.append((roi_arr[-1, 0, 0]
                                    - roi_arr[0, 0, 0]) / (
                                       max(roi_arr.shape[0] - 1, 1)), 0)
            column_A_4 = np.append(roi_arr[0, 0, 0], 1)
            A = np.stack([column_A_1, column_A_2, column_A_3,
                          column_A_4]).T.astype(np.float16)

            # Transform the 4D volume to comply with standard NIFTI shape.
            interpolation_fused = np.einsum("ijk...->kji...", np.flip(
                interpolation_fused, axis=1))

            nifti_file = nib.Nifti1Image(interpolation_fused, affine=A)
            nib.save(nifti_file, output_path)

        # Otherwise, save as a .npy file.
        else:
            output_path += ".npy"
            file_format = "NPY"

            np.save(arr=interpolation_fused.astype(float), file=output_path)

        nl = ",\n"
        print(F"Fused DICOM directories at\n"
              F"[{nl.join(dicom_dirs)}]\nto a {file_format} file at"
              F"\n{output_path}.")

    print("\nSaved all available ROIs. Extraction process finished.")
    return
"""

"""
# SOME DEBUG CODE FOR DICOM FUSION

if dir_idx == -1:
    collision_coords = get_affine_voxel_coordinates(
        array_shape=(N,) + slices[0].pixel_array.shape,
        image_orientation_patient=image_orientation_patient,
        image_position_patient_1=image_position_patient_1,
        image_position_patient_N=image_position_patient_N,
        N=N,
        pixel_spacing=pixel_spacing,
        slice_idx=collision_slice_idxs[0])
    print(N)
    print(image_position_patient_N -
          image_position_patient_1)
    print(direction_slice)
    print(slices[0].SliceThickness)
    print(pixel_spacing)
    print(np.linalg.norm(direction_slice))
    print(coordinates_modality[0, 0, 0])
    print(affinity_B2)
    print(collision_coords[0])

    print(slices_arr.shape, collision_slice_idxs)
    slices_arr_coords = np.vstack(
        np.where(np.ones(slices_arr.shape))).T
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    disp_idxs_interp = [i for i in
                        range(len(affine_coordinates_interpolated))
                        if i % 10 == 0]
    disp_idxs_mod = [i for i in range(len(slices_arr_coords))
                     if i % 10000 == 0]  # [:1000]
    ax.scatter(affine_coordinates_interpolated[disp_idxs_interp, 0],
               affine_coordinates_interpolated[disp_idxs_interp, 1],
               affine_coordinates_interpolated[disp_idxs_interp, 2],
               color="blue")
    ax.scatter(slices_arr_coords[disp_idxs_mod, 0],
               slices_arr_coords[disp_idxs_mod, 1],
               slices_arr_coords[disp_idxs_mod, 2],
               color="red", alpha=0.5)
    plt.show()
"""

"""
# OLD FUNCTION THAT IS NO LONGER NEEDED.
def load_dicom_directory(directory_path: str,
                         return_coordinates: bool = False,
                         dtype=np.float32):
    Loads in an entire directory of DICOM files and returns the values
    necessary to create a voxel array instance.
    :param directory_path: The path to the directory containing the DICOM
    files that are to be read in.
    :param return_coordinates: Decide whether or not to get the affine
    coordinates for each voxel of the 3D array.
    :param dtype: The data-type of the returned array.
    :return: The array, name, features voxel size,
    and metadata for the voxel array object.
    # TODO: Move all metadata acquisition code away from order_dicom_slices
    #  to this function.
    slices, n_modalities_per_position, acquisition_times, \
    weight, injected_dose = order_dicom_slices(
        directory_path=directory_path,
        discard_outliers=False)
    array: np.ndarray = transform_ordered_slices_to_4D_array(slices=slices,
                                                             n_modalities_per_position
                                                             =n_modalities_per_position)

    if slices[0].Modality == 'PT':
        try:
            frame_count: int = get_dicom_attribute(slice=slices[0],
                                                   address=[0x0054, 0x0101])
            frame_count: int = int(slices[0][0x0054, 0x0101].value)
        except KeyError:
            # In case the PET data is not dynamic, just assume 1 as the
            # number of frames.
            frame_count = 1
        features: List[str] = [F"Frame_{i + 1}" for i in range(
            frame_count)]
        metadata: List[str] = [str(slices[i]) for i in range(frame_count)]
    else:
        features: List[str] = [slices[0].Modality]
        metadata: List[str] = [str(slices[0])]

    name: str = slices[0].PatientName
    # Attribute named "Pixel Spacing"
    pixel_spacing: List[float] = list(slices[0][0x0028, 0x0030].value)
    # Attribute named "Slice Thickness" (0018,0050)
    slice_thickness: float = slices[0].SliceThickness
    voxel_size: List[float] = pixel_spacing[::-1] + [slice_thickness]

    if return_coordinates:
        # Attribute named "Image Orientation (Patient)"
        image_orientation_patient: np.ndarray = np.array(list(
            slices[0][0x0020, 0x0037].value))

        # Attribute named "Image Position (Patient)"
        image_position_patient_1: np.ndarray = np.array(
            list(slices[0][0x0020, 0x0032].value))

        image_position_patient_N: np.ndarray = np.array(
            list(slices[-1][0x0020, 0x0032].value))

        N: int = len(n_modalities_per_position)

        coordinates = get_affine_voxel_coordinates(
            array_shape=array.shape,
            image_orientation_patient=image_orientation_patient,
            image_position_patient_1=image_position_patient_1,
            image_position_patient_N=image_position_patient_N,
            pixel_spacing=pixel_spacing,
            N=N).reshape(array.shape[:-1] + (3,))
    else:
        coordinates = None

    return array.astype(dtype), name, features, voxel_size, \
           acquisition_times, weight, injected_dose, metadata, coordinates
"""
