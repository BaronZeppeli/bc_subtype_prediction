import os.path
import warnings

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.filebasedimages import ImageFileError

from pydicom import dcmread
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm

"""


# CODE FOR SKULL STRIPPING

import nipype.pipeline.engine as pe
from nipype.interfaces.mipav.developer import (JistIntensityMp2rageMasking,
                                               MedicAlgorithmSPECTRE2010)

wf = pe.Workflow("skullstripping")

mask = pe.Node(JistIntensityMp2rageMasking(), name="masking")
folder_path = '/Users/filo/7t_trt/niftis/sub001/session_1/'
# mask.inputs.inSecond = folder_path + "MP2RAGE_INV2.nii.gz"
# mask.inputs.inQuantitative = folder_path + "MP2RAGE_UNI.nii.gz"
mask.inputs.inT1weighted = "C:/Users/rakrezp1/Desktop/FPGAL-PATIENT-2_vsize=1.nii.gz"
mask.inputs.outMasked = True
mask.inputs.outMasked2 = True
mask.inputs.outSignal = True
mask.inputs.outSignal2 = True

skullstrip = pe.Node(MedicAlgorithmSPECTRE2010(), name="skullstrip")
skullstrip.inputs.outStripped = True
skullstrip.inputs.xDefaultMem = 6000

wf.connect(mask, 'outMasked', skullstrip, 'inInput')
wf.run()
"""

DECIMALS = 4


def load_NIFTI(path: str, decode: bool = True):
    """
    Load a NIFTI file and return the contents of its volume and its affine
    matrix.
    """

    # Make sure the file in question is in fact a NIFTI file.
    if not is_nifti(path):
        raise ValueError(
            F"The file provided at {path} cannot be read in as a NIFTI file.")

    # Reading the NIFTI file.
    nifti_file = nib.load(path)
    nifti_arr_proxy = nifti_file.get_fdata()
    nifti_arr = np.copy(nifti_arr_proxy)
    nifti_affine = nifti_file.affine
    nifti_header = nifti_file.header.copy()

    # Decode, if wanted, the arrays into usable decoded ones.
    if decode:
        nifti_affine = decode_affine(affine=nifti_affine,
                                     encoded_array_shape=nifti_arr.shape)
        nifti_arr = decode_array(arr=nifti_arr)

    # MAKES URE TO DELETE ALL NIFTI IMAGE RELATED
    # OBJECTS, IN ORDER TO ALLOW THE ORIGINAL NIFTI FILE
    # TO BE OVERWRITTEN.
    nifti_file.uncache()
    del nifti_file
    del nifti_arr_proxy

    return nifti_arr, nifti_affine, nifti_header


def encode_array(arr: np.ndarray):
    """
    Convert an array, as used in the VoxelArray class, into the shape present in
    NIFTI files.
    """
    # Important to note: Flip the y-axis to account for the fact that
    # NIFTI stores with a flipped definition of left and right, as compared
    # to the radiological standard.
    encoded_arr = arr[:, ::-1, :, ...]
    # Transform the 4D volume to comply with standard NIFTI shape.
    encoded_arr = np.einsum("ijk...->kji...", encoded_arr)

    return encoded_arr


def decode_array(arr: np.ndarray):
    """
    Convert an array, as saved in a NIFTI file,
    into the shape as used in the class VoxelArray.
    """

    # Reverse the steps specified in encode_array to convert the array back
    # to its original VoxelArray-compatible shape.
    if arr.ndim > 3:
        transform_str = "kji...->ijk..."
    else:
        transform_str = "kji->ijk"
    decoded_arr = np.einsum(transform_str, arr)
    decoded_arr = decoded_arr[:, ::-1, :, ...]

    return decoded_arr


def encode_affine(affine: np.ndarray,
                  decoded_array_shape=None,
                  len_y=None):
    """
    Turn a sensible affine matrix of a nifti file used for computation into the
    nonsensical shape used for saving them.
    """
    if (decoded_array_shape is None) & (len_y is None):
        raise ValueError("Either the parameter encoded_array_shape or len_y "
                         "have to be not None.")
    # NOTE: The parameter decoded_array_shape is redundant and is only needed
    # for compatibility and convenience.
    if decoded_array_shape is not None:
        len_y = decoded_array_shape[1]
    # MAKE SURE TO COPY THE ARRAY, TO AVOID MUTABLE ERRORS!!!
    column_A_1, column_A_2, column_A_3, column_A_4 = np.copy(affine).T
    # For some reason, these components need to be negated.
    column_A_1[2] = -column_A_1[2]
    column_A_2[:2] = -column_A_2[:2]
    column_A_3[:2] = -column_A_3[:2]

    # For some reason, column comes before row.
    M = np.stack([column_A_2, column_A_1, column_A_3]).T
    # For some reason, the x and y coordinate need to be negated.
    flipped_position = np.copy(column_A_4[:3])
    flipped_position[:2] = -flipped_position[:2]
    abc = np.append(flipped_position - M[:-1].dot(
        [0, len_y - 1, 0]), 1)
    # IMPORTANT: For NIFTI, the first and second column need to be switched.
    encoded_affine = np.append(M, abc[..., np.newaxis], axis=1)

    # Make sure to erase -0.0.
    encoded_affine[np.abs(encoded_affine) < 1e-10] = 0

    return encoded_affine


def decode_affine(affine: np.ndarray,
                  encoded_array_shape=None,
                  decoded_array_shape=None,
                  len_y=None):
    """
    Turn a nonsensical affine matrix of a nifti file used for saving into the
    sensible shape used for computation.
    """
    if (encoded_array_shape is None) & (decoded_array_shape is None) & (
            len_y is None):
        raise ValueError("Either the parameter encoded_array_shape, "
                         "decoded_array_shape or len_y have to be not None.")
    # Encode the array shape, in case a decoded array shape was provided.
    if decoded_array_shape is not None:
        encoded_array_shape = get_encoded_array_shape(
            decoded_array_shape=decoded_array_shape)
    # NOTE: The parameter encoded_array_shape is redundant and is only needed
    # for compatibility and convenience.
    if encoded_array_shape is not None:
        len_y = encoded_array_shape[1]
    # For some reason, column comes before row.
    # MAKE SURE TO COPY THE ARRAY, TO AVOID MUTABLE ERRORS!!!
    column_A_2, column_A_1, column_A_3, _ = np.copy(affine).T
    # For some reason, these components need to be negated.
    column_A_1[2] = -column_A_1[2]
    column_A_2[:2] = -column_A_2[:2]
    column_A_3[:2] = -column_A_3[:2]

    # Recover the true ImagePositionPatient attribute from the weirdly negated
    # one  generated by the encoded affine matrix.
    image_position_patient = apply_affine(aff=affine,
                                          pts=[0, len_y - 1,
                                               0])
    # For some reason, the x and y coordinate need to be negated.
    image_position_patient[:2] = -image_position_patient[:2]
    column_A_4 = np.append(image_position_patient, 1)

    # Combine the individually recovered columns to the complete decoded affine.
    decoded_affine = np.stack(
        [column_A_1, column_A_2, column_A_3, column_A_4]).T

    # Make sure to erase -0.0.
    decoded_affine[np.abs(decoded_affine) < 1e-10] = 0

    return decoded_affine


def standardize_orientation(volume: np.ndarray,
                            affine: np.ndarray,
                            orientation: str,
                            decode: bool = False):
    """
    Function to ensure that a volume is aligned with the desired canonical
    scan orientation.
    """
    if decode:
        volume_shape = volume.shape[:3]
        volume = decode_array(arr=volume)
        affine = decode_affine(affine=affine, encoded_array_shape=volume_shape)

    image_orientation_patient = np.zeros(6)
    image_orientation_patient[:3] = affine[:-1, 1]
    image_orientation_patient[3:] = affine[:-1, 0]

    orientation_old, flip_rotate_queue = determine_slice_orientation(
        direction_cosine=image_orientation_patient,
        default_orientation=orientation)

    if orientation_old.lower() != orientation:
        warnings.warn(F"The scan was recorded in {orientation_old.lower()} "
                      F"orientation. The volume will be transformed to be "
                      F"aligned to {orientation} orientation.")
        affine = transform_affine(affine=affine,
                                  old_array_shape=volume.shape[:3],
                                  old_orientation=orientation_old,
                                  new_orientation=orientation,
                                  decode=False)
        if orientation_old.lower() == "coronal":
            volume = np.einsum('yzxf->zyxf', volume)
            volume = np.flip(volume, axis=(0, 1))
            # voxel_size = voxel_size[[2, 1, 0]]
        elif orientation_old.lower() == "sagittal":
            volume = np.einsum('xzyf->zyxf', volume)
            volume = np.flip(volume, axis=(0,))
            # voxel_size = voxel_size[[1, 2, 0]]
    for transform_task in flip_rotate_queue:
        if np.isreal(transform_task):
            volume = np.rot90(m=volume,
                              k=1,
                              axes=(1, 2))
        elif transform_task.lower() == "column":
            volume = np.flip(volume, axis=(1,))
            # Don't forget to negate the affine as well!
            affine[:, 0] = -affine[:, 0]
        elif transform_task.lower() == "row":
            volume = np.flip(volume, axis=(2,))
            # Don't forget to negate the affine as well!
            affine[:, 1] = -affine[:, 1]
        else:
            raise ValueError("Unknown orientation transformation "
                             f"operation {transform_task}")

    return volume, affine


def determine_slice_orientation(direction_cosine: np.ndarray,
                                default_orientation: str = "axial"):
    """
    Function to identify the orientation of a scan, given its direction
    cosine, e.g. encoded in the DICOM attribute ImageOrientationPatient.
    """
    ROTATION_TOL = 10.0 / 360 * (2 * np.pi)

    X_DIRECTION = [1, 0, 0]
    Y_DIRECTION = [0, 1, 0]
    Z_DIRECTION = [0, 0, -1]

    CANONICAL_DIRECTIONS = np.array([X_DIRECTION,
                                     Y_DIRECTION,
                                     Z_DIRECTION])

    row_direction = direction_cosine[:3]
    column_direction = direction_cosine[3:]
    img_directions = np.array([row_direction,
                               column_direction])

    matched_canonical = []
    flip_rotate_queue = []

    for img_idx, img_direction in enumerate(img_directions):
        for canonical_idx, canonical_direction in enumerate(
                CANONICAL_DIRECTIONS):
            angle = get_angle(v1=img_direction, v2=canonical_direction)
            if np.isclose(angle, 0, atol=ROTATION_TOL):
                matched_canonical += [canonical_idx]
                break
            elif np.isclose(angle, np.pi, atol=ROTATION_TOL):
                matched_canonical += [canonical_idx]
                flip_rotate_queue += ["row" if img_idx == 0 else "column"]
                break

    matched_canonical = np.array(matched_canonical)
    if len(matched_canonical) == 2:
        if np.allclose(np.sort(matched_canonical), [0, 1]):
            orientation = "axial"
            # In case the row- and column-direction was switched, rotate and
            # flip the images.
            if matched_canonical[0] == 1:
                flip_rotate_queue += ["column", 90]
        elif np.allclose(np.sort(matched_canonical), [0, 2]):
            orientation = "coronal"
            # In the case of row=x and column=y, a column flip is necessary
            # to get to the coronal window.
            if matched_canonical[0] == 2:
                flip_rotate_queue += ["column", 90]
        elif np.allclose(np.sort(matched_canonical), [1, 2]):
            orientation = "sagittal"
            # The same for sagittal as for coronal orientation.
            if matched_canonical[0] == 2:
                flip_rotate_queue += ["column", 90]
        else:
            orientation = default_orientation
            flip_rotate_queue = []
    else:
        orientation = default_orientation
        flip_rotate_queue = []

    return orientation, flip_rotate_queue


def get_angle(v1: np.ndarray, v2: np.ndarray):
    """
    Helper function to calculate the angle between 2 vectors.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def transform_affine(affine: np.ndarray,
                     old_array_shape: tuple,
                     old_orientation: str,
                     new_orientation: str,
                     decode: bool = False):
    """
    Transforms the affine of a volume according to the orientation
    transformation being executed.
    """

    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"

    old_orientation = old_orientation.lower()
    new_orientation = new_orientation.lower()

    if (old_orientation not in [AXIAL, CORONAL, SAGITTAL]) | (
            new_orientation not in [AXIAL, CORONAL, SAGITTAL]):
        raise ValueError(F"The orientations need to be one of "
                         F"{AXIAL, CORONAL, SAGITTAL}.")

    if decode:
        affine = decode_affine(affine=affine,
                               encoded_array_shape=old_array_shape)
        old_array_shape = get_decoded_array_shape(
            encoded_array_shape=old_array_shape)

    # Figure out how the columns of the affine have to be swapped and which
    # columns have to be negated.
    if old_orientation == AXIAL:
        if new_orientation == AXIAL:
            new_image_position = [0, 0, 0]
            F_order = [0, 1, 2]
            F_negate = []
        elif new_orientation == CORONAL:
            new_image_position = [0, 0, old_array_shape[0] - 1]
            F_order = [2, 1, 0]
            F_negate = [2]
        else:
            new_image_position = [0, 0, old_array_shape[0] - 1]
            F_order = [2, 0, 1]
            F_negate = [2]
    elif old_orientation == CORONAL:
        if new_orientation == AXIAL:
            new_image_position = [old_array_shape[1] - 1, 0, 0]
            F_order = [2, 1, 0]
            F_negate = [2]
        elif new_orientation == CORONAL:
            new_image_position = [0, 0, 0]
            F_order = [0, 1, 2]
            F_negate = []
        else:
            new_image_position = [0, 0, 0]
            F_order = [0, 2, 1]
            F_negate = []
    else:
        if new_orientation == AXIAL:
            new_image_position = [old_array_shape[1] - 1, 0, 0]
            F_order = [1, 2, 0]
            F_negate = [2]
        elif new_orientation == CORONAL:
            new_image_position = [0, 0, 0]
            F_order = [0, 2, 1]
            F_negate = []
        else:
            new_image_position = [0, 0, 0]
            F_order = [0, 1, 2]
            F_negate = []

    # Calculate the new ImagePositionPatient.
    new_image_position = apply_affine(aff=affine, pts=new_image_position)

    # Insert the changes into a NEW array.
    new_affine = np.copy(affine)
    new_affine[:, [0, 1, 2]] = new_affine[:, F_order]
    new_affine[:, F_negate] = -new_affine[:, F_negate]
    new_affine[:3, 3] = new_image_position

    return new_affine


def get_voxel_size(affine: np.ndarray,
                   decode: bool = False,
                   array_shape=None,
                   len_y=None):
    """
    Helper function to extract the voxel size from the affine matrix of a
    NIFTI file.
    """
    if decode:
        affine = decode_affine(affine=affine,
                               encoded_array_shape=array_shape,
                               len_y=len_y)
    column_A_1, column_A_2, column_A_3, column_A_4 = affine.T
    row_thickness = np.linalg.norm(column_A_1[:3])
    column_thickness = np.linalg.norm(column_A_2[:3])
    slice_thickness = np.linalg.norm(column_A_3[:3])

    voxel_size = np.round([row_thickness, column_thickness, slice_thickness],
                          DECIMALS)

    return voxel_size


def get_affine(image_orientation_patient: np.ndarray,
               image_position_patient: np.ndarray,
               pixel_spacing: np.ndarray,
               slice_thickness: float,
               decoded_array_shape: np.ndarray = None,
               encode: bool = False):
    """
    Determine the affine of a series by a set of position attributes.
    """
    dr, dc = pixel_spacing
    F_1 = image_orientation_patient[3:] / np.linalg.norm(
        image_orientation_patient[3:])
    F_2 = image_orientation_patient[:3] / np.linalg.norm(
        image_orientation_patient[:3])
    # The slice direction.
    F_3 = -np.cross(F_1, F_2)
    F_3 /= np.linalg.norm(F_3)
    # Flip the slice direction around, in case the leading direction is
    # negative. This is valid, as the volume will always be transformed such
    # that the slice direction is directed towards increasing values.
    if F_3[np.argmax(np.abs(F_3))] < 0:
        F_3 = -F_3

    ds = slice_thickness

    column_A_1 = np.append(F_1 * dr, 0)
    column_A_2 = np.append(F_2 * dc, 0)
    column_A_3 = np.append(F_3 * ds, 0)
    column_A_4 = np.append(image_position_patient, 1)
    affine = np.stack([column_A_1, column_A_2, column_A_3, column_A_4]).T

    # Make sure to erase -0.0.
    affine[np.abs(affine) < 1e-10] = 0

    if encode:
        # Simply get the slice count in y-direction by looking at the number
        # of rows in the slice pixel array.
        len_y = decoded_array_shape[0]
        affine = encode_affine(affine=affine, len_y=len_y)

    return affine


def get_affine_from_DICOM(slice,
                          encode: bool = False):
    """
    Reconstruct the affine matrix of a NIFTI file from a single DICOM slice.
    """
    # In case a string was passed, load the DICOM slice at the location
    # specified in the string.
    if isinstance(slice, str):
        slice = dcmread(slice)

    image_orientation_patient = slice.ImageOrientationPatient
    image_position_patient = slice.ImagePositionPatient
    pixel_spacing = slice.PixelSpacing
    slice_thickness = slice.SliceThickness
    decoded_array_shape = slice.pixel_array.shape[0]

    affine = get_affine(image_orientation_patient=image_orientation_patient,
                        image_position_patient=image_position_patient,
                        pixel_spacing=pixel_spacing,
                        slice_thickness=slice_thickness,
                        decoded_array_shape=decoded_array_shape,
                        encode=encode)

    return affine


def get_DICOM_attrs_from_affine(affine: np.ndarray,
                                n_slices=None,
                                decode: bool = False,
                                encoded_array_shape=None,
                                len_y=None):
    """
    Function to get extract all the metadata information from a NIFTI affine.
    """
    if decode:
        affine = decode_affine(affine=affine,
                               encoded_array_shape=encoded_array_shape,
                               len_y=len_y)
    dim_y, dim_x, dim_z, _ = np.linalg.norm(affine, axis=0)
    pixel_spacing = np.array([dim_y, dim_x])
    slice_thickness = dim_z
    image_orientation_patient = np.zeros(6)
    image_orientation_patient[:3] = affine[:-1, 1]
    image_orientation_patient[3:] = affine[:-1, 0]
    # Make sure the direction cosines have unit length!
    image_orientation_patient[:3] /= np.linalg.norm(
        image_orientation_patient[:3])
    image_orientation_patient[3:] /= np.linalg.norm(
        image_orientation_patient[3:])
    image_position_patient_1 = affine[:-1, 3]
    if n_slices is not None:
        image_position_patient_N = image_position_patient_1 \
                                   + affine[:-1, 2] * n_slices
    else:
        image_position_patient_N = None

    return pixel_spacing, slice_thickness, image_orientation_patient, \
        image_position_patient_1, image_position_patient_N


def update_affine_voxel_size(affine: np.ndarray,
                             voxel_size: np.ndarray,
                             encoded: bool = False):
    """
    Update the affine information, in case the voxel size of the volume was
    changed.
    """
    # Make sure to consider the nonsensical switch of rows and columns in the
    # case of an encoded affine.
    if encoded & (affine is not None):
        affine[:, [0, 1]] = affine[:, [1, 0]]
    # Enables the user to not pass an affine, which assumes the canonical
    # affine for the rest of the computation.
    elif affine is None:
        affine = np.identity(4)
        # Don't forget to switch the first and second column, as the
        # canonical affine is NOT the identity matrix!!!!
        affine = affine[:, [1, 0, 2, 3]]



    # Row size.
    column_A_1 = affine[:, 0]
    column_A_1 = column_A_1 / np.linalg.norm(column_A_1) * voxel_size[0]

    # Column size.
    column_A_2 = affine[:, 1]
    column_A_2 = column_A_2 / np.linalg.norm(column_A_2) * voxel_size[1]

    # Slice size.
    column_A_3 = affine[:, 2]
    column_A_3 = column_A_3 / np.linalg.norm(column_A_3) * voxel_size[2]

    # Image position patient remains unchanged.
    column_A_4 = affine[:, 3]

    affine = np.stack([column_A_1, column_A_2, column_A_3, column_A_4]).T

    # Don't forget to switch back the columns of the encoded array after
    # scaling them!
    if encoded:
        affine[:, [0, 1]] = affine[:, [1, 0]]

    return affine


def get_encoded_array_shape(decoded_array_shape: tuple):
    """
    Get the shape an array would have were it to be
    encoded for storing it as a NIFTI.
    """
    encoded_array_shape = (decoded_array_shape[2],
                           decoded_array_shape[1],
                           decoded_array_shape[0]) + decoded_array_shape[3:]

    return encoded_array_shape


def get_decoded_array_shape(encoded_array_shape: tuple):
    """
    Get the shape an array that is encoded for NIFTI storage if it were decoded.
    """
    return get_encoded_array_shape(decoded_array_shape=encoded_array_shape)


def scale_volume(arr: np.ndarray,
                 new_voxel_size=None,
                 new_affine=None,
                 old_voxel_size=None,
                 old_affine=None,
                 method: str = "nearest",
                 fill_value: int = 0,
                 decode: bool = False):
    """
    Scale a 4D volume and its corresponding affine to a new specified voxel
    resolution.
    """
    # The interpolation object.
    rgi = RegularGridInterpolator(points=(np.arange(arr.shape[0]),
                                          np.arange(arr.shape[1]),
                                          np.arange(arr.shape[2])),
                                  values=arr,
                                  method=method,
                                  fill_value=fill_value,
                                  bounds_error=False)
    if new_affine is not None:
        if decode:
            new_affine = decode_affine(affine=new_affine,
                                       encoded_array_shape=arr.shape)
            arr = decode_array(arr=arr)

        new_voxel_size = get_voxel_size(affine=new_affine, decode=False)
    elif new_voxel_size is None:
        raise ValueError("Either new_voxel_size or new_affine have to be not "
                         "None.")
    if old_affine is not None:
        if decode:
            old_affine = decode_affine(affine=old_affine,
                                       encoded_array_shape=arr.shape)
            arr = decode_array(arr=arr)

        old_voxel_size = get_voxel_size(affine=old_affine, decode=False)
    elif old_voxel_size is None:
        raise ValueError("Either old_voxel_size or old_affine have to be not "
                         "None.")

    # Small value to enable the inclusion of the endpoint of arange to be in
    # the list of interpolated coordinates.
    eps = 1e-10

    # NOTE: This assumes the ordering of dimensions of column-row-slice and
    # the array to be DECODED.
    coords_x = np.arange(0, arr.shape[2] - 1 + eps, new_voxel_size[
        1] / old_voxel_size[1])
    coords_y = np.arange(0, arr.shape[1] - 1 + eps, new_voxel_size[
        0] / old_voxel_size[0])
    coords_z = np.arange(0, arr.shape[0] - 1 + eps, new_voxel_size[
        2] / old_voxel_size[2])

    # The new shape of the array.
    # Make sure the resulting array has the same dimensionality as the
    # original array.
    shape_interpolated = (len(coords_z), len(coords_y), len(coords_x)) \
                         + arr.shape[3:]

    arr_scaled = np.zeros(shape_interpolated)

    # The actual interpolation step.
    for slice_idx, coord_z in enumerate(tqdm(coords_z, desc="Scaling volume")):
        # NOTE: This assumes the ordering of dimensions of row-column-slice and
        # the array to be in classical NIFTI shape.
        zz, yy, xx = np.meshgrid([coord_z], coords_y, coords_x)
        coords_interpolated = np.array(
            [zz.flatten(), yy.flatten(), xx.flatten()]).T

        # Interpolate a single slice.
        slice_interpolated = rgi(coords_interpolated)
        # Update the final output.
        arr_scaled[slice_idx] = slice_interpolated.reshape(
            shape_interpolated[1:])

    if old_affine is None:
        old_affine = np.identity(4)

    # Update the affine to reflect the new voxel size.
    affine_scaled = update_affine_voxel_size(affine=old_affine,
                                             voxel_size=new_voxel_size,
                                             encoded=False)

    # Return the updated volume and affine matrix.
    return arr_scaled, affine_scaled


def is_nifti(path: str):
    """
    Determine whether a specified path leads to a NIFTI file.
    """
    # A directory can never be a NIFTI file.
    if os.path.isdir(path):
        return False

    # Try loading the specified path as a NIFTI file. If it fails, the file
    # is of no compatible NIFTI format.
    try:
        nifti_file = nib.load(path)
        return True
    except ImageFileError:
        return False
