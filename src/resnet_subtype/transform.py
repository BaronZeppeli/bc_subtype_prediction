import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import v2
import torch.nn.functional as F2
from scipy.ndimage import rotate
from PIL import Image


class ResizeWithPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Resize the image while maintaining the aspect ratio
        c, h, w = img.size()

        # Determine padding needed to make the image square
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = v2.Resize(size=(new_h, new_w))(img)

        # Pad the resized image to reach the target size
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        padding = (pad_w, pad_h, self.target_size - new_w - pad_w,
                   self.target_size - new_h - pad_h)
        img = F.pad(img, padding)

        assert img.size() == (c, self.target_size, self.target_size), ("Image "
                                                                       "padding failed.")

        return img


class ResizeWithConstantRatio:
    def __init__(self,
                 target_size:int,
                 interpolation_mode:str=None):
        assert isinstance(target_size, int), ("Only integers are supported for "
                                              "target_size!")
        self.target_size = target_size
        # Optional use of an interpolation mode. If none is provided, linear
        # interpolation will be used.
        self.interpolation_mode = interpolation_mode

        return

    def __call__(self, img):
        # Resize the image while maintaining the aspect ratio
        if img.dim() == 3:
            c, h, w = img.size()
            d = 1
        elif img.dim() == 4:
            c, h, d, w = img.size()
        else:
            raise ValueError("Transformation is only supported for either 3D "
                             "or 4D tensors!")
        dims = [h, d, w] if img.dim() == 4 else [h, w]
        min_dim = np.min(dims)
        new_size = [int(dim / min_dim * self.target_size) for dim in dims]

        if self.interpolation_mode is None:
            mode = "trilinear" if img.dim() == 4 else "bilinear"
        else:
            mode = self.interpolation_mode

        img = F2.interpolate(img.unsqueeze(0),
                             size=new_size,
                             mode=mode)[0]

        assert min(img.size()[1:]) == self.target_size, "Image resizing failed."

        return img


import torch
import random


class RandomCrop2D3D:
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size (D, H, W) for depth, height, and width.
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): 3D volume tensor of shape (C, D, H, W) or (D, H, W).

        Returns:
            torch.Tensor: Randomly cropped 3D volume tensor.
        """
        if isinstance(self.output_size, int):
            new_size = [self.output_size] * (sample.dim() - 1)
        else:
            new_size = self.output_size

        # If the input is in shape (C, H, W), we don't have a depth dimension.
        if sample.dim() == 3:
            C, H, W = sample.shape
            D = 1
        elif sample.dim() == 4:
            C, H, D, W = sample.shape
        else:
            raise ValueError("Input tensor must be 2D or 3D.")

        old_size = [H, W] if sample.dim() == 3 else [H, D, W]
        # Ensure the output size is not larger than the input dimensions.
        assert np.all(np.array(
            old_size) >= new_size), ("Output size must be smaller than "
                                     "or equal to input size.")

        # Randomly select the start indices for depth, height, and width
        start_idxs = [random.randint(0, old_dim - new_dim) for old_dim, new_dim
                      in zip(old_size, new_size)]

        if sample.dim() == 3:
            new_sample = sample[:,
                         start_idxs[0]:start_idxs[0] + new_size[0],
                         start_idxs[1]:start_idxs[1] + new_size[1]]
        else:
            new_sample = sample[:,
                         start_idxs[0]:start_idxs[0] + new_size[0],
                         start_idxs[1]:start_idxs[1] + new_size[1],
                         start_idxs[2]:start_idxs[2] + new_size[2]]

        return new_sample.detach().clone()


class RandomFlip2D3D:
    def __init__(self):
        """Initialize the RandomFlip3D transform."""
        pass

    def __call__(self, tensor):
        """
        Apply random flips to the 3D tensor.

        Args:
            tensor (torch.Tensor): A 3D tensor of shape (D, H, W).

        Returns:
            torch.Tensor: The randomly flipped tensor.
        """
        # Check if the tensor is 2D or 3D.
        assert (tensor.dim() == 3) | (tensor.dim() == 4), (
            "Input tensor must be "
            "2D or 3D.")

        # Randomly decide to flip along each dimension.
        flip_decision = [random.choice([True, False]) for _ in
                         range(tensor.dim() - 1)]

        # Create a list of dimensions to flip.
        flip_dims = [i + 1 for i in range(tensor.dim() - 1) if flip_decision[i]]

        # Flip the tensor if needed.
        if flip_dims:
            tensor = torch.flip(tensor, dims=flip_dims)

        return tensor


class RandomRotation3D:
    def __init__(self):
        """Initialize the RandomRotate3D transform."""
        pass

    def __call__(self, tensor):
        """
        Apply a random rotation to the 3D tensor.

        Args:
            tensor (torch.Tensor): A 3D tensor of shape (D, H, W).

        Returns:
            torch.Tensor: The rotated tensor.
        """
        if tensor.ndimension() != 4:
            raise ValueError(
                "Input tensor must be a 3D tensor with shape (C, H, D, W).")

        # Randomly choose a rotation angle and axis.
        rotate_decision = [random.choice([True, False]) for _ in
                           range(tensor.dim() - 1)]
        rotate_axes = [i + 1 for i in range(tensor.dim() - 1) if
                       rotate_decision[i]]

        repeats = [1, 2, 3]  # Corresponding to 90, 180, and 270 degrees.
        repeat_axes = [random.choice(repeats) for _ in rotate_axes]

        # Rotate tensor.
        rotated_tensor = tensor
        for rotate_axis, repeat_axis in zip(rotate_axes, repeat_axes):
            rotated_tensor = tensor.rot90(k=repeat_axis,
                                          dims=self._get_rotation_dims(
                                              rotate_axis))

        return rotated_tensor

    def _get_rotation_dims(self, axis):
        """Get dimensions for rotation based on the specified axis."""
        if axis == 1:  # Rotate around depth.
            return (2, 3)  # Rotate height and width.
        elif axis == 2:  # Rotate around height.
            return (1, 3)  # Rotate depth and width.
        elif axis == 3:  # Rotate around width.
            return (1, 2)  # Rotate depth and height.
        else:
            raise ValueError("Axis must be 1, 2, or 3.")
