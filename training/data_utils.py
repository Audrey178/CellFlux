import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from torchvision.transforms import InterpolationMode

try:
    import pydicom
except ImportError:  # pragma: no cover - dependency is declared in environment.yml
    pydicom = None


def _read_dicom_pixels(path):
    if pydicom is None:
        raise ImportError(
            "pydicom is required for the ctpet dataset. Install it from environment.yml."
        )
    return pydicom.dcmread(path).pixel_array.astype(np.float32)


def min_max_normalize(image, eps=1e-6):
    img_min = torch.min(image)
    img_max = torch.max(image)
    scale = torch.clamp(img_max - img_min, min=eps)
    return (image - img_min) / scale


class CTPETTransform:
    """Preprocess paired CT/PET slices with shared resize and augmentation."""

    def __init__(self, image_size=256, augment=False, normalize=True):
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

    def _resize(self, image):
        return TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def __call__(self, ct_image, pet_image):
        ct_image = min_max_normalize(ct_image)
        pet_image = min_max_normalize(pet_image)

        ct_image = self._resize(ct_image)
        pet_image = self._resize(pet_image)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                ct_image = TF.hflip(ct_image)
                pet_image = TF.hflip(pet_image)
            if torch.rand(1).item() < 0.5:
                ct_image = TF.vflip(ct_image)
                pet_image = TF.vflip(pet_image)

        if self.normalize:
            ct_image = ct_image * 2.0 - 1.0
            pet_image = pet_image * 2.0 - 1.0

        return ct_image, pet_image
    
def read_files_dicom(file_name, ct_dir, pet_dir, transform , dataset_name):
    """
    Read and process DICOM images for CT and PET scans.
    
    Args:
        file_name (str): Name of the file containing the sample information.
        ct_dir (str): Directory containing CT scan images.
        pet_dir (str): Directory containing PET scan images.
        transform (callable): Transformation to apply to the images.
        dataset_name (str): Name of the dataset.
    Returns:
        dict: Dictionary containing processed CT and PET images, and the file name.
    """
    ct_path = Path(ct_dir) / file_name
    pet_path = Path(pet_dir) / file_name
    
    ct_image = _read_dicom_pixels(ct_path)
    pet_image = _read_dicom_pixels(pet_path)
    
    ct_image = torch.from_numpy(ct_image).unsqueeze(0)  # Add channel dimension
    pet_image = torch.from_numpy(pet_image).unsqueeze(0)  # Add channel dimension
    
    if transform:
        ct_image, pet_image = transform(ct_image, pet_image)

    return {
        "X": (ct_image, pet_image),
        "file_names": (file_name, file_name),
        "idx_ct": 0,
        "idx_pet": 0,
    }


def convert_6ch_to_3ch(images):
    """
    Convert 6-channel images to 3-channel RGB composite images.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 6, H, W), values in range [0, 1].
        
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    # Define the weights for each channel in RGB
    # Channel 1-6 mapped to specific colors
    weights = torch.tensor([
        [0, 0, 1],   # Channel 1 -> Blue
        [0, 1, 0],   # Channel 2 -> Green
        [1, 0, 0],   # Channel 3 -> Red
        [0, 0.5, 0.5],  # Channel 4 -> Cyan (lower intensity)
        [0.5, 0, 0.5],  # Channel 5 -> Magenta (lower intensity)
        [0.5, 0.5, 0],  # Channel 6 -> Yellow (lower intensity)
    ], dtype=images.dtype, device=images.device)
    
    # Perform matrix multiplication to combine channels
    # Shape transformation: (batch_size, 6, H, W) -> (batch_size, 3, H, W)
    images_rgb = torch.einsum('bchw,cn->bnhw', images, weights)
    
    # Clip the result to ensure it's within [0, 1]
    images_rgb = torch.clamp(images_rgb, -1, 1)
    
    return images_rgb

def convert_5ch_to_3ch(images):
    """
    Convert 5-channel images to 3-channel RGB composite images.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 5, H, W), values in range [0, 1] or [-1, 1].
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    images_rgb = images[:, :3, :, :]
    return images_rgb
