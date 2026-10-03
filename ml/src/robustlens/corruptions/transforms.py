"""Image corruptions using standard torchvision and PIL."""

import torch
import torchvision.transforms.functional as F
from PIL import Image
import io

class GaussianNoise:
    """Add Gaussian noise to image tensor."""
    def __init__(self, severity=1):
        self.severity = severity
        # Severities: standard deviations
        self.stds = [0.04, 0.08, 0.12, 0.18, 0.26]
        
    def __call__(self, img):
        std = self.stds[self.severity - 1]
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)

class GaussianBlur:
    """Apply Gaussian blur."""
    def __init__(self, severity=1):
        self.severity = severity
        # Severities: kernel sizes and sigmas
        self.params = [(3, 1.0), (5, 2.0), (7, 3.0), (9, 4.0), (11, 5.0)]
        
    def __call__(self, img):
        kernel_size, sigma = self.params[self.severity - 1]
        return F.gaussian_blur(img, [kernel_size, kernel_size], [sigma, sigma])

class JpegCompression:
    """Apply JPEG compression via PIL."""
    def __init__(self, severity=1):
        self.severity = severity
        # Quality drops
        self.qualities = [80, 65, 50, 35, 20]
        
    def __call__(self, img):
        quality = self.qualities[self.severity - 1]
        
        # Convert tensor to PIL
        pil_img = F.to_pil_image(img)
        
        # Compress
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        # Load back and convert to tensor
        compressed_pil = Image.open(buffer)
        return F.to_tensor(compressed_pil)

class Brightness:
    """Change image brightness."""
    def __init__(self, severity=1):
        self.severity = severity
        # Brightness factors
        self.factors = [1.5, 2.0, 2.5, 3.0, 3.5]
        
    def __call__(self, img):
        factor = self.factors[self.severity - 1]
        return F.adjust_brightness(img, factor)

class Contrast:
    """Change image contrast."""
    def __init__(self, severity=1):
        self.severity = severity
        # Contrast factors
        self.factors = [0.6, 0.4, 0.3, 0.2, 0.1]
        
    def __call__(self, img):
        factor = self.factors[self.severity - 1]
        return F.adjust_contrast(img, factor)

# Map of available corruptions
CORRUPTIONS = {
    "gaussian_noise": GaussianNoise,
    "gaussian_blur": GaussianBlur,
    "jpeg_compression": JpegCompression,
    "brightness": Brightness,
    "contrast": Contrast
}

def get_corruption_transform(corruption_name, severity):
    """Get the corruption callable."""
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be between 1 and 5")
        
    return CORRUPTIONS[corruption_name](severity)
