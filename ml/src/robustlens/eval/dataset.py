"""Dataset loading and mapping for ImageNet subsets."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

# Map Imagenette 10 classes to ImageNet-1K indices
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench, Tinca tinca
    1: 217,  # English springer, English springer spaniel
    2: 482,  # cassette player
    3: 491,  # chain saw, chainsaw
    4: 405,  # church, church building
    5: 513,  # French horn, horn
    6: 569,  # garbage truck, dustcart
    7: 571,  # gas pump, gasoline pump, petrol pump, island dispenser
    8: 574,  # golf ball
    9: 701,  # parachute, chute
}

class ImagenetteDataset(Dataset):
    """Imagenette dataset wrapped for PyTorch with ImageNet labels."""
    
    def __init__(self, split="validation", transform=None, size="320px"):
        self.dataset = load_dataset("frgfm/imagenette", size, split=split)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Original imagenette label (0-9)
        orig_label = item["label"]
        
        # Map to ImageNet 1k label
        imagenet_label = IMAGENETTE_TO_IMAGENET[orig_label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, imagenet_label, orig_label

def get_imagenette_loader(batch_size=32, transform=None, num_workers=2, subset_size=None):
    """Get a DataLoader for Imagenette."""
    dataset = ImagenetteDataset(split="validation", transform=transform)
    
    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
        
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
