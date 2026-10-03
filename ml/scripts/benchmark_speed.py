"""Benchmark inference speed of pretrained models on CIFAR-10.

Usage:
    uv run python scripts/benchmark_speed.py
"""

import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    # 1. Load CIFAR-10 (small dataset for baseline)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to ImageNet size for pretrained models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # We use a small subset (e.g., 500 images) just for speed testing
    subset = torch.utils.data.Subset(dataset, range(500))
    batch_size = 32
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Define baseline models to evaluate
    models_to_test = [
        "resnet50.a1_in1k",
        "convnext_tiny.in12k_ft_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "swin_tiny_patch4_window7_224.ms_in1k"
    ]

    results = []

    for model_name in models_to_test:
        print(f"\nLoading {model_name}...")
        try:
            model = timm.create_model(model_name, pretrained=True)
            model = model.to(device)
            model.eval()

            # Warmup
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)

            # Benchmark
            start_time = time.time()
            total_images = 0
            
            with torch.no_grad():
                for images, _ in tqdm(dataloader, desc=f"Benchmarking {model_name}"):
                    images = images.to(device)
                    _ = model(images)
                    total_images += images.size(0)
                    
            end_time = time.time()
            duration = end_time - start_time
            images_per_sec = total_images / duration
            
            # Estimate full evaluation budget (CIFAR-10 test set: 10k images, 10 corruptions, 5 severities = 500,000 passes)
            total_passes = 10000 * 10 * 5
            estimated_hours = (total_passes / images_per_sec) / 3600
            
            results.append({
                "model": model_name,
                "images_per_sec": images_per_sec,
                "estimated_eval_hours": estimated_hours
            })
            
            print(f"Throughput: {images_per_sec:.2f} images/sec")
            print(f"Estimated time for full eval (500k passes): {estimated_hours:.2f} hours")
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

    print("\n--- Benchmark Summary ---")
    print(f"{'Model':<45} | {'Images/sec':<12} | {'Est. Hours (Full Eval)'}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<45} | {res['images_per_sec']:<12.2f} | {res['estimated_eval_hours']:.2f}h")

if __name__ == "__main__":
    main()
