import os
import json
import urllib.request
from PIL import Image, ImageFilter, ImageEnhance
import io
import torch
import timm
from torchvision import transforms
import numpy as np

def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = urllib.request.urlopen(url)
        classes = [line.decode("utf-8").strip() for line in response.readlines()]
        return classes
    except Exception as e:
        print(f"Error loading imagenet classes: {e}")
        return [f"class_{i}" for i in range(1000)]

classes = get_imagenet_classes()

import requests

IMAGES = [
    {"url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG", "label": 0},
    {"url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG", "label": 1},
    {"url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01484850_great_white_shark.JPEG", "label": 2},
    {"url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01491361_tiger_shark.JPEG", "label": 3},
    {"url": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01494475_hammerhead.JPEG", "label": 4},
]

def download_and_crop(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    img = transforms.Resize(256)(img)
    img = transforms.CenterCrop(224)(img)
    return img


def add_gaussian_noise(img, severity):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    img = np.array(img) / 255.
    img = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1) * 255
    return Image.fromarray(np.uint8(img))

def add_defocus_blur(img, severity):
    c = [1, 2, 3, 4, 5][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=c))

def add_jpeg_compression(img, severity):
    c = [80, 65, 50, 35, 20][severity - 1]
    output = io.BytesIO()
    img.save(output, 'JPEG', quality=c)
    return Image.open(output)



def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_dir = os.path.join(base_dir, 'apps', 'web', 'public', 'examples')
    res_dir = os.path.join(base_dir, 'results', 'demo')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    models = {
        "resnet50.a1_in1k": timm.create_model("resnet50.a1_in1k", pretrained=True).eval(),
        "convnext_tiny.in12k_ft_in1k": timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True).eval(),
        "vit_base_patch16_224.augreg_in21k_ft_in1k": timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval(),
        "swin_tiny_patch4_window7_224.ms_in1k": timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True).eval(),
    }
    
    data_configs = {name: timm.data.resolve_data_config({}, model=m) for name, m in models.items()}
    val_transforms = {name: timm.data.create_transform(**cfg, is_training=False) for name, cfg in data_configs.items()}
    
    results = []
    
    for i, item in enumerate(IMAGES):
        print(f"Processing image {i}")
        img = download_and_crop(item['url'])
        true_label = item['label']
        true_class_name = classes[true_label]
        
        img.save(os.path.join(img_dir, f"img_{i}_clean.jpg"))
        
        corruptions = {
            'clean': {1: img},
            'gaussian_noise': {s: add_gaussian_noise(img, s) for s in range(1, 6)},
            'defocus_blur': {s: add_defocus_blur(img, s) for s in range(1, 6)},
            'jpeg_compression': {s: add_jpeg_compression(img, s) for s in range(1, 6)}
        }
        
        for corr, sevs in corruptions.items():
            for s, c_img in sevs.items():
                if corr != 'clean':
                    c_img.save(os.path.join(img_dir, f"img_{i}_{corr}_s{s}.jpg"))
                
                for model_name, model in models.items():
                    input_tensor = val_transforms[model_name](c_img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)[0]
                        pred_idx = torch.argmax(probs).item()
                        confidence = probs[pred_idx].item()
                    
                    results.append({
                        "image_id": i,
                        "corruption": corr,
                        "severity": 0 if corr == 'clean' else s,
                        "model": model_name,
                        "prediction": classes[pred_idx],
                        "pred_idx": pred_idx,
                        "confidence": float(confidence),
                        "correct": pred_idx == true_label,
                        "true_label": true_class_name,
                        "true_idx": true_label
                    })
    
    with open(os.path.join(res_dir, 'demo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Done")

if __name__ == "__main__":
    main()
