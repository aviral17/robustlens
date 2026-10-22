import os
import json
import torch
import timm
from PIL import Image
from torchvision import transforms

def main():
    print("Starting rapid inference on existing images...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_dir = os.path.join(base_dir, 'apps', 'web', 'public', 'examples')
    res_dir = os.path.join(base_dir, 'results', 'demo')
    
    models = {
        "resnet50.a1_in1k": timm.create_model("resnet50.a1_in1k", pretrained=True).eval(),
        "convnext_tiny.in12k_ft_in1k": timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True).eval(),
        "vit_base_patch16_224.augreg_in21k_ft_in1k": timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval(),
        "swin_tiny_patch4_window7_224.ms_in1k": timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True).eval(),
    }
    
    data_configs = {name: timm.data.resolve_data_config({}, model=m) for name, m in models.items()}
    val_transforms = {name: timm.data.create_transform(**cfg, is_training=False) for name, cfg in data_configs.items()}
    
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes = [line.decode("utf-8").strip() for line in urllib.request.urlopen(url).readlines()]
    
    labels = {
        0: 281, # cat
        1: 207, # golden retriever
        2: 15   # robin
    }
    
    results = []
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    for img_name in images:
        parts = img_name.replace('.jpg', '').split('_')
        # img_0_clean or img_0_gaussian_noise_s1
        img_id = int(parts[1])
        if parts[2] == 'clean':
            corr = 'clean'
            sev = 0
        else:
            sev = int(parts[-1].replace('s', ''))
            corr = '_'.join(parts[2:-1])
            
        true_label = labels[img_id]
        true_class_name = classes[true_label]
        
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        for model_name, model in models.items():
            input_tensor = val_transforms[model_name](img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
            
            results.append({
                "image_id": img_id,
                "corruption": corr,
                "severity": sev,
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
    print("Done generating demo results!")

if __name__ == "__main__":
    main()
