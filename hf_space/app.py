import gradio as gr
import torch
import timm
from PIL import Image, ImageFilter
import io
import numpy as np
import time
import urllib.request
from torchvision import transforms

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = urllib.request.urlopen(url)
        classes = [line.decode("utf-8").strip() for line in response.readlines()]
        return classes
    except Exception as e:
        return [f"class_{i}" for i in range(1000)]

CLASSES = get_imagenet_classes()

# Load Models
models_dict = {
    "ResNet-50": "resnet50.a1_in1k",
    "ConvNeXt-Tiny": "convnext_tiny.in12k_ft_in1k",
    "ViT-Base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "Swin-Tiny": "swin_tiny_patch4_window7_224.ms_in1k"
}
loaded_models = {}
val_transforms = {}

print("Loading models...")
for friendly_name, model_id in models_dict.items():
    try:
        model = timm.create_model(model_id, pretrained=True).eval().to(device)
        data_config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        loaded_models[friendly_name] = model
        val_transforms[friendly_name] = transform
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
print("Models loaded successfully.")

def apply_gaussian_noise(img, severity):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    img_arr = np.array(img) / 255.
    img_arr = np.clip(img_arr + np.random.normal(size=img_arr.shape, scale=c), 0, 1) * 255
    return Image.fromarray(np.uint8(img_arr))

def apply_defocus_blur(img, severity):
    c = [1, 2, 3, 4, 5][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=c))

def apply_jpeg_compression(img, severity):
    c = [80, 65, 50, 35, 20][severity - 1]
    output = io.BytesIO()
    img.save(output, 'JPEG', quality=c)
    return Image.open(output)

def apply_brightness(img, severity):
    from PIL import ImageEnhance
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 + c)

def apply_contrast(img, severity):
    from PIL import ImageEnhance
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1 + c)

def apply_corruption(img, corruption, severity):
    if corruption == "gaussian_noise": return apply_gaussian_noise(img, severity)
    if corruption == "defocus_blur": return apply_defocus_blur(img, severity)
    if corruption == "jpeg_compression": return apply_jpeg_compression(img, severity)
    if corruption == "brightness": return apply_brightness(img, severity)
    if corruption == "contrast": return apply_contrast(img, severity)
    return img

@spaces.GPU
def run_inference(image, model_choice, corruption, severity):
    start_time = time.time()
    
    # 1. Apply corruption
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize keeping aspect ratio, crop center to ensure valid input size
    image = transforms.Resize(256)(image)
    image = transforms.CenterCrop(224)(image)
    
    corrupted_image = apply_corruption(image, corruption, int(severity))
    
    results = []
    
    models_to_run = list(loaded_models.keys()) if model_choice == "Compare All Models" else [model_choice]
    
    for m_name in models_to_run:
        model = loaded_models[m_name]
        transform = val_transforms[m_name]
        
        input_tensor = transform(corrupted_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            
            top5_prob, top5_catid = torch.topk(probs, 5)
            
            top_k = []
            for i in range(top5_prob.size(0)):
                top_k.append({
                    "class": CLASSES[top5_catid[i]],
                    "probability": float(top5_prob[i].item())
                })
            
            pred_class = top_k[0]["class"]
            confidence = top_k[0]["probability"]
            
            results.append({
                "model": m_name,
                "prediction": pred_class,
                "confidence": confidence,
                "top_k": top_k
            })
            
    inf_time = time.time() - start_time
    
    return {
        "status": "success",
        "results": results,
        "inference_time_seconds": round(inf_time, 2)
    }

# Gradio Interface
demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(choices=["ResNet-50", "ConvNeXt-Tiny", "ViT-Base", "Swin-Tiny", "Compare All Models"], label="Model"),
        gr.Dropdown(choices=["gaussian_noise", "defocus_blur", "jpeg_compression", "brightness", "contrast"], label="Corruption"),
        gr.Slider(minimum=1, maximum=5, step=1, label="Severity")
    ],
    outputs="json",
    title="RobustLens Inference API",
    description="Live inference API for testing vision models under distribution shift."
)

if __name__ == "__main__":
    demo.launch()
