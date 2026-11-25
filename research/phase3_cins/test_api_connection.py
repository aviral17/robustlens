import urllib.request
import json
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def numpy_to_base64_jpeg(arr_np):
    arr_uint8 = np.clip(arr_np * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("utf-8")

def query_model(img_b64, model="resnet"):
    req = urllib.request.Request(
        "https://robustlens.vercel.app/api/inference",
        data=json.dumps({"image": img_b64, "model": model, "corruption": "none", "severity": 1}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.loads(res.read().decode("utf-8"))
        return data.get("result", {})

if __name__ == "__main__":
    test_img = np.random.RandomState(42).uniform(0.2, 0.8, (224, 224, 3)).astype(np.float32)
    b64 = numpy_to_base64_jpeg(test_img)
    print("Testing local inference API connection...")
    try:
        res = query_model(b64, model="all")
        print("API Response Status:", res.get("status"))
        if "results" in res:
            for m in res["results"]:
                print(f"Model: {m['model']}, Pred: {m['prediction']}, Conf: {m['confidence']:.4f}")
    except Exception as e:
        print("API connection test error:", e)
