"""Download and extract the 10 Imagenette classes for all 3 ImageNet-V2 variants.

Variants:
1. MatchedFrequency
2. Threshold0.7
3. TopImages

Target classes (10 ImageNet-1K indices):
[0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
Expected: exactly 10 images per class x 10 classes = 100 images per variant.
Total: 300 unique images across the three variants.
"""

import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "imagenet_v2"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    0: ("n01440764", "tench"),
    217: ("n02102040", "English springer"),
    482: ("n02979186", "cassette player"),
    491: ("n03000684", "chain saw"),
    497: ("n03028079", "church"),
    566: ("n03394916", "French horn"),
    569: ("n03417042", "garbage truck"),
    571: ("n03425413", "gas pump"),
    574: ("n03445777", "golf ball"),
    701: ("n03888257", "parachute"),
}

VARIANTS = {
    "MatchedFrequency": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
    "Threshold0.7": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
    "TopImages": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
}


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_variant(variant_name: str, url: str):
    var_dir = DATA_DIR / variant_name
    var_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing Variant: {variant_name}...")
    # Check if already fully extracted
    extracted_manifest = var_dir / "manifest.json"
    if extracted_manifest.exists():
        with open(extracted_manifest, "r") as f:
            data = json.load(f)
        if len(data.get("images", [])) == 100:
            print(f"  Already extracted 100 images for {variant_name}.")
            return data["images"]

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    res = urllib.request.urlopen(req)

    # Use streaming tar reader
    try:
        t = tarfile.open(mode="r|", fileobj=res)
    except Exception:
        # Fallback if server sent compressed bytes without auto-decompression
        res.close()
        res = urllib.request.urlopen(req)
        t = tarfile.open(mode="r|gz", fileobj=res)

    collected_images = []
    class_counts = {c: 0 for c in TARGET_CLASSES}
    total_target = len(TARGET_CLASSES) * 10  # 100 images

    for member in t:
        parts = member.name.split("/")
        # Path format: <prefix>/<class_idx>/<hash>.jpeg
        if len(parts) >= 3:
            class_str = parts[-2]
            filename = parts[-1]
            if class_str.isdigit():
                class_idx = int(class_str)
                if class_idx in TARGET_CLASSES and filename.endswith(".jpeg"):
                    class_dir = var_dir / str(class_idx)
                    class_dir.mkdir(parents=True, exist_ok=True)
                    out_path = class_dir / filename

                    # Extract file content
                    f_obj = t.extractfile(member)
                    if f_obj is not None:
                        content = f_obj.read()
                        with open(out_path, "wb") as out_f:
                            out_f.write(content)

                        # Verify valid image
                        try:
                            with Image.open(out_path) as im:
                                w, h = im.size
                        except Exception as e:
                            print(f"  Error reading extracted image {out_path}: {e}")
                            continue

                        csum = sha256_file(out_path)
                        wnid, cname = TARGET_CLASSES[class_idx]
                        collected_images.append({
                            "variant": variant_name,
                            "class_idx": class_idx,
                            "wordnet_id": wnid,
                            "class_name": cname,
                            "filename": filename,
                            "relative_path": f"{variant_name}/{class_idx}/{filename}",
                            "width": w,
                            "height": h,
                            "bytes": len(content),
                            "sha256": csum,
                        })
                        class_counts[class_idx] += 1
                        if len(collected_images) % 10 == 0:
                            print(f"  Extracted {len(collected_images)}/{total_target} images for {variant_name}...")

                        if len(collected_images) >= total_target and all(v == 10 for v in class_counts.values()):
                            print(f"  Successfully extracted all 100 images for {variant_name}!")
                            break

    res.close()

    # Save manifest
    with open(extracted_manifest, "w", encoding="utf-8") as f:
        json.dump({"variant": variant_name, "total_images": len(collected_images), "images": collected_images}, f, indent=2)

    return collected_images


def main():
    print("=" * 80)
    print("DOWNLOADING & EXTRACTING IMAGENET-V2 TEN-SYNSET REPLICATION SETS")
    print("=" * 80)

    all_variants_images = {}
    for var_name, url in VARIANTS.items():
        imgs = extract_variant(var_name, url)
        all_variants_images[var_name] = imgs
        print(f"Variant '{var_name}': {len(imgs)} valid images.")

    # Save master manifest
    master_manifest_path = DATA_DIR / "imagenet_v2_replication_manifest.json"
    with open(master_manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_variants_images, f, indent=2)
    print(f"\nSaved Master Replication Manifest to: {master_manifest_path}")


if __name__ == "__main__":
    main()
