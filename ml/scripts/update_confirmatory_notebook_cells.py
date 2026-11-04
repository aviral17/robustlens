import json
from pathlib import Path

nb_path = Path("ml/notebooks/02_distortion_fingerprints_confirmatory.ipynb")
pilot_ids_path = Path("results/matched_fidelity_study/pilot_exploratory_100_image_ids.json")

with open(pilot_ids_path, "r", encoding="utf-8") as f:
    pilot_info = json.load(f)

# Load the official manifest report to get exact dataset row indices for the 100 pilot images
with open("results/matched_fidelity_study/official_manifest_matching_report.json", "r", encoding="utf-8") as f:
    manifest_report = json.load(f)

pilot_row_indices = manifest_report["pilot_exploratory_audit"]["dataset_row_indices"]
assert len(pilot_row_indices) == 100

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c4 = nb["cells"][4]["source"]

# Find where val_ds is defined
new_c4 = []
for line in c4:
    if 'full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")' in line:
        new_c4.append(line)
        new_c4.append(f"# Quarantined 100 Exploratory Pilot Image Dataset Row Indices\n")
        new_c4.append(f"PILOT_ROW_INDICES = set({pilot_row_indices})\n")
        new_c4.append(f"# Strictly select the 3,825 untouched confirmatory validation images\n")
        new_c4.append(f"confirmatory_indices = [r for r in range(9469, 13394) if r not in PILOT_ROW_INDICES]\n")
        new_c4.append(f"val_ds = full_ds.select(confirmatory_indices)\n")
        new_c4.append(f"assert len(val_ds) == 3825, f'Expected 3,825 confirmatory images, got {{len(val_ds)}}'\n")
        new_c4.append(f"print(f'Successfully loaded {{len(val_ds)}} untouched confirmatory validation images (100 exploratory pilot images strictly quarantined).')\n")
    elif 'val_ds = full_ds.select(range(9469, len(full_ds)))' in line or 'assert len(val_ds) == 3925' in line or 'assert len(val_ds) == 3825' in line:
        continue  # Replaced above
    elif 'LEVELS = ["Level_1_Mild", "Level_2_ModMild", "Level_3_Moderate"]' in line:
        new_c4.append('LEVELS = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]\n')
    else:
        new_c4.append(line)

nb["cells"][4]["source"] = new_c4

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebook Cell 4 with exact 3,825 confirmatory images and level names.")
