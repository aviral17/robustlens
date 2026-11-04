import json
from pathlib import Path

nb_path = Path("ml/notebooks/02_distortion_fingerprints_confirmatory.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Total cells in confirmatory notebook:", len(nb["cells"]))

# Update any mention of 3,925 or 298,300 to 3,825 and 290,700
updated = False
for idx, cell in enumerate(nb["cells"]):
    new_source = []
    cell_updated = False
    for line in cell["source"]:
        new_line = line
        if "298,300" in new_line:
            new_line = new_line.replace("298,300", "290,700")
            cell_updated = True
        if "298300" in new_line:
            new_line = new_line.replace("298300", "290700")
            cell_updated = True
        if "3,925" in new_line:
            new_line = new_line.replace("3,925", "3,825")
            cell_updated = True
        if "3925" in new_line and "FastAI" in new_line:
            new_line = new_line.replace("3925", "3825")
            cell_updated = True
        new_source.append(new_line)
    if cell_updated:
        cell["source"] = new_source
        updated = True
        print(f"Updated Cell {idx}")

if updated:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated notebook with exact 3,825 images / 290,700 forward pass count at {nb_path}")
else:
    print("Notebook already checked.")
