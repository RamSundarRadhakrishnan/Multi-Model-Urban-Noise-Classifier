import pandas as pd
from pathlib import Path

manifest_path = Path("../../datasets/vggsound/vggsound_subset_manifest.csv")
mapping_path = Path("vggsound_urban_map.csv")

manifest = pd.read_csv(manifest_path)
mapping = pd.read_csv(mapping_path)

print("Manifest columns:", manifest.columns.tolist())
print("Mapping columns:", mapping.columns.tolist())

manifest["vggsound_label_norm"] = manifest["vggsound_label"].astype(str).str.lower().str.strip()
mapping["vggsound_label_norm"] = mapping["vggsound_label"].astype(str).str.lower().str.strip()

label_to_target = dict(zip(mapping["vggsound_label_norm"], mapping["target_class"]))

manifest["target_class"] = manifest["vggsound_label_norm"].map(label_to_target)

missing = manifest[manifest["target_class"].isna()]["vggsound_label"].drop_duplicates().tolist()

if missing:
    print("Missing mappings:")
    for label in missing:
        print(label)
    raise SystemExit("Some labels are not mapped.")

manifest = manifest.drop(columns=["vggsound_label_norm"])
manifest.to_csv(manifest_path, index=False)

print("Updated manifest:", manifest_path)
print(manifest["target_class"].value_counts())