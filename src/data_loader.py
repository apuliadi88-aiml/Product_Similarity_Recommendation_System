import os
import pandas as pd
import shutil
import random

#--------------------------------
# Dataset path
#--------------------------------
BASE_DIR = "/Users/amritha/Desktop/AI_ML/GUVI Projects/ComputerVision/Stanford_Online_Products/"
categories = ["bicycle_final", "chair_final", "toaster_final"]
images_per_category = 600

#--------------------------------
# Collect images
#--------------------------------
category_images = {}
selected_images = []

for cat in categories:
    cat_path = os.path.join(BASE_DIR, cat)
    
    all_files = [
        os.path.join(cat_path, f)
        for f in os.listdir(cat_path)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]
    
    random.shuffle(all_files)
    
    cat_selected = all_files[:images_per_category] if len(all_files) >= images_per_category else all_files
    
    category_images[cat] = cat_selected
    selected_images.extend(cat_selected)
    
    print(f"{cat}: {len(cat_selected)} images selected")

print("Total selected images:", len(selected_images))

# Normalize paths
selected_images_set = set(os.path.normpath(p).lower() for p in selected_images)

#--------------------------------
# Create folders
#--------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

for cat in categories:
    os.makedirs(os.path.join(DATA_DIR, cat), exist_ok=True)

#--------------------------------
# Copy images
#--------------------------------
path_mapping = {}  # OLD -> NEW mapping

for cat, files in category_images.items():
    for img_path in files:
        dest_path = os.path.join(DATA_DIR, cat, os.path.basename(img_path))
        
        if not os.path.exists(dest_path):
            shutil.copy(img_path, dest_path)
        
        # Store mapping (normalized)
        path_mapping[os.path.normpath(img_path).lower()] = dest_path

print("Selected images copied.")

#--------------------------------
# Load metadata
#--------------------------------
meta_file = os.path.join(BASE_DIR, "Ebay_info.txt")

df_meta = pd.read_csv(
    meta_file,
    sep=" ",
    header=None,
    names=["image_id", "class_id", "super_class_id", "path"]
)

# Normalize metadata paths
df_meta["full_path"] = df_meta["path"].apply(
    lambda x: os.path.normpath(os.path.join(BASE_DIR, x)).lower()
)

#--------------------------------
# Filter metadata
#--------------------------------
df_filtered = df_meta[df_meta["full_path"].isin(selected_images_set)].copy()

print("Metadata matches:", len(df_filtered))

#--------------------------------
# Map directly using dictionary (FAST)
#--------------------------------
df_filtered["full_path"] = df_filtered["full_path"].map(path_mapping)

# Drop any missing mappings
df_filtered = df_filtered.dropna(subset=["full_path"]).reset_index(drop=True)

#--------------------------------
# Save metadata (ONLY new full_path)
#--------------------------------
df_filtered = df_filtered[["image_id", "class_id", "super_class_id", "full_path"]]

metadata_file = os.path.join(DATA_DIR, "filtered_metadata.csv")
df_filtered.to_csv(metadata_file, index=False)

print("Final filtered metadata count:", len(df_filtered))
print("Saved at:", metadata_file)