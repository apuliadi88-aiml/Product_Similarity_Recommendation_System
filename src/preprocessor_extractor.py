# ==========================================
# 1. Imports
# ==========================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import faiss

# ==========================================
# 2. Paths
# ==========================================

ROOT_DIR = "specify_your_root_directory_here"   # folder containing category subfolders
METADATA_FILE = os.path.join(ROOT_DIR, "data/filtered_metadata.csv")

# ==========================================
# 3. Load Metadata
# ==========================================
df = pd.read_csv(METADATA_FILE)

# Use full_path directly (best practice)
image_paths = df["full_path"].tolist()

print("Total images:", len(image_paths))

# ==========================================
# 4. Load ResNet50 Model (Feature Extractor)
# ==========================================
model = ResNet50(weights="imagenet",include_top=False,pooling="avg")   # gives 2048-d vectors

# ==========================================
# 5. Image Preprocessing Function
# ==========================================
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ==========================================
# 6. Embedding Extraction Function
# ==========================================
def get_embedding(img_path):
    img = preprocess_image(img_path)
    embedding = model.predict(img, verbose=0)
    return embedding[0]

# ==========================================
# 7. Extract Embeddings for All Images
# ==========================================
embeddings = []

for i, path in enumerate(image_paths):
    try:
        path = os.path.join(ROOT_DIR, path)
        emb = get_embedding(path)
        embeddings.append(emb)

        if i % 100 == 0:
            print(f"Processed {i} images")

    except Exception as e:
        print(f"Error processing {path}: {e}")

embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)

# ==========================================
# 8. Normalize Embeddings (IMPORTANT)
# ==========================================
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# ==========================================
# 9. Build FAISS Index (Cosine Similarity)
# ==========================================
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine (after normalization)
index.add(embeddings)

print("FAISS index size:", index.ntotal)

# ==========================================
# 10. Save Everything for Streamlit
# ==========================================

# Save FAISS index
faiss.write_index(index, os.path.join(ROOT_DIR, "faiss_index.bin"))

# Save embeddings
np.save(os.path.join(ROOT_DIR, "embeddings.npy"), embeddings)

# Save metadata
df.to_csv(os.path.join(ROOT_DIR, "metadata.csv"), index=False)

print("✅ All files saved successfully!")