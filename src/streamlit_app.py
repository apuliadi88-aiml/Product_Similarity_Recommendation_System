import streamlit as st
import numpy as np
import pandas as pd
import faiss
from PIL import Image
import tensorflow as tf
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
st.set_page_config(layout="wide")
# ==========================================
# Load Model (cached)
# ==========================================
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# ==========================================
# Load FAISS + Metadata (cached)
# ==========================================
@st.cache_resource
def load_data():
    index = faiss.read_index("faiss_index.bin")
    metadata = pd.read_csv("metadata.csv")
    return index, metadata

index, metadata = load_data()

# ==========================================
# Preprocess Uploaded Image
# ==========================================
def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ==========================================
# Get Embedding - Using ResNet50
# ==========================================
def get_embedding_from_upload(uploaded_file):
    img = preprocess_uploaded_image(uploaded_file)
    emb = model.predict(img, verbose=0)[0]
    
    # Normalize
    emb = emb / np.linalg.norm(emb)
    return emb

# ==========================================
# Function to Search Similar Products
# ==========================================
def search_similar(uploaded_file, top_k, class_type):
    
    query_emb = get_embedding_from_upload(uploaded_file).reshape(1, -1)
    
    # Search more to allow filtering
    distances, indices = index.search(query_emb, top_k * 3)
    
    results = []
    
    for i, idx in enumerate(indices[0]):
        row = metadata.iloc[idx]
        
        # Apply class filter
        if class_type != "All":
            if class_type.lower() not in row["full_path"].lower():
                continue
        
        similarity = float(distances[0][i]) * 100  # convert to %
        
        results.append({
            "image_path": row["full_path"],
            "similarity": similarity
        })
        
        if len(results) >= top_k:
            break
    
    return results

# ==========================================
# Streamlit UI - TWO PANEL LAYOUT
# ==========================================

st.title("Product Similarity Recommendation System")

# Create two main columns
left_col, right_col = st.columns([1, 2])  # left smaller, right bigger

# ==========================================
# LEFT SIDE → INPUTS
# ==========================================
with left_col:
    st.header("Upload & Filters")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    class_type = st.selectbox(
        "Select Class Type", 
        ["All", "Bicycle", "Chair", "Toaster"]
    )

    top_k = st.selectbox(
    "Number of results",
    [6, 7, 8, 9, 10],
    index=0  # default selection
    )

    search_button = st.button("Find Similar Products")

# ==========================================
# RIGHT SIDE → RESULTS
# ==========================================
with right_col:
    st.header("Results")

    if search_button:
        if uploaded_file is not None:

            st.subheader("Uploaded Image")
            st.image(uploaded_file, width=250)

            st.write("Finding similar products...")

            results = search_similar(uploaded_file, top_k, class_type)

            if len(results) == 0:
                st.warning("No matching results found.")
            else:
                st.success(f"Showing Top {len(results)} Results")

                cols = st.columns(3, gap="large")

                for i, res in enumerate(results):
                    with cols[i % 3]:
                        st.image(
                            res["image_path"],
                            use_column_width=True
                        )
                        st.caption(f"Similarity: {res['similarity']:.2f}%")

        else:
            st.warning("Please upload an image file.")