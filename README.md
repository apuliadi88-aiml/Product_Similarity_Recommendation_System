# Product_Similarity_Recommendation_System
Overview

This project is a machine learning-based product similarity recommendation system built using deep learning ResNet50 embeddings and FAISS for fast similarity search. It recommends visually or feature-similar products based on image and feature representations.

The system is designed to work on the Stanford Online Products dataset and provides an end-to-end pipeline from data preprocessing to similarity-based retrieval and visualization.

#Project Structure
Product_Similarity_Recommendation_System/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessor_extractor.py
│   ├── streamlit_app.py
│
├── requirements.txt          
└── README.md                     

# How to Run the Project
1. Clone the repository
git clone https://github.com/apuliadi88-aiml/Product_Similarity_Recommendation_System.git

cd Product_Similarity_Recommendation_System

3. Install dependencies
pip install -r requirements.txt

4. Download the Dataset
Download the Stanford Online Products Dataset from the official source:
https://www.tensorflow.org/datasets/catalog/stanford_online_products
After downloading:
Extract the dataset into your project directory (or a data/ folder)

5. Run Data Loader
This step loads and organizes the dataset.

python src/data_loader.py

6. Run Preprocessing & Feature Extraction
This step:
Preprocesses images
Extracts embeddings using the deep learning model
Builds feature representations

python src/preprocessor_extractor.py

7. Run Streamlit Application
Launch the interactive UI for product similarity search:

streamlit run src/streamlit_app.py


