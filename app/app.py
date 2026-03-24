import streamlit as st
import torch
import numpy as np
import pickle
import os
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "features.pkl")
PATHS_PATH = os.path.join(BASE_DIR, "..", "model", "paths.pkl")

DATASET_DIR = os.path.join(BASE_DIR, "..", "fashion-dataset", "images")

# LOAD FEATURES
@st.cache_data
def load_features():
    features = pickle.load(open(FEATURES_PATH, "rb"))
    image_paths = pickle.load(open(PATHS_PATH, "rb"))
    return features, image_paths

features, image_paths = load_features()

# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# FEATURE EXTRACTION
def extract_features(img):
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)

    feat = feat.cpu().numpy().flatten()
    feat = feat / np.linalg.norm(feat)

    return feat

# RECOMMEND FUNCTION
def recommend(query_img, top_k=5):
    query_feat = extract_features(query_img)

    similarities = cosine_similarity([query_feat], features)[0]
    indices = similarities.argsort()[-top_k:][::-1]

    return indices

#  UI
st.title("Fashion Recommendation System (PyTorch + CNN)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.write("## Similar Products")

    indices = recommend(img)

    cols = st.columns(5)

    for i, col in enumerate(cols):
        try:
            original_path = image_paths[indices[i]]
            img_name = os.path.basename(original_path)

            correct_path = os.path.join(DATASET_DIR, img_name)

            if os.path.exists(correct_path):
                col.image(correct_path, use_container_width=True)
            else:
                col.write(" Not found")

        except:
            col.write(" Error")
        
st.markdown(
    "<div style='text-align:center; margin-top:30px;'>"
    "Built by <b>Satheesh</b> | Machine Learning Project"
    "</div>",
    unsafe_allow_html=True
)