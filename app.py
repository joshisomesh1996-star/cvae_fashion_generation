import streamlit as st
import torch
from model import CVAE
from utils import generate_images
from config import *

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model once and cache it
@st.cache_resource
def load_model():

    model = CVAE()

    checkpoint = torch.load("best_cvae.pth", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


model = load_model()


# Page Title
st.title("👕 Fashion Outfit Generator (CVAE)")
st.write("Generate AI fashion outfits based on conditions.")


# Sidebar Controls
st.sidebar.header("Choose Outfit Conditions")

gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
article = st.sidebar.selectbox("Article", list(article_map.keys()))
color = st.sidebar.selectbox("Color", list(color_map.keys()))
season = st.sidebar.selectbox("Season", list(season_map.keys()))

num = st.sidebar.slider("Number of Samples", 1, 8, 4)


# Generate Button
if st.button("Generate Outfit"):

    with st.spinner("Generating outfits..."):

        imgs = generate_images(model, gender, article, color, season, num)

    st.subheader("Generated Outfits")

    cols = st.columns(num)

    for i in range(num):

        img = imgs[i].transpose(1, 2, 0)

        cols[i].image(img, width="stretch")