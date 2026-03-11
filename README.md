# 👕 CVAE Fashion Outfit Generator

A **Conditional Variational Autoencoder (CVAE)** that generates fashion clothing images based on attributes such as **gender, article type, color, and season**.

This project includes a **Streamlit web application** that allows users to interactively generate outfits using the trained model.

---

# 🧠 Project Overview

Generative models can learn complex data distributions and create new samples.  
In this project, a **Conditional Variational Autoencoder (CVAE)** is implemented to generate clothing images conditioned on fashion attributes.

The model learns a **latent representation of clothing images** and uses conditional vectors to control the generation process.

Users can select clothing attributes from a web interface and generate new fashion items using the trained model.

---

# 🏗 Model Architecture

The CVAE model consists of three main components:

### Encoder
- Convolutional layers extract image features
- Flattened features are concatenated with a **condition vector**
- The encoder outputs **mean (μ)** and **log variance (log σ²)**

### Latent Space
A latent vector **z** is sampled using the **reparameterization trick**:

```
z = μ + σ * ε
ε ~ N(0,1)
```

### Decoder
- The sampled latent vector **z** is concatenated with the condition vector
- Transposed convolution layers reconstruct the image
- Output image size: **3 × 128 × 128**

---

# 🎛 Conditioning Variables

The model conditions image generation on four attributes:

| Attribute | Options |
|--------|--------|
| Gender | Men, Women, Unisex |
| Article Type | Jeans, Shirts, Trousers, Tshirts |
| Color | Multiple fashion colors |
| Season | Fall, Spring, Summer, Winter |

These attributes are encoded as a **30-dimensional condition vector**.

---

# 📂 Project Structure

```
cvae_fashion_generation/
│
├── app.py
├── model.py
├── utils.py
├── config.py
├── requirements.txt
├── cvae-fashion-generation.ipynb
├── README.md
└── .gitignore
```

---

# 🚀 Running the Application

### 1️⃣ Create Environment

```bash
conda create -n cvae-fashion python=3.10
conda activate cvae-fashion
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Model Weights

Due to GitHub file size limits, the trained model is not included.

Download it and place it in the project root:

```
best_cvae.pth
```

---

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py --server.port 8503
```

Open the browser:

```
http://localhost:8503
```

---

# 🖥 Streamlit Interface

The web app allows users to:

- Select **gender**
- Select **article type**
- Select **color**
- Select **season**
- Generate multiple outfits

The model produces new fashion samples based on these conditions.

---

# 🧪 Training

Training was performed using:

- **PyTorch**
- **Fashion dataset with attribute annotations**
- **Reconstruction Loss (MSE)**
- **KL Divergence Loss**

Total Loss:

```
Loss = Reconstruction Loss + KL Divergence
```

---

# 📊 Results

The trained CVAE can generate diverse fashion items while maintaining the selected attributes.

Example outputs include:

- Men's blue jeans
- Women's black t-shirts
- Spring shirts
- Winter trousers

---

# 🛠 Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Matplotlib

---

# 💡 Future Improvements

Possible extensions for this project:

- Latent space visualization
- Fashion style interpolation
- Higher resolution generation
- Comparison with GAN models
- Deploying the app on Streamlit Cloud

---

# 📜 License

This project is for educational and research purposes.

---

# ✨ Author

Somesh Joshi  
M.Tech AI Project