# NeuralScript — Character Recognition 🧠✏️

![Render Deployment](https://img.shields.io/badge/Render-Deployed-success?style=for-the-badge&logo=render)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?style=for-the-badge&logo=pytorch)

NeuralScript is an AI-powered handwriting character recognition tool featuring a responsive, web-based canvas interface. Users can draw alphabetic characters on the provided whiteboard, and the backend neural network instantly predicts the written character in real time. 

The application is built completely with vanilla HTML/JS on the frontend using an ultra-modern aesthetic, and relies on **FastAPI** to connect the UI to a custom **PyTorch** Convolutional Neural Network (CNN).

---

## ✨ Features

- **Live On-Canvas Prediction**: Predicts your handwriting as you draw using a debounced, highly-responsive API endpoint.
- **Custom Model Retraining**: Allows users to save their own handwritten character samples directly from the UI and retrain the CNN on-the-fly to adapt to specific handwriting styles.
- **Sleek Cyberpunk Interface**: Ultra-modern dark mode user interface featuring dynamic CSS gradients, glowing animations, layout preservation, and responsive resizing.
- **Touch-Friendly**: Fully supports stylus arrays and touch-screens on mobile or tablet environments.
- **Ready for Production Deploy**: Includes a `render.yaml` blueprint to natively host this exact web architecture on Render without tricky desktop UI configurations.

---

## 🏗️ Architecture

- `app/main.py`: The core FastAPI backend server logic handling image processing, inference triggers, and static routing.
- `app/static/`: Javascript canvas listeners and CSS variables for the web UI.
- `inference/`: Responsible for preprocessing the raw bytes coming from the HTML canvas mapping them into PyTorch-friendly grayscale arrays, and running forward passes on the trained CNN.
- `model/`: The PyTorch Neural Network architecture definition (`cnn.py`) and training scripts.

---

## 🚀 Setup & Installation

### Prerequisites
You need **Python 3.10+** installed on your machine. We recommend managing your environment natively using `venv`.

1. **Clone the project:**
   ```bash
   git clone https://github.com/forex911/Character_Recognition.git
   cd Character_Recognition
   ```

2. **Setup the Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On MacOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application locally:**
   ```bash
   uvicorn app.main:app --reload
   ```
   *Your server will start at: `http://localhost:8000`*

---

## 🌍 Cloud Deployment (Render)

We have included a `render.yaml` blueprint for one-click deployment!

1. Commit and push your code to your GitHub/Gitlab account. 
2. Link your repository via your [Render Dashboard](https://dashboard.render.com). 
3. Render will automatically parse the layout, resolve the Python dependencies, and spawn the Unicorn webservice.

---
*Created by **forex911***
