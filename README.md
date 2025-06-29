# 🦠 Multimodal COVID-19 Detection & Chatbot

An AI-powered web application that detects COVID-19 from Chest X-ray images using a Convolutional Neural Network (CNN) and also answers user queries related to COVID-19 via a simple chatbot interface.

---

## 📌 Features

- 🧠 **Rule-based Chatbot** that answers common COVID-related queries.
- 🩻 **CNN-based COVID Detector** that analyzes uploaded chest X-ray images.
- 📈 **Confidence Score** with predictions.
- 🌐 **Streamlit Web App** for interactive interface.
- 🗂️ Image classification into: `COVID` or `Normal`.

---

## 🖼️ Sample Interface
![Screenshot 2025-06-29 160211](https://github.com/user-attachments/assets/e0f8eb2a-df5a-4308-8151-87ad216405bc)
![Screenshot 2025-06-29 160225](https://github.com/user-attachments/assets/558d3fe3-1bb8-4889-a727-b3dc24160e0b)





## 🛠️ Tech Stack

| Layer       | Technology                  |
|-------------|------------------------------|
| Frontend    | Streamlit                    |
| Backend     | Python                       |
| ML Framework | TensorFlow / Keras          |
| Model Type  | CNN (Convolutional Neural Network) |


---

## 📁 Project Structure
│
├── app.py # Streamlit app
├── chatbot.py # Model logic & chatbot responses
├── covid_weights.h5 # Trained CNN model weights
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

### 🔹 1. Clone the repository

```bash
git clone https://github.com/yourusername/Multimodal-COVID-Detection.git
cd Multimodal-COVID-Detection

**Install dependencies**
pip install -r requirements.txt

**Run the Streamlit app**
streamlit run app.py


**Model Architecture (CNN)**

Input Layer (256x256x3)
↓
Conv2D → ReLU
↓
Conv2D → ReLU
↓
MaxPooling → Dropout
↓
Conv2D → MaxPooling → Dropout
↓
Conv2D → MaxPooling → Dropout
↓
Flatten
↓
Dense (ReLU) → Dropout
↓
Dense (Sigmoid) → Output (0 or 1)

**Model Training**

model.fit(
    train_data,
    steps_per_epoch=8,
    epochs=10,
    validation_data=val_data
)
