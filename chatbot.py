 # chatbot.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.utils import img_to_array

# Build the CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load model weights
def load_trained_weights(model, path="covid_weights.h5"):
    model.load_weights(path)
    return model

# Predict image label
def predict_image(img):
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "🟥 COVID-19 Detected" if prediction > 0.5 else "🟩 Normal (No COVID)"
    return label, prediction

# Basic rule-based chatbot
def get_response(user_input):
    user_input = user_input.lower()

    if "covid" in user_input:
        return "COVID-19 is caused by the SARS-CoV-2 virus. It affects the lungs and breathing."
    elif "symptoms" in user_input:
        return "Common symptoms include fever, cough, fatigue, and breathing difficulty."
    elif "prevent" in user_input:
        return "Prevention includes vaccination, wearing masks, hand hygiene, and distancing."
    elif "model" in user_input:
        return "Our AI model uses Convolutional Neural Networks to detect COVID from X-rays."
    elif "accuracy" in user_input:
        return "The model performs well with high accuracy on validation data."
    elif "help" in user_input:
        return "You can ask about COVID, symptoms, prevention, or how the model works."
    else:
        return "Sorry, I didn’t understand. Try asking about COVID or the model."

# Load model once globally
model = build_model()
model = load_trained_weights(model, path="covid_weights.h5")
