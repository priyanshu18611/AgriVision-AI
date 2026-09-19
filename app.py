import os

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image


app = Flask(__name__)

# ==========================================
# AgriVision AI - Configuration
# ==========================================

MODEL_PATH = "models/agrivision_model.keras"
CLASS_NAMES_PATH = "models/class_names.txt"

IMG_SIZE = (224, 224)


# ==========================================
# Load AI Model
# ==========================================

model = None
class_names = []


def load_ai_model():
    global model, class_names

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("AgriVision AI model loaded successfully.")

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(f"Loaded {len(class_names)} disease classes.")


load_ai_model()


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Health Check
# ==========================================

@app.route("/health")
def health():
    return {
        "status": "healthy",
        "project": "AgriVision AI",
        "version": "2.0",
        "model_loaded": model is not None,
        "classes": len(class_names)
    }


# ==========================================
# AI Prediction API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "AI model is not loaded."
        }), 500

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    try:
        # Open image
        image = Image.open(image_file).convert("RGB")

        # Resize
        image = image.resize(IMG_SIZE)

        # Convert to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        predicted_class = class_names[predicted_index]

        return jsonify({
            "success": True,
            "disease": predicted_class,
            "confidence": round(confidence, 2),
            "message": "Plant image analyzed successfully."
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
