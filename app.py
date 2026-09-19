import os

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image


app = Flask(__name__)

# ==========================================
# AgriVision AI - Configuration
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "agrivision_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.txt"
)

IMG_SIZE = (224, 224)


# ==========================================
# Load AI Model
# ==========================================

model = None
class_names = []


def load_ai_model():

    global model, class_names

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print(
            "AgriVision AI model loaded successfully."
        )

    if os.path.exists(CLASS_NAMES_PATH):

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            class_names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(
            f"Loaded {len(class_names)} disease classes."
        )


load_ai_model()


# ==========================================
# Disease Information
# ==========================================

def get_disease_info(disease):

    name = disease.lower()

    # ------------------------------
    # Healthy
    # ------------------------------

    if "healthy" in name:

        crop = disease.replace(
            " healthy",
            ""
        ).strip()

        return {
            "crop": crop.title(),
            "status": "Healthy",
            "symptoms": [
                "No major visible disease symptoms detected.",
                "Leaf appearance is consistent with a healthy sample."
            ],
            "recommendation": [
                "Continue regular plant monitoring.",
                "Maintain appropriate watering and sunlight.",
                "Keep the growing area clean."
            ]
        }


    # ------------------------------
    # Leaf Scorch
    # ------------------------------

    if "leaf scorch" in name:

        crop = disease.replace(
            " Leaf Scorch",
            ""
        ).replace(
            " leaf scorch",
            ""
        ).strip()

        return {
            "crop": crop.title(),
            "status": "Needs Attention",
            "symptoms": [
                "Brown or scorched-looking leaf areas.",
                "Dry or damaged leaf margins may appear.",
                "Leaf tissue may gradually lose its healthy appearance."
            ],
            "recommendation": [
                "Check watering consistency and avoid severe moisture stress.",
                "Inspect the plant for environmental stress.",
                "Remove severely damaged leaves if appropriate.",
                "Continue monitoring new leaf growth."
            ]
        }


    # ------------------------------
    # Bacterial Disease
    # ------------------------------

    if "bacterial" in name:

        crop = disease.split("___")[0]

        return {
            "crop": crop.replace("_", " ").title(),
            "status": "Needs Attention",
            "symptoms": [
                "Leaf discoloration or damaged spots may occur.",
                "Affected areas can expand under favorable conditions."
            ],
            "recommendation": [
                "Remove badly affected plant material where appropriate.",
                "Avoid unnecessary leaf wetting.",
                "Improve air circulation around plants.",
                "Monitor nearby plants for similar symptoms."
            ]
        }


    # ------------------------------
    # Fungal Disease
    # ------------------------------

    fungal_words = [
        "blight",
        "mildew",
        "rust",
        "mold",
        "spot",
        "rot"
    ]

    if any(word in name for word in fungal_words):

        crop = disease.split("___")[0]

        return {
            "crop": crop.replace("_", " ").title(),
            "status": "Needs Attention",
            "symptoms": [
                "Visible spots, discoloration or abnormal leaf areas.",
                "Affected tissue may become dry or damaged."
            ],
            "recommendation": [
                "Remove severely affected leaves when appropriate.",
                "Avoid prolonged leaf wetness.",
                "Improve airflow around the plant.",
                "Monitor surrounding plants for similar symptoms."
            ]
        }


    # ------------------------------
    # Generic Disease
    # ------------------------------

    crop = disease.split("___")[0]

    return {
        "crop": crop.replace("_", " ").title(),
        "status": "Needs Attention",
        "symptoms": [
            "The AI detected a potentially abnormal plant condition.",
            "Visible symptoms can vary depending on the disease."
        ],
        "recommendation": [
            "Inspect the affected plant closely.",
            "Check watering, light and growing conditions.",
            "Monitor the plant for changes.",
            "Consider consulting a local agricultural expert for confirmation."
        ]
    }


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# Health Check
# ==========================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "project": "AgriVision AI",
        "version": "3.1",
        "model_loaded": model is not None,
        "classes": len(class_names)
    }


# ==========================================
# AI Prediction API
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
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

        image = Image.open(
            image_file
        ).convert("RGB")


        # Resize

        image = image.resize(
            IMG_SIZE
        )


        # Convert to NumPy

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # Add batch dimension

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # AI Prediction

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


        predicted_class = class_names[
            predicted_index
        ]


        # Disease information

        info = get_disease_info(
            predicted_class
        )


        return jsonify({

            "success": True,

            "disease": predicted_class,

            "confidence": round(
                confidence,
                2
            ),

            "crop": info["crop"],

            "status": info["status"],

            "symptoms": info["symptoms"],

            "recommendation": info["recommendation"],

            "message":
                "Plant image analyzed successfully."

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
