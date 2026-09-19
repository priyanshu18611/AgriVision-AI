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
    "agrivision_model.tflite"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.txt"
)

IMG_SIZE = (224, 224)


# ==========================================
# Load TFLite AI Model
# ==========================================

interpreter = None
input_details = None
output_details = None
class_names = []


def load_ai_model():

    global interpreter
    global input_details
    global output_details
    global class_names

    # -----------------------------
    # Load TFLite Model
    # -----------------------------

    if os.path.exists(MODEL_PATH):

        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=2
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("AgriVision AI TFLite model loaded successfully.")

    else:

        print("ERROR: TFLite model not found.")


    # -----------------------------
    # Load Disease Classes
    # -----------------------------

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

    else:

        print("ERROR: class_names.txt not found.")


load_ai_model()


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

    return jsonify({

        "status": "healthy",

        "project": "AgriVision AI",

        "version": "3.0",

        "model_loaded": interpreter is not None,

        "classes": len(class_names)

    })


# ==========================================
# AI Prediction API
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------
    # Check Model
    # -----------------------------

    if interpreter is None:

        return jsonify({

            "success": False,

            "error": "AI model is not loaded."

        }), 500


    # -----------------------------
    # Check Image
    # -----------------------------

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

        # -----------------------------
        # Open Image
        # -----------------------------

        image = Image.open(
            image_file
        ).convert("RGB")


        # -----------------------------
        # Resize Image
        # -----------------------------

        image = image.resize(
            IMG_SIZE
        )


        # -----------------------------
        # Convert to NumPy
        # -----------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # -----------------------------
        # Add Batch Dimension
        # -----------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # -----------------------------
        # Set Input Tensor
        # -----------------------------

        input_index = input_details[0]["index"]

        interpreter.set_tensor(
            input_index,
            image_array
        )


        # -----------------------------
        # Run AI Prediction
        # -----------------------------

        interpreter.invoke()


        # -----------------------------
        # Get Prediction
        # -----------------------------

        output_index = output_details[0]["index"]

        predictions = interpreter.get_tensor(
            output_index
        )


        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            predictions[0][predicted_index] * 100
        )


        # -----------------------------
        # Get Disease Name
        # -----------------------------

        if predicted_index < len(class_names):

            predicted_class = class_names[
                predicted_index
            ]

        else:

            predicted_class = "Unknown"


        # -----------------------------
        # Crop Health
        # -----------------------------

        if confidence >= 80:

            health = "High Confidence"

        elif confidence >= 60:

            health = "Moderate Confidence"

        else:

            health = "Low Confidence"


        # -----------------------------
        # Response
        # -----------------------------

        return jsonify({

            "success": True,

            "disease": predicted_class,

            "confidence": round(
                confidence,
                2
            ),

            "health": health,

            "message":
                "Plant image analyzed successfully."

        })


    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

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

        debug=False

    )
