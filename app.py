import os

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image


app = Flask(__name__)


# ==========================================
# AgriVision AI - Configuration
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# AI Model
# ==========================================

interpreter = None
input_details = None
output_details = None

class_names = []


# ==========================================
# Load TFLite Model
# ==========================================

def load_ai_model():

    global interpreter
    global input_details
    global output_details
    global class_names

    # ------------------------------
    # Load TFLite model
    # ------------------------------

    if os.path.exists(MODEL_PATH):

        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=1
        )

        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()
        )

        output_details = (
            interpreter.get_output_details()
        )

        print(
            "AgriVision AI TFLite model loaded successfully."
        )

    else:

        print(
            "ERROR: TFLite model not found."
        )


    # ------------------------------
    # Load class names
    # ------------------------------

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

        print(
            "ERROR: class_names.txt not found."
        )


load_ai_model()


# ==========================================
# Disease Information
# ==========================================

def get_disease_info(disease):

    # Normalize class name

    clean_name = (
        disease
        .replace("___", " ")
        .replace("_", " ")
        .strip()
    )

    name = clean_name.lower()


    # ======================================
    # Detect Crop
    # ======================================

    crop = clean_name.split()[0]

    if "corn" in name:
        crop = "Corn"

    elif "apple" in name:
        crop = "Apple"

    elif "tomato" in name:
        crop = "Tomato"

    elif "potato" in name:
        crop = "Potato"

    elif "grape" in name:
        crop = "Grape"

    elif "peach" in name:
        crop = "Peach"

    elif "pepper" in name:
        crop = "Pepper"

    elif "strawberry" in name:
        crop = "Strawberry"

    elif "cherry" in name:
        crop = "Cherry"

    elif "blueberry" in name:
        crop = "Blueberry"

    elif "raspberry" in name:
        crop = "Raspberry"

    elif "squash" in name:
        crop = "Squash"


    # ======================================
    # Healthy Plant
    # ======================================

    if "healthy" in name:

        return {

            "crop": crop,

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


    # ======================================
    # Leaf Scorch
    # ======================================

    if "leaf scorch" in name:

        return {

            "crop": crop,

            "status": "Needs Attention",

            "symptoms": [

                "Brown or scorched-looking areas may appear on leaves.",

                "Leaf margins may become dry or damaged.",

                "Affected tissue may gradually lose its healthy appearance."

            ],

            "recommendation": [

                "Check watering consistency and avoid severe moisture stress.",

                "Inspect the plant for environmental stress.",

                "Remove severely damaged leaves when appropriate.",

                "Continue monitoring new leaf growth."

            ]

        }


    # ======================================
    # Bacterial Disease
    # ======================================

    if "bacterial" in name:

        return {

            "crop": crop,

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


    # ======================================
    # Fungal Disease
    # ======================================

    fungal_words = [

        "blight",
        "mildew",
        "rust",
        "mold",
        "spot",
        "rot"

    ]


    if any(
        word in name
        for word in fungal_words
    ):

        return {

            "crop": crop,

            "status": "Needs Attention",

            "symptoms": [

                "Visible spots, discoloration or abnormal leaf areas may appear.",

                "Affected tissue may become dry or damaged."

            ],

            "recommendation": [

                "Remove severely affected leaves when appropriate.",

                "Avoid prolonged leaf wetness.",

                "Improve airflow around the plant.",

                "Monitor surrounding plants for similar symptoms."

            ]

        }


    # ======================================
    # Generic Disease
    # ======================================

    return {

        "crop": crop,

        "status": "Needs Attention",

        "symptoms": [

            "The AI detected a potentially abnormal plant condition.",

            "Visible symptoms can vary depending on the detected condition."

        ],

        "recommendation": [

            "Inspect the affected plant closely.",

            "Check watering, light and growing conditions.",

            "Monitor the plant for changes.",

            "Consider consulting a local agriculture expert for confirmation."

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

        "version": "4.0",

        "model_loaded":
            interpreter is not None,

        "classes":
            len(class_names),

        "model_type":
            "TensorFlow Lite"

    }


# ==========================================
# AI Prediction
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if interpreter is None:

        return jsonify({

            "success": False,

            "error":
                "AI model is not loaded."

        }), 500


    if not class_names:

        return jsonify({

            "success": False,

            "error":
                "Disease classes are not loaded."

        }), 500


    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded."

        }), 400


    image_file = request.files["image"]


    if image_file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "Please select an image."

        }), 400


    try:

        # ==================================
        # Open Image
        # ==================================

        image = Image.open(
            image_file
        ).convert("RGB")


        # ==================================
        # Resize Image
        # ==================================

        image = image.resize(
            IMG_SIZE
        )


        # ==================================
        # Convert to NumPy
        # ==================================

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # Add batch dimension

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ==================================
        # TFLite Input
        # ==================================

        input_index = (
            input_details[0]["index"]
        )

        interpreter.set_tensor(
            input_index,
            image_array
        )


        # ==================================
        # Run AI Inference
        # ==================================

        interpreter.invoke()


        # ==================================
        # Get Prediction
        # ==================================

        output_index = (
            output_details[0]["index"]
        )

        predictions = (
            interpreter.get_tensor(
                output_index
            )
        )


        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            predictions[0][
                predicted_index
            ] * 100
        )


        # ==================================
        # Safety Check
        # ==================================

        if predicted_index >= len(
            class_names
        ):

            raise ValueError(
                "Prediction class index is outside the class list."
            )


        predicted_class = (
            class_names[
                predicted_index
            ]
        )


        # ==================================
        # Disease Information
        # ==================================

        info = get_disease_info(
            predicted_class
        )


        # ==================================
        # Response
        # ==================================

        return jsonify({

            "success": True,

            "disease":
                predicted_class,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "crop":
                info["crop"],

            "status":
                info["status"],

            "symptoms":
                info["symptoms"],

            "recommendation":
                info["recommendation"],

            "message":
                "Plant image analyzed successfully."

        })


    except Exception as error:

        print(
            "Prediction Error:",
            str(error)
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

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
