import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# AgriVision AI - Plant Disease Classifier
# ==========================================

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


# ==========================================
# Check Dataset
# ==========================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        "Dataset folder not found. "
        "Please add the PlantVillage dataset."
    )


# ==========================================
# Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==========================================
# Load Validation Dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==========================================
# Get Disease Classes
# ==========================================

class_names = train_dataset.class_names

print("\nDetected Classes:")
for index, name in enumerate(class_names):
    print(index, ":", name)

NUM_CLASSES = len(class_names)

print("\nTotal Classes:", NUM_CLASSES)


# ==========================================
# Improve Performance
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])


# ==========================================
# Pretrained MobileNetV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers

base_model.trainable = False


# ==========================================
# Build AgriVision AI Model
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Model Summary
# ==========================================

model.summary()


# ==========================================
# Early Stopping
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# ==========================================
# Save Model
# ==========================================

os.makedirs("models", exist_ok=True)

model.save(
    "models/agrivision_model.keras"
)


# ==========================================
# Save Class Names
# ==========================================

with open(
    "models/class_names.txt",
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:
        file.write(class_name + "\n")


print("\n===================================")
print("AgriVision AI Training Completed!")
print("===================================")

print(
    "Model saved to: "
    "models/agrivision_model.keras"
)

print(
    "Classes saved to: "
    "models/class_names.txt"
)
