# 🌱 AgriVision AI

### Intelligent Plant Disease Detection & Crop Health Monitoring

AgriVision AI is an AI-powered computer vision application designed to detect plant diseases from leaf images and provide useful crop-health insights.

The system uses a deep learning image classification model trained on plant disease data and provides an easy-to-use web interface for instant analysis.

---

## 🚀 Live Demo

🌐 **AgriVision AI**

https://agrivision-ai-7zml.onrender.com

---

## 💡 Project Overview

Plant diseases can significantly affect crop productivity and agricultural outcomes.

AgriVision AI provides a simple way to analyze plant leaf images using Artificial Intelligence.

### Workflow

```text
Plant Leaf Image
       ↓
Image Upload
       ↓
Image Preprocessing
       ↓
Deep Learning Model
       ↓
Disease Classification
       ↓
Confidence Score
       ↓
Crop Health Analysis
       ↓
Symptoms & Care Guidance
```

---

## ✨ Key Features

### 🤖 AI Disease Detection
Upload a plant image and let the deep learning model classify the detected condition.

### 🎯 Confidence Score
Displays the model's prediction confidence for the detected class.

### 🌿 Crop Identification
Identifies the crop associated with the predicted disease class.

### 🔎 Symptom Information
Provides possible symptoms associated with the detected condition.

### 💡 Smart Care Guidance
Provides general recommendations for managing the detected condition.

### 📊 Agriculture Dashboard
The dashboard provides:

- Total analyses
- Healthy plants
- Plants requiring attention
- Average AI confidence
- AI monitoring status

### 📈 Advanced Plant Analytics

Includes:

- Healthy rate
- Attention rate
- Average confidence
- Disease distribution
- Crop analysis
- Plant health distribution

### 📜 Prediction History

Stores recent analysis results locally and allows users to:

- View previous predictions
- View detailed analysis
- Delete individual records
- Clear prediction history

### 📱 Responsive Design

The application is optimized for:

- 📱 Mobile
- 💻 Desktop
- 📟 Tablet

---

## 🧠 AI Model

AgriVision AI uses a transfer-learning based image classification approach.

### Model Architecture

**MobileNetV2**

### Training Approach

```text
PlantVillage Dataset
        ↓
Image Preprocessing
        ↓
Training / Validation Split
        ↓
MobileNetV2
        ↓
Transfer Learning
        ↓
Fine-Tuning
        ↓
Disease Classification
```

### Model Details

- Architecture: MobileNetV2
- Input Size: 224 × 224
- Number of Classes: 39
- Framework: TensorFlow / Keras
- Model Format: `.keras`
- Deployment Format: TensorFlow Lite `.tflite`

---

## 🧪 Supported Disease Classes

The trained model currently supports **39 plant-condition classes** from the training dataset.

The class names are stored in:

```text
models/class_names.txt
```

---

## 🛠️ Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Responsive Web Design

### Backend
- Python
- Flask
- REST API

### Artificial Intelligence
- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning
- Computer Vision

### Image Processing
- Pillow
- NumPy
- OpenCV

### Data & Machine Learning
- Pandas
- Scikit-learn

### Deployment
- Render
- Gunicorn

### Version Control
- Git
- GitHub

---

## 📂 Project Structure

```text
AgriVision-AI/
│
├── app.py
├── requirements.txt
├── Procfile
├── .python-version
├── README.md
│
├── templates/
│   └── index.html
│
├── training/
│   └── train.py
│
└── models/
    ├── agrivision_model.keras
    ├── agrivision_model.tflite
    ├── class_names.txt
    └── .gitkeep
```

---

## ⚙️ API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "project": "AgriVision AI",
  "model_loaded": true,
  "classes": 39
}
```

### Plant Prediction

```http
POST /predict
```

Request:

```text
multipart/form-data
image=<plant-image>
```

Example response:

```json
{
  "success": true,
  "disease": "Apple___Cedar_apple_rust",
  "confidence": 52.13,
  "crop": "Apple",
  "status": "Needs Attention",
  "message": "Plant image analyzed successfully."
}
```

---

## 🖥️ Local Installation

### 1. Clone Repository

```bash
git clone https://github.com/priyanshu18611/AgriVision-AI.git
```

### 2. Open Project

```bash
cd AgriVision-AI
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Application

```bash
python app.py
```

Application:

```text
http://127.0.0.1:5000
```

---

## ☁️ Deployment

AgriVision AI is deployed using Render.

### Production Server

```text
Gunicorn
```

### Start Command

```bash
gunicorn --timeout 120 --workers 1 app:app
```

---

## 🔐 Important Note

AgriVision AI is intended as a **screening and educational assistance tool**.

AI predictions are not guaranteed agricultural diagnoses.

For important crop-health decisions, users should verify the result with a qualified agricultural professional.

---

## 📊 Current Capabilities

| Feature | Status |
|---|---|
| AI Model | ✅ |
| 39 Classes | ✅ |
| Image Upload | ✅ |
| Disease Prediction | ✅ |
| Confidence Score | ✅ |
| Crop Detection | ✅ |
| Symptoms | ✅ |
| Care Recommendations | ✅ |
| Dashboard | ✅ |
| Advanced Analytics | ✅ |
| Prediction History | ✅ |
| Responsive UI | ✅ |
| REST API | ✅ |
| Render Deployment | ✅ |
| TensorFlow Lite Model | ✅ |

---

## 🔮 Future Improvements

Possible future upgrades include:

- 📸 Camera-based plant scanning
- 🌦️ Weather integration
- 🌍 GPS-based agricultural insights
- 🗺️ Crop health mapping
- 👤 User authentication
- ☁️ Cloud database
- 📄 PDF diagnosis reports
- 📊 Advanced analytics
- 🔔 Disease alerts
- 🌐 Multilingual support
- 📱 Progressive Web App
- 🤖 AI agricultural assistant
- 📷 Real-time camera detection

---

## 🎯 Project Objective

The goal of AgriVision AI is to demonstrate how Artificial Intelligence and Computer Vision can be applied to agriculture to assist with early plant-disease identification and crop-health monitoring.

---

## 👨‍💻 Developer

### Priyanshu Kumar

**B.Tech – Computer Science & Engineering**

Interested in:

- Software Engineering
- Full Stack Development
- Data Analytics
- Artificial Intelligence
- Machine Learning
- Computer Vision

---

## 🔗 Connect

GitHub:

https://github.com/priyanshu18611

LinkedIn:

https://www.linkedin.com/in/priyanshuroy18/

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🌱 AgriVision AI

**AI • Computer Vision • Agriculture • Innovation**
