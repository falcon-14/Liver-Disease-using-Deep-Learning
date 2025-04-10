# 🩺 Liver Disease By Deep Learning

## 📌 Overview
Liver By Deep Learning is a machine learning project designed to predict liver diseases using deep learning techniques. It utilizes a dataset of liver patient records to train a model capable of assessing liver health.

## ✨ Features
- 🔍 Predicts liver disease based on patient data.
- 🤖 Utilizes a deep learning model for accurate assessment.
- 🌐 Provides a simple interface for making predictions.

## 🛠 Technologies Used
- 🐍 Python
- 🌍 Flask (for web interface)
- 🔬 TensorFlow/Keras (for deep learning model)
- 📊 Pandas & NumPy (for data processing)
- 📈 Scikit-learn (for preprocessing and evaluation)

## 🚀 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/falcon-14/liver-by-deep-learning.git
   cd liver-by-deep-learning
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```

## 🏃 Usage
- 🎛 Run the Flask application and access the web interface.
- 📥 Upload patient data or input values manually.
- 📊 Get predictions on liver disease risk.

## 📂 Dataset
The project uses the **Liver Patient Dataset (LPD)** for training and evaluation.

## 🎯 Model Training
To retrain the model, use:
```sh
python model_train.py
```

## 📁 File Structure
- `app.py` - 🖥 Main application file
- `liver_model.py` - 🧠 Model definition
- `liver_predictor.py` - 🔮 Prediction logic
- `load_model.py` - 📦 Load pre-trained model
- `model_train.py` - 🎓 Training script
- `templates/` - 📝 HTML templates for web interface
- `saved_model/` - 💾 Stored deep learning model
