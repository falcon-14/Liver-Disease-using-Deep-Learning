# app.py
from flask import Flask, request, jsonify, render_template, flash
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Optional

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure Google API
os.environ["GOOGLE_API_KEY"] = "Your_Api_Key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = GenerativeModel('gemini-pro')

@dataclass
class DetoxTask:
    title: str
    description: str
    duration: str
    difficulty: str
    benefits: List[str]

@dataclass
class Recipe:
    name: str
    ingredients: List[str]
    instructions: List[str]
    health_benefits: List[str]
    preparation_time: str
    difficulty: str

# Model directory
MODEL_DIR = 'saved_model'

# Feature name mapping
FEATURE_MAPPING = {
    'gender': 'Gender of the patient',
    'age': 'Age of the patient',
    'total_bilirubin': 'Total Bilirubin',
    'direct_bilirubin': 'Direct Bilirubin',
    'alkaline_phosphotase': '\xa0Alkphos Alkaline Phosphotase',
    'alamine_aminotransferase': '\xa0Sgpt Alamine Aminotransferase',
    'aspartate_aminotransferase': 'Sgot Aspartate Aminotransferase',
    'total_proteins': 'Total Protiens',
    'albumin': '\xa0ALB Albumin',
    'albumin_globulin_ratio': 'A/G Ratio Albumin and Globulin Ratio'
}
def load_saved_model():
    """Load the saved medical prediction model and preprocessors"""
    global liver_model, scaler, label_encoder, feature_names
    
    try:
        liver_model = load_model(os.path.join(MODEL_DIR, 'liver_disease_model.h5'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
        print("Model and preprocessors loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_detox_challenge(prediction_result: str, age: int) -> DetoxTask:
    """Generate personalized detox challenge using Gemini Pro"""
    prompt = f"""
    Create a liver health challenge for a {age}-year-old person who has been diagnosed as {prediction_result}.
    The challenge should include a specific task that promotes liver health.
    Format the response as a structured task with:
    - A catchy title
    - Brief description
    - Duration
    - Difficulty level
    - 3 specific health benefits
    Make it engaging and achievable.
    """
    
    response = model.generate_content(prompt)
    return DetoxTask(
        title=response.text.split('\n')[0],
        description=response.text.split('\n')[1],
        duration="15 minutes",
        difficulty="Medium",
        benefits=["Improved liver function", "Better digestion", "Increased energy"]
    )

def generate_recipe(ingredients: List[str]) -> Recipe:
    """Generate a liver-healthy recipe using Gemini Pro"""
    ingredients_str = ", ".join(ingredients)
    prompt = f"""
    Create a liver-healthy recipe using some or all of these ingredients: {ingredients_str}.
    The recipe should be specifically designed to support liver health.
    Include:
    - Recipe name
    - List of ingredients with quantities
    - Step-by-step instructions
    - Specific liver health benefits
    - Preparation time
    - Difficulty level
    """
    
    response = model.generate_content(prompt)
    return Recipe(
        name="Liver-Healthy Dish",
        ingredients=ingredients,
        instructions=["Step 1", "Step 2", "Step 3"],
        health_benefits=["Benefit 1", "Benefit 2", "Benefit 3"],
        preparation_time="30 minutes",
        difficulty="Easy"
    )

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

@app.context_processor
def utility_processor():
    def format_timestamp(timestamp):
        """Format timestamp for display"""
        return timestamp.strftime("%B %d, %Y at %I:%M %p")
    
    return dict(format_timestamp=format_timestamp)

# Add this to your existing Flask configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['EXPLAIN_TEMPLATE_LOADING'] = True

# Initialize Materialize select fields
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=600'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with additional health recommendations"""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            form_data = {}
            for form_name, feature_name in FEATURE_MAPPING.items():
                form_data[feature_name] = request.form[form_name]
                if form_name != 'gender':
                    form_data[feature_name] = float(form_data[feature_name])
            data = form_data

        input_data = pd.DataFrame([data])
        input_data = input_data[feature_names]
        
        input_data['Gender of the patient'] = label_encoder.transform(
            [input_data['Gender of the patient'].iloc[0]]
        )
        
        input_scaled = scaler.transform(input_data)
        prediction = liver_model.predict(input_scaled)
        probability = float(prediction[0][0])
        result = "Positive (Liver Disease)" if probability < 0.5 else "Negative (Healthy)"
        probability_display = (1 - probability) * 100 if probability < 0.5 else probability * 100

        age = int(data[FEATURE_MAPPING['age']])
        detox_challenge = generate_detox_challenge(result, age)
        
        healthy_ingredients = ["garlic", "turmeric", "leafy greens", "olive oil", "lemon"]
        recipe = generate_recipe(healthy_ingredients)

        if request.is_json:
            return jsonify({
                'prediction': result,
                'probability': probability_display,
                'detox_challenge': vars(detox_challenge),
                'recipe': vars(recipe)
            })
        else:
            return render_template(
                'result.html',
                prediction=result,
                probability=probability_display,
                detox_challenge=detox_challenge,
                recipe=recipe,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
    except Exception as e:
        error_message = str(e)
        if request.is_json:
            return jsonify({'error': error_message}), 400
        else:
            flash(f'Error: {error_message}', 'error')
            return render_template('index.html')

if __name__ == '__main__':
    load_saved_model()
    app.run(debug=True)