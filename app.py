import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load AdaBoost Model dynamically relative to script location
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Adaboost.pkl')
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

# Exact 18 features expected by the trained AdaBoost model
FEATURES = [
    {"name": "age", "label": "Age", "type": "number", "min": 10, "max": 100, "default": 18, "step": 1},
    {"name": "gender", "label": "Gender", "type": "select", "options": [("0", "Female"), ("1", "Male"), ("2", "Other")]},
    {"name": "state", "label": "State Code / Region", "type": "number", "min": 0, "max": 50, "default": 1, "step": 1},
    {"name": "family_income", "label": "Family Income ($)", "type": "number", "min": 0, "max": 1000000, "default": 65000, "step": 1000},
    {"name": "high_school_gpa", "label": "High School GPA", "type": "number", "min": 0.0, "max": 4.0, "default": 3.5, "step": 0.01},
    {"name": "sat_score", "label": "SAT Score", "type": "number", "min": 400, "max": 1600, "default": 1200, "step": 10},
    {"name": "act_score", "label": "ACT Score", "type": "number", "min": 1, "max": 36, "default": 26, "step": 1},
    {"name": "attendance_rate", "label": "Attendance Rate (%)", "type": "number", "min": 0, "max": 100, "default": 95, "step": 0.1},
    {"name": "ap_courses", "label": "AP Courses Count", "type": "number", "min": 0, "max": 20, "default": 3, "step": 1},
    {"name": "extracurricular_count", "label": "Extracurriculars", "type": "number", "min": 0, "max": 20, "default": 4, "step": 1},
    {"name": "volunteer_hours", "label": "Volunteer Hours", "type": "number", "min": 0, "max": 1000, "default": 50, "step": 1},
    {"name": "leadership_positions", "label": "Leadership Positions", "type": "number", "min": 0, "max": 10, "default": 1, "step": 1},
    {"name": "coding_projects", "label": "Coding Projects", "type": "number", "min": 0, "max": 50, "default": 2, "step": 1},
    {"name": "social_media_hours", "label": "Social Media Hours/Day", "type": "number", "min": 0, "max": 24, "default": 2, "step": 0.5},
    {"name": "online_certifications", "label": "Online Certifications", "type": "number", "min": 0, "max": 20, "default": 1, "step": 1},
    {"name": "essay_score", "label": "Essay Score (0-100)", "type": "number", "min": 0, "max": 100, "default": 85, "step": 1},
    {"name": "recommendation_score", "label": "Recommendation Score (0-100)", "type": "number", "min": 0, "max": 100, "default": 88, "step": 1},
    {"name": "interview_score", "label": "Interview Score (0-100)", "type": "number", "min": 0, "max": 100, "default": 80, "step": 1}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Predictive Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --accent: #38bdf8;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --input-bg: rgba(17, 24, 39, 0.7);
            --input-border: rgba(255, 255, 255, 0.12);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2.5rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 980px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #818cf8, #c084fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .actions {
            margin-top: 2.5rem;
            text-align: center;
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--primary) 0%, #4f46e5 100%);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.9rem 3rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.75rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            display: none;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out forwards;
        }

        .result-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--accent);
        }

        .probability-bar-wrapper {
            margin-top: 1.25rem;
            height: 8px;
            width: 100%;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }

        .probability-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.6s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>AdaBoost Analytics Portal</h1>
        <p>Enter the candidate feature attributes below to generate evaluation predictions.</p>
    </header>

    <form id="predictionForm">
        <div class="grid">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature.name }}">{{ feature.label }}</label>
                {% if feature.type == 'select' %}
                <select id="{{ feature.name }}" name="{{ feature.name }}">
                    {% for val, text in feature.options %}
                    <option value="{{ val }}">{{ text }}</option>
                    {% endfor %}
                </select>
                {% else %}
                <input type="number" 
                       id="{{ feature.name }}" 
                       name="{{ feature.name }}" 
                       min="{{ feature.min }}" 
                       max="{{ feature.max }}" 
                       step="{{ feature.step }}" 
                       value="{{ feature.default }}" 
                       required>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <div class="actions">
            <button type="submit" class="btn-submit">Run Prediction</button>
        </div>
    </form>

    <div id="result" class="result-box">
        <div class="result-title">Prediction Result</div>
        <div id="predictionClass" class="result-value">-</div>
        <div id="probabilityText" style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-muted);"></div>
        <div class="probability-bar-wrapper">
            <div id="probBar" class="probability-bar"></div>
        </div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => data[key] = parseFloat(value));

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultBox = document.getElementById('result');
            
            if (response.ok) {
                resultBox.style.display = 'block';
                document.getElementById('predictionClass').innerText = `Class: ${result.prediction}`;
                
                if (result.probabilities) {
                    const maxProb = Math.max(...result.probabilities);
                    const percentage = (maxProb * 100).toFixed(1);
                    document.getElementById('probabilityText').innerText = `Confidence: ${percentage}%`;
                    document.getElementById('probBar').style.width = `${percentage}%`;
                }
            } else {
                alert(result.error || 'An error occurred during prediction.');
            }
        } catch (err) {
            alert('Failed to connect to the server.');
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file (Adaboost.pkl) not loaded or invalid.'}), 500

    try:
        data = request.get_json()
        input_data = [data.get(f['name'], 0) for f in FEATURES]
        features_array = np.array([input_data])
        
        prediction = model.predict(features_array)[0]
        response = {'prediction': int(prediction)}

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0].tolist()
            response['probabilities'] = probabilities

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
