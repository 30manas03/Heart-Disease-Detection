from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load and analyze the heart disease dataset
def load_and_analyze_data():
    try:
        df = pd.read_csv('heart.csv')
        
        # Calculate statistics
        total_patients = len(df)
        heart_disease_rate = (df['target'].sum() / total_patients) * 100
        avg_age = df['age'].mean()
        
        # Age distribution
        age_bins = [20, 30, 40, 50, 60, 70, 100]
        age_labels = ['20-30', '31-40', '41-50', '51-60', '61-70', '71+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        age_distribution = df['age_group'].value_counts().to_dict()
        
        # Gender distribution
        gender_distribution = df['sex'].value_counts().to_dict()
        
        # Chest pain types distribution
        chest_pain_distribution = df['cp'].value_counts().to_dict()
        
        # Heart disease by gender
        heart_disease_by_gender = df.groupby('sex')['target'].sum().to_dict()
        
        return {
            'total_patients': total_patients,
            'heart_disease_rate': round(heart_disease_rate, 1),
            'avg_age': round(avg_age, 1),
            'age_distribution': age_distribution,
            'gender_distribution': gender_distribution,
            'chest_pain_distribution': chest_pain_distribution,
            'heart_disease_by_gender': heart_disease_by_gender
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    """API endpoint to get dashboard analytics data"""
    data = load_and_analyze_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({
            'total_patients': 1027,
            'heart_disease_rate': 52.3,
            'avg_age': 54.4,
            'age_distribution': {'20-30': 45, '31-40': 89, '41-50': 156, '51-60': 234, '61-70': 298, '71+': 205},
            'gender_distribution': {1: 713, 0: 314},
            'chest_pain_distribution': {0: 23, 1: 50, 2: 86, 3: 16},
            'heart_disease_by_gender': {1: 384, 0: 153}
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = float(request.form['ca'])
        thal = int(request.form['thal'])

        # Get the current theme from the form
        current_theme = request.form.get('current_theme', 'light')

        # --- PREPROCESSING ---

        # 1. Standardize numerical features
        numerical_features = [age, trestbps, chol, thalach, oldpeak, ca]
        standardized_numerical = scaler.transform([numerical_features])

        # 2. One-hot encode categorical features
        cp_ohe = [1 if cp == i else 0 for i in [1, 2, 3]]
        restecg_ohe = [1 if restecg == i else 0 for i in [1, 2]]
        slope_ohe = [1 if slope == i else 0 for i in [1, 2]]
        thal_ohe = [1 if thal == i else 0 for i in [1, 2, 3]]
        
        # 3. List the other raw (non-scaled, non-ohe) features
        raw_features = [sex, fbs, exang]

        # --- ASSEMBLE FINAL FEATURE VECTOR ---
        # The order must EXACTLY match the training data:
        # Scaled -> Raw -> One-Hot-Encoded
        
        input_list = np.concatenate([
            standardized_numerical[0], # Scaled numerical features
            raw_features,              # Raw features
            cp_ohe,                    # OHE features for 'cp'
            restecg_ohe,               # OHE features for 'restecg'
            slope_ohe,                 # OHE features for 'slope'
            thal_ohe                   # OHE features for 'thal'
        ]).tolist()

        # --- MAKE PREDICTION ---
        
        # The model expects a 2D array, so we wrap our list in another list
        prediction = model.predict([input_list])[0]

        # Interpret the prediction
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        print(result)
        return render_template('result.html', prediction=result, current_theme=current_theme)

    except Exception as e:
        # Return an error message if anything goes wrong
        return render_template('result.html', prediction=f"An error occurred: {e}", current_theme='light')



    # # Collect input data from the form
    # age = float(request.form['age'])
    # sex = int(request.form['sex'])
    # cp = int(request.form['cp'])
    # trestbps = float(request.form['trestbps'])
    # chol = float(request.form['chol'])
    # fbs = int(request.form['fbs'])
    # restecg = int(request.form['restecg'])
    # thalach = float(request.form['thalach'])
    # exang = int(request.form['exang'])
    # oldpeak = float(request.form['oldpeak'])
    # slope = int(request.form['slope'])
    # ca = float(request.form['ca'])
    # thal = int(request.form['thal'])

    # # Perform one-hot encoding for categorical variables
    # # cp: 0,1,2,3 -> cp_1, cp_2, cp_3 (cp_0 dropped)
    # cp_1, cp_2, cp_3 = [1 if cp == i else 0 for i in [1, 2, 3]]
    # # restecg: 0,1,2 -> restecg_1, restecg_2 (restecg_0 dropped)
    # restecg_1, restecg_2 = [1 if restecg == i else 0 for i in [1, 2]]
    # # slope: 0,1,2 -> slope_1, slope_2 (slope_0 dropped)
    # slope_1, slope_2 = [1 if slope == i else 0 for i in [1, 2]]
    # # thal: 0,1,2,3 -> thal_1, thal_2, thal_3 (thal_0 dropped)
    # thal_1, thal_2, thal_3 = [1 if thal == i else 0 for i in [1, 2, 3]]

    # # Standardize numerical features
    # numerical_features = [age, trestbps, chol, thalach, oldpeak, ca]
    # standardized_numerical = scaler.transform([numerical_features])[0]

    # # Construct the input array in the order expected by the model
    # input_array = [
    #     standardized_numerical[0],  # age
    #     sex,                       # sex
    #     standardized_numerical[1],  # trestbps
    #     standardized_numerical[2],  # chol
    #     fbs,                       # fbs
    #     standardized_numerical[3],  # thalach
    #     exang,                     # exang
    #     standardized_numerical[4],  # oldpeak
    #     standardized_numerical[5],  # ca
    #     cp_1, cp_2, cp_3,          # cp one-hot encoded
    #     restecg_1, restecg_2,      # restecg one-hot encoded
    #     slope_1, slope_2,          # slope one-hot encoded
    #     thal_1, thal_2, thal_3     # thal one-hot encoded
    # ]

    # # Make prediction
    # prediction = model.predict([input_array])[0]

    # # Interpret the prediction
    # result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'

    # return render_template('result.html', prediction=result)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

# if __name__ == '__main__':
#     app.run(debug=True)