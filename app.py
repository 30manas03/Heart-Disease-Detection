from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # Perform one-hot encoding for categorical variables
    # cp: 0,1,2,3 -> cp_1, cp_2, cp_3 (cp_0 dropped)
    cp_1, cp_2, cp_3 = [1 if cp == i else 0 for i in [1, 2, 3]]
    # restecg: 0,1,2 -> restecg_1, restecg_2 (restecg_0 dropped)
    restecg_1, restecg_2 = [1 if restecg == i else 0 for i in [1, 2]]
    # slope: 0,1,2 -> slope_1, slope_2 (slope_0 dropped)
    slope_1, slope_2 = [1 if slope == i else 0 for i in [1, 2]]
    # thal: 0,1,2,3 -> thal_1, thal_2, thal_3 (thal_0 dropped)
    thal_1, thal_2, thal_3 = [1 if thal == i else 0 for i in [1, 2, 3]]

    # Standardize numerical features
    numerical_features = [age, trestbps, chol, thalach, oldpeak, ca]
    standardized_numerical = scaler.transform([numerical_features])[0]

    # Construct the input array in the order expected by the model
    input_array = [
        standardized_numerical[0],  # age
        sex,                       # sex
        standardized_numerical[1],  # trestbps
        standardized_numerical[2],  # chol
        fbs,                       # fbs
        standardized_numerical[3],  # thalach
        exang,                     # exang
        standardized_numerical[4],  # oldpeak
        standardized_numerical[5],  # ca
        cp_1, cp_2, cp_3,          # cp one-hot encoded
        restecg_1, restecg_2,      # restecg one-hot encoded
        slope_1, slope_2,          # slope one-hot encoded
        thal_1, thal_2, thal_3     # thal one-hot encoded
    ]

    # Make prediction
    prediction = model.predict([input_array])[0]

    # Interpret the prediction
    result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)