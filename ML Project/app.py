import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np      
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## import the ridge model and  standard scaler pickle files
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# route for home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])
        
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # scale the input data
        scaled_data = scaler.transform(input_data)
        
        # predict using the ridge model
        result = ridge_model.predict(scaled_data)
        
        return render_template('home.html', result=(result[0]))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
    