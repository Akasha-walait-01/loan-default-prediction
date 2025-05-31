from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Dummy ML model simulation for now
def predict_loan(features):
    # Replace with your real model prediction
    return "Approved" if features['credit_history'] == '1' else "Rejected"

# Store history temporarily
history = []

@app.route('/')
def home():
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    form = request.form
    result = predict_loan(form)

    # Store history
    history.append({
        'gender': form['gender'],
        'married': form['married'],
        'dependents': form['dependents'],
        'education': form['education'],
        'self_employed': form['self_employed'],
        'total_income': int(form['applicant_income']) + int(form['coapplicant_income']),
        'loan_amount': form['loan_amount'],
        'loan_term': form['loan_term'],
        'credit_history': form['credit_history'],
        'property_area': form['property_area'],
        'prediction': result
    })

    return render_template('result.html', prediction_text=result)

@app.route('/history')
def view_history():
    return render_template('history.html', data=history)

if __name__ == '__main__':
    app.run(debug=True)
