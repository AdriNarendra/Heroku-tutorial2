import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('simple_lg.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term']
    int_features = [int(x) for x in request.form.values()]
    scaled_features = scaler.transform(np.array(int_features).reshape(1,-1))
    final_features = np.array(scaled_features)
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'Approved'
    else:
        output = 'Rejected'

    return render_template('index.html', prediction_text='Loan is {}'.format(output))


if __name__ == "__main__":
    # port-int(os.environ.get('PORT',5000))
    app.run(debug=True)