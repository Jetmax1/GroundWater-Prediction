from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib


# Define Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('model/decision_tree_regressor.joblib')

@app.route('/')
def index():
    return render_template('index.html')


# Define endpoint for testing
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [[float(data['WLM_RPE']), float(data['WLM_GSE']), float(data['RPE_WSE']),
                   float(data['GSE_WSE']), float(data['WSE_QC'])]]
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': prediction})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
