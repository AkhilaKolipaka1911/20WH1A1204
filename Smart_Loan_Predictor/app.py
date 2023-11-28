from flask import Flask, render_template, request,url_for,redirect
import pickle
import numpy as np

app = Flask(__name__, template_folder='template', static_folder='static')

# Load the trained model
model = pickle.load(open('gb.pkl', 'rb'))

# Mapping for categorical variables
def map_education(education):
    if education == "Bachelor":
        return 1
    elif education == "Master":
        return 2
    elif education == "Advanced/Professional":
        return 3

def map_boolean(value):
    return 1 if value == "Yes" else 0

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None  # Initialize prediction variable

    if request.method == 'POST':
        # Get user inputs from the form
        age = int(request.form['age'])
        income = int(request.form['income'])
        zipcode = int(request.form['zipcode'])
        family = int(request.form['family'])
        cc_avg = float(request.form['cc_avg'])
        education = map_education(request.form['education'])
        mortgage = int(request.form['mortgage'])
        securities_account = map_boolean(request.form['securities_account'])
        cd_account = map_boolean(request.form['cd_account'])
        online = map_boolean(request.form['online'])
        credit_card = map_boolean(request.form['credit_card'])

        # Preprocess the data and make prediction
        arr = np.array([[age, income, zipcode, cc_avg, mortgage, family, education,
                         securities_account, cd_account, online, credit_card]])
        
        prediction = model.predict(arr)[0]

        # Render the result page with the prediction
        if prediction == 1:
            return redirect(url_for('la'))
        else:
            return redirect(url_for('lr'))

    # Return the predict.html page for GET requests
    return render_template('predict.html')

@app.route('/la')
def la():
    return render_template("LA.html")

@app.route('/lr')
def lr():
    return render_template("LR.html")

if __name__ == "__main__":
    app.run(debug=True)
