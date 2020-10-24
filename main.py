from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("model_2.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("predicted")
    if request.method == 'POST':
        #print(request.form.get('sepal_length'))
        try:
            manufacturer = int(request.form['var1']) # taking feature values
            transmission = int(request.form['var2'])
            transmission_type = float(request.form['var3'])
            engine_capacity = float(request.form['var4'])
            fuel_type = int(request.form['var5'])
            urban_metric = float(request.form['var6'])
            extra_urban_metric = float(request.form['var7'])
            combined_metric = float(request.form['var8'])
            urban_imperial = float(request.form['var9'])
            extra_urban_imperial = float(request.form['var10'])
            combined_imperial = float(request.form['var11'])
            noise_level = float(request.form['var12'])
            fuel_cost_10_miles = float(request.form['var13'])
            pred_args = [manufacturer, transmission, transmission_type, engine_capacity,fuel_type,urban_metric,extra_urban_metric,combined_metric,
             urban_imperial,extra_urban_imperial,combined_imperial,noise_level,fuel_cost_10_miles]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction = ml_model.predict(pred_args_arr)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
