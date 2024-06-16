from flask import Flask,render_template, request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__)


model=pickle.load(open('LinearR_model.pkl','rb'))
car_data=pd.read_csv('Cleaned_car.csv')

@app.route('/')
def index():
    companies=sorted(car_data['company'].unique())
    car_model=sorted(car_data['name'].unique())
    year=sorted(car_data['year'].unique(),reverse=True)
    fuel_type=sorted(car_data['fuel_type'].unique())
    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies,car_model=car_model,years=year,fuel_type=fuel_type)


@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('model')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel')
    km_driven=request.form.get('km_d')
    print(company,car_model,year,fuel_type,km_driven)
    
    prediction=model.predict(pd.DataFrame([[car_model,company,year,km_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    # print()
    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
    
    
