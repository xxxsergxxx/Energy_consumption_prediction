import streamlit as st
import pandas as pd
import datetime
from datetime import datetime, time
from sklearn.preprocessing import PolynomialFeatures
import numpy as np 
import pandas as pd 
import math
from joblib import dump, load
import xgboost as xgb

new_model = load("xgb_model_bez_consump(888).sav")
timing_model = load("xgb_model_by_time_usage.sav")
CO2_model = load("xgb_model_CO2.sav")

#Define WeekStatus
def WeekStatus(weekday):
    '''This is function define days for weekstatus feauture'''
    return 1 if weekday<5 else 0

def energy_predict ():
    '''Predicting function'''

    dict_feautures = {'WeekStatus': [WeekStatus(weekday)], 'Day_of_week':[weekday], 'day':[date_input.day], 
                        'month':[date_input.month], 'mean_temp':[temp_input], 'workday':[workday_input]                         
                        }
    X_for_prediction = pd.DataFrame(dict_feautures)
   
    

    poly = PolynomialFeatures(7)
    X_poly = poly.fit_transform(X_for_prediction)

    
    pred = new_model.predict(X_poly)
    
    return pred


def energy_predict_timing (prd):
    '''Predicting function'''

    time_prd = []
    for i in range(1, 97):
        dict_feautures = {'WeekStatus': [WeekStatus(weekday)], 'Day_of_week':[weekday], 'day':[date_input.day],
                            'month':[date_input.month], 'time': [i], 'mean_temp':[temp_input], 
                            'day_usage' : [prd] ,'workday':[workday_input]
                            }
        
        X_for_prediction = pd.DataFrame(dict_feautures)
    
        poly = PolynomialFeatures(7)
        X_poly = poly.fit_transform(X_for_prediction)
        pred_timing = timing_model.predict(X_poly)
        time_prd.append(abs(pred_timing))
    return time_prd

def co2_predict (time_prd):
    '''Predicting function'''
    print("OUR TYPE OF PRD::: /n", time_prd)
    co2_prd = []
    for i in range(1, 97):
        dict_feautures = {'Usage_kWh':[time_prd[i-1]], 'WeekStatus': [WeekStatus(weekday)], 
                          'Day_of_week':[weekday], 'day':[date_input.day],
                            'month':[date_input.month], 'time': [i], 'mean_temp':[temp_input], 
                            'day_usage' : [prd] ,'workday':[workday_input]
                            }
        
        X_for_prediction = pd.DataFrame(dict_feautures).astype('float')
        print("OUR FEATURES IS: \n\n", X_for_prediction)
        print('SHAPE IS:', X_for_prediction.shape)

        
        pred_CO2 = CO2_model.predict(X_for_prediction)
        co2_prd.append(abs(pred_CO2))
    return co2_prd

st.title("""
Monytoring and Prediction of Energy Consumption in Industry
""")
df0 = pd.read_pickle(r'data0.csv')
st.bar_chart(data=df0, x='month', y='Usage_kWh', color='#FFAA00', width=0, height=0, use_container_width=True)
df = pd.read_pickle(r'data.csv')


# Get date input
date_input = st.sidebar.date_input("Select Date")
weekday = date_input.weekday()



# New Get temperature input
temp_input = st.sidebar.number_input("Enter temperature",-5, 50)

# New Get workday input
workday_input = st.sidebar.checkbox("Work Day")

#Process button
if st.sidebar.button("Process"):
    prd = energy_predict()
    time_prd = energy_predict_timing (prd)
    
    print("our prd is :", prd)
    st.title(f"day consumption is : {str(*prd)}")
    st.title('kWh Usage by minutes')
    st.line_chart(data=energy_predict_timing(prd),  color='#FFAA00', width=0, height=0, use_container_width=True)
    print(energy_predict_timing(prd))
    CO2 = co2_predict(time_prd)
    st.title(f"day exhaust CO2 : {str(*sum(CO2))}")
    st.title('CO2 by time')
    st.line_chart(data=CO2,  color='#FFAA00', width=0, height=0, use_container_width=True)
    print(co2_predict(time_prd))
