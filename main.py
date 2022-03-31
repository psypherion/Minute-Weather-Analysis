from numpy import array
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import streamlit as st
from warnings import filterwarnings
filterwarnings('ignore')

temp_model = r'models/temp_model.sav'
humid_model = r'models/humid_model.sav'
pred_temp = pickle.load(open(temp_model, 'rb'))
pred_humid = pickle.load(open(humid_model, 'rb'))

temp_scaler_attr = {'mean_': array([9.16829300e+02, 1.62029069e+02, 2.77375705e+00, 1.63463955e+02,
                                    3.39882273e+00, 1.66862985e+02, 2.13290007e+00, 1.24790287e-03,
                                    5.07305306e-01, 4.76037205e+01]),
                    'scale_': array([3.04994532, 95.20905515,  2.06063396, 92.38005076,  2.42261699,
                                     97.45225696,  1.7453824,  0.7676629, 76.96323774, 26.21002362]),
                    'var_': array([9.30216646e+00, 9.06476418e+03, 4.24621233e+00, 8.53407378e+03,
                                   5.86907307e+00, 9.49694239e+03, 3.04635973e+00, 5.89306321e-01,
                                   5.92333996e+03, 6.86965338e+02]),
                    'n_samples_seen_': 1428140,
                    'n_features_in_': 10,
                    'feature_names_in_': array(['air_pressure', 'avg_wind_direction', 'avg_wind_speed',
                                                'max_wind_direction', 'max_wind_speed', 'min_wind_direction',
                                                'min_wind_speed', 'rain_accumulation', 'rain_duration',
                                                'relative_humidity'], dtype=object)}

humid_scaler_attr = {'mean_': array([9.16829300e+02, 1.62029069e+02, 2.77375705e+00, 6.18577438e+01,
                               1.63463955e+02, 3.39882273e+00, 1.66862985e+02, 2.13290007e+00,
                               1.24790287e-03, 5.07305306e-01]),
               'scale_': array([3.04994532, 95.20905515,  2.06063396, 11.83327558, 92.38005076,
                                2.42261699, 97.45225696,  1.7453824,  0.7676629, 76.96323774]),
               'var_': array([9.30216646e+00, 9.06476418e+03, 4.24621233e+00, 1.40026411e+02,
                              8.53407378e+03, 5.86907307e+00, 9.49694239e+03, 3.04635973e+00,
                              5.89306321e-01, 5.92333996e+03]),
               'n_samples_seen_': 1428140,
               'n_features_in_': 10,
               'feature_names_in_': array(['air_pressure', 'avg_wind_direction', 'avg_wind_speed', 'air_temp',
                                           'max_wind_direction', 'max_wind_speed', 'min_wind_direction',
                                           'min_wind_speed', 'rain_accumulation', 'rain_duration'],
                                          dtype=object)}

temp_scaler = StandardScaler()
for i in temp_scaler_attr:
    temp_scaler.__setattr__(i, temp_scaler_attr[i])
    
humid_scaler = StandardScaler()
for i in humid_scaler_attr:
    humid_scaler.__setattr__(i, humid_scaler_attr[i])

selectbox = st.sidebar.selectbox(
    "Choose What you want to Predict : ",
    ["Homepage", "Air Temperature", "Relative Humidity"]
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
if selectbox == "Homepage":
    st.image('src\images\weather_forcasting.jpg')
    st.title("Weather Prediction")
    st.write("""
    This app predicts the air temperature & Relative Humidity based on the given inputs.
    """)
    st.write("""
    The inputs are as follows :
    """)
    st.write("""
    Air Pressure : The air pressure in millibars.
    """)
    st.write("""
    Air Temperature : The air temperature in degrees Celsius.
    """)
    st.write("""
    Average Wind Direction : The average wind direction in degrees.
    """)
    st.write("""
    Average Wind Speed : The average wind speed in kilometers per hour.
    """)
    st.write("""
    Max Wind Direction : The maximum wind direction in degrees.
    """)
    st.write("""
    Max Wind Speed : The maximum wind speed in kilometers per hour.
    """)
    st.write("""
    Min Wind Direction : The minimum wind direction in degrees.
    """)
    st.write("""
    Min Wind Speed : The minimum wind speed in kilometers per hour.
    """)
    st.write("""
    Rain Accumulation : The amount of rain accumulated in the last 3 hours in millimeters.
    """)
    st.write("""
    Rain Duration : The duration of the last rain in hours.
    """)
    st.write("""
    Relative Humidity : The relative humidity in percent.
    """)
    st.write("""
    The output is as follows :
    """)
    st.title("""
    The predicted air temperature can be :
    """)
    st.write("""
    Relatively Cooler : The air temperature is cooler than normal.
    """)
    st.write("""
    Relatively Hotter : The air temperature is hotter than normal.
    """)
    st.title("""
             The predicted relative humidity can be :
             """)
    st.write("""
             Dry Day : The relative humidity is less than normal.
             """)
    st.write("""
             Humid Day : The relative humidity is more than normal.
             """)
elif selectbox == "Air Temperature":
    st.image('src\images\weatherman.jpg')
    st.title("Air Temperature Prediction")
    air_press = st.number_input("Air Pressure : ")
    avg_wind_dir = st.number_input("Average Wind Direction : ")
    avg_wind_speed = st.number_input("Average Wind Speed : ")
    max_wind_dir = st.number_input("Max Wind Direction : ")
    max_wind_speed = st.number_input("Max Wind Speed : ")
    min_wind_dir = st.number_input("Min. Wind Direction : ")
    min_wind_speed = st.number_input("Min. wind Speed : ")
    rain_acc = st.number_input("Rain Accumulation : ")
    rain_dur = st.number_input("Rain Duration : ")
    rel_humid = st.number_input("Relative Humidity : ")

    ip_array = np.array([air_press, avg_wind_dir, avg_wind_speed, max_wind_dir,
                         max_wind_speed, min_wind_dir, min_wind_speed, rain_acc, rain_dur, rel_humid])
    ip_array = ip_array.reshape(1, -1)
    scaled_ip = temp_scaler.transform(ip_array)
    
    result = ""
    predicted_temp = pred_temp.predict(X=scaled_ip)
    if predicted_temp == 0:
        result = "Relatively Cooler"
    elif predicted_temp == 1:
        result = "Hotter than Normal"
    # print(result)
    if(st.button('Submit')):    
        if len(result) > 0:
            st.success(result)
        else:
            st.error("Please enter All Values & Press Submit")
    
elif selectbox == "Relative Humidity":
    st.image('src\images\weatherman.jpg')
    st.title("Weather Humidity Prediction")
    air_press = st.number_input("Air Pressure : ")
    avg_wind_direction = st.number_input("Average Wind Direction : ")
    avg_wind_speed = st.number_input("Average Wind Speed : ")
    air_temp = st.number_input("Air Temperature : ")
    max_wind_direction = st.number_input("Max Wind Direction : ")
    max_wind_speed = st.number_input("Max Wind Speed : ")
    min_wind_direction = st.number_input("Min. Wind Direction : ")
    min_wind_speed = st.number_input("Min. wind Speed : ")
    rain_accumulation = st.number_input("Rain Accumulation : ")
    rain_duration = st.number_input("Rain Duration : ")
    
    ip_array = np.array([air_press,avg_wind_direction, avg_wind_speed,  air_temp, max_wind_direction,
                         max_wind_speed, min_wind_direction, min_wind_speed, rain_accumulation, rain_duration])
    ip_array = ip_array.reshape(1, -1)
    scaled_ip = humid_scaler.transform(ip_array)
    predicted_humidity = pred_humid.predict(X=scaled_ip)
    result = ""
    if predicted_humidity == 0:
        result = "Relatively Dry"
    elif predicted_humidity == 1:
        result = "Highly Humid"
    if(st.button('Submit')):    
        if len(result) > 0:
            st.success(result)
        else : 
            st.error("Please enter All Values & Press Submit")
        
