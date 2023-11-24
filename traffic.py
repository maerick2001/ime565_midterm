# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings
warnings.filterwarnings('ignore')

st.title('Traffic Volume Prediction: A Machine Learning App') 

# Display the image
st.image('traffic_image.gif', width = 400)

st.subheader("Utilize our advanced Machine Learning application to predict traffic volume.") 
st.write('Use the following form to get started.')

# Reading the pickle file that we created before 
dt_pickle = open('decision_tree_traffic.pickle', 'rb') 
rf_pickle = open('rf_traffic.pickle','rb')
ada_pickle = open('ada_traffic.pickle','rb')
xgb_pickle = open('xgb_traffic.pickle','rb')
dt = pickle.load(dt_pickle)  
rf = pickle.load(rf_pickle)
ada = pickle.load(ada_pickle)
xgb = pickle.load(xgb_pickle)
dt_pickle.close()
rf_pickle.close()
ada_pickle.close()
xgb_pickle.close()

with st.form('user_inputs'): 
  holiday = st.selectbox('Choose whether today is a designated holiday or not',options=['None','Washingtons Birthday','Veterans Day','Thanksgiving Day','State Fair','New Years Day','Memorial Day','Martin Luther King Jr Day','Labor Day','Independence Day','Columbus Day','Christmas Day'])
  temp = st.number_input('Average temperature in Kelvin', min_value = 0.00, value=297.00)
  rain_1h = st.number_input('Amount in mm of rain that occurred in the hour', min_value = 0.0, value = 0.0)
  snow_1h = st.number_input('Amount in mm of snow that occurred in the hour',min_value = 0.00)
  clouds_all = st.number_input('Percentage of cloud cover',min_value = 0, max_value = 100, step = 1)
  weather_main = st.selectbox('Choose the current weather',options=['Clouds','Clear','Drizzle','Smoke','Squall','Thunderstorm','Haze','Fog','Rain','Mist','Snow'])
  month = st.selectbox('Choose month',options = ['January','February','March','April','May','June','July','August','September','November','December'])
  day_of_week = st.selectbox('Choose day of the week',options=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
  time_of_day = st.selectbox('Choose hour', options=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
  ml_model =st.selectbox('Select Machine Learning Model for Prediction',options=['Decision Tree','Random Forest','AdaBoost','XGBoost'])
  st.write('These ML models exhibited the following predictive performance on the test dataset.')
  
  # model performance data pulled from traffic.ipynb file
  model_perf_data = [['Decision Tree', 0.9417, 488.790], ['Random Forest', 0.94981, 475.40], ['ADABoost', 0.8142, 867.62], ['XGBoost', 0.97498, 429.94]]
  model_perf = pd.DataFrame(model_perf_data, columns=['ML Model', 'R2', 'RMSE'])
  max_r2_index = model_perf['R2'].idxmax()
  min_r2_index = model_perf['R2'].idxmin()
  # Highlight the maximum and minimum R2 rows
  def highlight_max_min_r2(s):
      color = 'lime' if s.name == max_r2_index else ('orange' if s.name == min_r2_index else '')
      return ['background-color: {}'.format(color) for _ in s]
  styled_model_perf = model_perf.style.apply(highlight_max_min_r2, axis=1)
  st.dataframe(styled_model_perf, hide_index=True)
  st.form_submit_button() 

# Convert data to numbers to match original df
day_name_to_number = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
day_of_week = day_name_to_number[day_of_week]

month_name_to_number = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
month = month_name_to_number[month]

original_df = pd.read_csv('Traffic_Volume.csv') # Original data to create ML model

# Extract the month, day of the week, and time of day from the 'date_time' column
original_df['month'] = pd.to_datetime(original_df['date_time']).dt.month
original_df['day_of_week'] = pd.to_datetime(original_df['date_time']).dt.weekday
original_df['time_of_day'] = pd.to_datetime(original_df['date_time']).dt.strftime('%H')

original_df = original_df.drop(columns = ['traffic_volume','weather_description','date_time'])

# Concatenate two dataframes together along rows (axis = 0)
combined_df1 = original_df.copy()
combined_df1.loc[len(combined_df1)] = [holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,month,day_of_week,time_of_day]

# Number of rows in original dataframe
original_rows1 = original_df.shape[0]

# Create dummies for the combined dataframe
combined_df1['day_of_week'] = combined_df1['day_of_week'].astype(str)
combined_df1['month'] = combined_df1['month'].astype(str)
combined_df_encoded1 = pd.get_dummies(combined_df1,columns = ['holiday','weather_main','month','day_of_week','time_of_day'], dummy_na=False)

# Fixing the order of the columns
desired_column_order = [
    'temp',
    'rain_1h',
    'snow_1h',
    'clouds_all',
    'holiday_Christmas Day',
    'holiday_Columbus Day',
    'holiday_Independence Day',
    'holiday_Labor Day',
    'holiday_Martin Luther King Jr Day',
    'holiday_Memorial Day',
    'holiday_New Years Day',
    'holiday_State Fair',
    'holiday_Thanksgiving Day',
    'holiday_Veterans Day',
    'holiday_Washingtons Birthday',
    'weather_main_Clear',
    'weather_main_Clouds',
    'weather_main_Drizzle',
    'weather_main_Fog',
    'weather_main_Haze',
    'weather_main_Mist',
    'weather_main_Rain',
    'weather_main_Smoke',
    'weather_main_Snow',
    'weather_main_Squall',
    'weather_main_Thunderstorm',
    'month_1',
    'month_2',
    'month_3',
    'month_4',
    'month_5',
    'month_6',
    'month_7',
    'month_8',
    'month_9',
    'month_10',
    'month_11',
    'month_12',
    'day_of_week_0',
    'day_of_week_1',
    'day_of_week_2',
    'day_of_week_3',
    'day_of_week_4',
    'day_of_week_5',
    'day_of_week_6',
    'time_of_day_00',
    'time_of_day_01',
    'time_of_day_02',
    'time_of_day_03',
    'time_of_day_04',
    'time_of_day_05',
    'time_of_day_06',
    'time_of_day_07',
    'time_of_day_08',
    'time_of_day_09',
    'time_of_day_10',
    'time_of_day_11',
    'time_of_day_12',
    'time_of_day_13',
    'time_of_day_14',
    'time_of_day_15',
    'time_of_day_16',
    'time_of_day_17',
    'time_of_day_18',
    'time_of_day_19',
    'time_of_day_20',
    'time_of_day_21',
    'time_of_day_22',
    'time_of_day_23'
]

combined_df_encoded1 = combined_df_encoded1[desired_column_order]

# Split data into original and user input dataframes using row index
original_df_encoded1 = combined_df_encoded1[:original_rows1]
input_df_encoded1 = combined_df_encoded1.tail(1)

# Using predict() with new data provided by the user
new_prediction = dt.predict(input_df_encoded1) 
new_prediction2 = rf.predict(input_df_encoded1)
new_prediction3 = ada.predict(input_df_encoded1)
new_prediction4 = xgb.predict(input_df_encoded1)
# Show the predicted cnt on the app
st.subheader("Predicting Traffic Volume")

if ml_model == 'Decision Tree':
    st.markdown('<span style="color:red; font-weight: bold;">Decision Tree Traffic Volume Prediction: %d.</span>' % (new_prediction), unsafe_allow_html=True)
    st.subheader('Feature Importance')
    st.image('dt_traffic_importance.svg')
elif ml_model == 'Random Forest':
    st.markdown('<span style="color:red; font-weight: bold;">Random Forest Traffic Volume Prediction: %d.</span>' % (new_prediction2), unsafe_allow_html=True)
    st.subheader('Feature Importance')
    st.image('rf_traffic_importance.svg')
elif ml_model == 'AdaBoost':
    st.markdown('<span style="color:red; font-weight: bold;">AdaBoost Traffic Volume Prediction: %d.</span>' % (new_prediction3), unsafe_allow_html=True)
    st.subheader('Feature Importance')
    st.image('ada_traffic_importance.svg')
else:
    st.markdown('<span style="color:red; font-weight: bold;">XGBoost Traffic Volume Prediction: %d.</span>' % (new_prediction4), unsafe_allow_html=True)
    st.subheader('Feature Importance')
    st.image('xgb_traffic_importance.svg')
