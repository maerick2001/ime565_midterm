# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Fetal Health Classification: A Machine Learning App') 

# Display the image
st.image('fetal_health_image.gif', width = 400)

st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.") 

# Reading the pickle file that we created before 
rf_pickle = open('health.pickle', 'rb') 
rf = pickle.load(rf_pickle)  
rf_pickle.close()

# Display an example dataset and prompt the user 
# to submit the data in the required format.
st.write("Please ensure that your data adheres to this specific format:")

# Cache the dataframe so it's only loaded once
@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  df = df[df.columns[:-1]]
  return df

data_format = load_data('fetal_health.csv')
st.dataframe(data_format.head(5), hide_index = True)

health_file = st.file_uploader('Upload your data in csv format.')
if health_file is not None:
    user_df = pd.read_csv(health_file) # User provided file
else :
    st.stop()

# Loading data
original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model


required_columns = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                    'percentage_of_time_with_abnormal_long_term_variability',
                    'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
                    'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                    'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
                    'histogram_tendency']

missing_columns = set(required_columns) - set(user_df.columns)

if missing_columns:
    st.write(f'Missing columns in user data: {missing_columns}')
else:
    # Select only the relevant features for prediction
    user_features = user_df[required_columns]

    # Make predictions with the trained model
    new_pred = rf.predict(user_features)
    new_pred_prob = rf.predict_proba(user_features).max()


# Show the prediction
user_df['Predicted Fetal Health'] = new_pred
user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].replace({1.0: 'Normal', 0.0: 'Suspect', 2.0: 'Pathological'})
user_df['Prediction Probability (%)'] = new_pred_prob

def highlight_fetal_health_status(val):
    if val == 'Normal':
        return 'background-color: lime'
    elif val == 'Suspect':
        return 'background-color: yellow'
    elif val == 'Pathological':
        return 'background-color: orange'
    else:
        return ''
  
styled_df = user_df.style.applymap(highlight_fetal_health_status, subset=['Predicted Fetal Health'])
st.subheader("Predicting Fetal Health Class")
st.dataframe(styled_df)

# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance","Confusion Matrix","Classification Report"])

with tab1:
  st.image('importance.png')
with tab2:
  st.image('cf_matrix.png')
with tab3:
  df = pd.read_csv('class_report.csv', index_col=0)
  st.dataframe(df)
