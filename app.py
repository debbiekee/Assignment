import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
# Ensure 'dataset.csv' is in the same folder in your GitHub repo
# @st.cache_data
# def load_data():
#     # Use the column names identified in your file metadata
#     columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
#     df = pd.read_csv('dataset_', names=columns, skiprows=14) # skip metadata headers
#     return df

# df = load_data()

@st.cache_data
def load_data():
    # These names match the @ATTRIBUTE tags in your file 
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    # 'skiprows=38' jumps over the text and starts at the first line of numbers 
    df = pd.read_csv('dataset.csv', names=columns, skiprows=38) 
    return df

df = load_data()

# 2. Build and Train the Model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. Streamlit User Interface
st.title("Diabetes Prediction AI")
st.write("Input patient data to predict the likelihood of diabetes.")

# Create input sliders based on your dataset attributes
pregnancies = st.slider("Pregnancies", 0, 17, 3)
glucose = st.slider("Glucose Level", 0, 200, 117)
bp = st.slider("Blood Pressure", 0, 122, 72)
skin = st.slider("Skin Thickness", 0, 99, 23)
insulin = st.slider("Insulin Level", 0, 846, 30)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.37)
age = st.slider("Age", 21, 81, 29)

# 4. Make Prediction
if st.button("Predict"):
    user_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("The model predicts a high risk of Diabetes.")
    else:
        st.success("The model predicts a low risk of Diabetes.")
