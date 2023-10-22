# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with your dataset path)
data = pd.read_csv("Water.csv")  # Replace with your dataset path

# Removing outliers using the IQR method
Q1 = data['Water_Loss_Percentage'].quantile(0.25)
Q3 = data['Water_Loss_Percentage'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers = data[(data['Water_Loss_Percentage'] >= lower_bound) & (data['Water_Loss_Percentage'] <= upper_bound)]
# Load your trained Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of trees as needed

# Train the model on the training data
model.fit(X_train, y_train)  # This line fits the model
# Function to make predictions
def predict_water_loss(pipe_diameter, distance_miles):
    features = [[pipe_diameter, distance_miles]]
    prediction = model.predict(features)
    return prediction[0]

# Create a Streamlit app
st.title("Water Loss Percentage Prediction App")

# User input
pipe_diameter = st.slider("Pipe Diameter (inches)", min_value=1, max_value=20, value=10)
distance_miles = st.slider("Distance (miles)", min_value=1, max_value=10, value=5)

# Visualization: Scatter plot
st.subheader("Scatter Plot of Pipe Diameter vs. Water Loss Percentage")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Pipe Diameter_inches", y="Water_Loss_Percentage")
plt.xlabel("Pipe Diameter (inches)")
plt.ylabel("Water Loss Percentage")
st.pyplot(plt)

# Prediction
if st.button("Predict"):
    prediction = predict_water_loss(pipe_diameter, distance_miles)
    st.success(f"Predicted Water Loss Percentage: {prediction:.2f}")
