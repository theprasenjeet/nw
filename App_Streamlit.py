# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("Water.csv")  # Replace with your dataset path

# Removing outliers using the IQR method
Q1 = data['Water_Loss_Percentage'].quantile(0.25)
Q3 = data['Water_Loss_Percentage'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Water_Loss_Percentage'] >= lower_bound) & (data['Water_Loss_Percentage'] <= upper_bound)]

data['Water_Quality'] = data['Water_Quality'].replace({'Excellent': 'E', 'Fair': 'F', 'Poor': 'P', 'Good': 'G'})
# Convert the column values to lowercase before replacing
data['Source_of_Raw_Water'] = data['Source_of_Raw_Water'].str.lower()
data['Source_of_Raw_Water'] = data['Source_of_Raw_Water'].replace({'reservoir': 'R', 'spring': 'S', 'river': 'R', 'well': 'W', 'lake': 'L'})

# Optionally, you can convert them back to uppercase
data['Source_of_Raw_Water'] = data['Source_of_Raw_Water'].str.upper()

st.title("Visualization of Relationships")
# Create a mapping of full-form to short-form
water_quality_mapping = {'Excellent': 'E', 'Fair': 'F', 'Poor': 'P', 'Good': 'G'}
raw_water_mapping = {'Reservoir': 'R', 'Spring': 'S', 'River': 'R', 'Well': 'W', 'Lake': 'L'}

# Create a mapping of full-form to short-form
water_quality_mapping = {'Excellent': 'E', 'Fair': 'F', 'Poor': 'P', 'Good': 'G'}
raw_water_mapping = {'Reservoir': 'R', 'Spring': 'S', 'River': 'R', 'Well': 'W', 'Lake': 'L'}

# Create a DataFrame to show the mapping
mapping_data = pd.DataFrame({
    'Full-Form': list(water_quality_mapping.keys()) + list(raw_water_mapping.keys()),
    'Short-Form': list(water_quality_mapping.values()) + list(raw_water_mapping.values())
})
# Display the table with headings
st.write("Abbreviations:")
st.write(mapping_data)


# User input for selecting columns to visualize
columns_to_visualize = st.multiselect("Select columns to visualize:", data.columns)

if len(columns_to_visualize) >= 2:
    # Create pair plots for selected columns with custom figsize
    plt.figure(figsize=(10, 8))  # Specify your desired figsize
    pair_plot = sns.pairplot(data=data, vars=columns_to_visualize[:2])
    st.pyplot(pair_plot)
else:
    st.warning("Please select at least two columns for visualization.")



# Split the data into training and testing sets
X = data[['Pipe Diameter_inches', 'Distance_miles']]
y = data['Water_Loss_Percentage']
y_log = np.log1p(y)  # Using log1p to avoid division by zero
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
model.fit(X_train, y_train)
# Calculate the R-squared (accuracy) for the model
accuracy = model.score(X_test, y_test)

# Function to make predictions and reverse the transformation
def predict_water_loss(pipe_diameter, distance_miles):
    features = [[pipe_diameter, distance_miles]]
    prediction_log = model.predict(features)
    # Reverse the log transformation
    prediction = np.expm1(prediction_log)
    return prediction[0]



st.title("NRW Prediction App")



# User input
pipe_diameter = st.slider("Pipe Diameter (inches)", min_value=1, max_value=12, value=10)
distance_miles = st.slider("Distance (miles)", min_value=1, max_value=6, value=5)



# Prediction
if st.button("Predict"):
    prediction = predict_water_loss(pipe_diameter, distance_miles)
    st.success(f"Predicted NRW: {prediction:.2f} %")


st.markdown("<p style='text-align: center;'>Team: Good Green Group</p>", unsafe_allow_html=True)

