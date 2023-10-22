# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Water.csv")  # Replace with your dataset path

# Create a mapping of full-form to short-form
water_quality_mapping = {'Excellent': 'E', 'Fair': 'F', 'Poor': 'P', 'Good': 'G'}
raw_water_mapping = {'Reservoir': 'R', 'Spring': 'S', 'River': 'R', 'Well': 'W', 'Lake': 'L'}

# Replace values in the original DataFrame
data['Water_Quality_Short'] = data['Water_Quality'].replace(water_quality_mapping)
data['Source_of_Raw_Water_Short'] = data['Source_of_Raw_Water'].replace(raw_water_mapping)

# Create a DataFrame to show full-form and short-form values
full_short_data = data[['Water_Quality', 'Water_Quality_Short', 'Source_of_Raw_Water', 'Source_of_Raw_Water_Short']]

# Display the table with headings
st.write("Abbreviations:")
st.write(full_short_data)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
model.fit(X_train, y_train)


# Function to make predictions
def predict_water_loss(pipe_diameter, distance_miles):
    features = [[pipe_diameter, distance_miles]]
    prediction = model.predict(features)
    return prediction[0]



st.title("NRW Prediction App")



# User input
pipe_diameter = st.slider("Pipe Diameter (inches)", min_value=1, max_value=20, value=10)
distance_miles = st.slider("Distance (miles)", min_value=1, max_value=10, value=5)

# # Visualization: Scatter plot
# st.subheader("Scatter Plot of Pipe Diameter vs. Water Loss Percentage")
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x="Pipe Diameter_inches", y="Water_Loss_Percentage")
# plt.xlabel("Pipe Diameter (inches)")
# plt.ylabel("Water Loss Percentage")
# st.pyplot(plt)

# Prediction
if st.button("Predict"):
    prediction = predict_water_loss(pipe_diameter, distance_miles)
    st.success(f"Predicted NRW: {prediction:.2f} %")


st.markdown("<p style='text-align: center;'>Team: Good Green Group</p>", unsafe_allow_html=True)

