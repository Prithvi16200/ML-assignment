# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load the data
data = pd.read_csv('diabetes.csv')

# Step 2: Exploratory Data Analysis (EDA)
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Step 3: Data Cleaning
# Handling missing values - assuming '0' values in certain columns are to be treated as missing
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

# Impute missing values with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
data[columns_with_zeros] = imputer.fit_transform(data[columns_with_zeros])

# Step 4: Feature Engineering
# If there are any categorical variables, encode them
# Assuming there are no categorical variables in this dataset, skip this step

# Step 5: Scaling and Normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Outcome', axis=1))

# Combine scaled data with the target variable
data_preprocessed = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_preprocessed['Outcome'] = data['Outcome']

# Step 6: Splitting the data into training and testing sets
X = data_preprocessed.drop('Outcome', axis=1)
y = data_preprocessed['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# The data is now preprocessed and ready for modeling
