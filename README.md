import pandas as pd
import matplotlib.pyplot as plt

# To load the dataset
file_path = ("C:/Users/HP SPECTER X360/Downloads/energy_dataset_.csv")
energy_data = pd.read_csv(file_path)

# To display the first few rows to inspect the data
print(energy_data.head())

# To check for missing values
print(energy_data.isnull().sum())

# To drop rows with missing values (if any)
energy_data = energy_data.dropna()

# A Scatter plot to show the relationship between Energy Production vs Installed Capacity
plt.figure(figsize=(10, 6))
plt.scatter(energy_data['Installed_Capacity_MW'], energy_data['Energy_Production_MWh'], alpha=0.6)
plt.title('Energy Production vs. Installed Capacity')
plt.xlabel('Installed Capacity (MW)')
plt.ylabel('Energy Production (MWh)')
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression

# To prepare the data for analysis
X = energy_data[['Installed_Capacity_MW']]  # Independent variable
y = energy_data['Energy_Production_MWh']   # Dependent variable

# To create a linear regression model
model = LinearRegression()
model.fit(X, y)

# To get the model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# To predict energy production based on the installed capacity
y_pred = model.predict(X)

# To plot the regression line along with the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Energy Production vs. Installed Capacity')
plt.xlabel('Installed Capacity (MW)')
plt.ylabel('Energy Production (MWh)')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import r2_score

# To calculate R-squared
r_squared = r2_score(y, y_pred)
print(f"R-squared: {r_squared}")

# Preparing the features (independent variables)
X_extended = energy_data[['Installed_Capacity_MW', 'Energy_Storage_Capacity_MWh', 'Storage_Efficiency_Percentage', 'Grid_Integration_Level']]

# We are using the same dependent variable (energy production)
y = energy_data['Energy_Production_MWh']

# Creating and training the linear regression model
model_extended = LinearRegression()
model_extended.fit(X_extended, y)

# GTo gt the model coefficients and intercept
print(f"Intercept: {model_extended.intercept_}")
print(f"Coefficients: {model_extended.coef_}")

# To make predictions using the extended model
y_pred_extended = model_extended.predict(X_extended)

# Evaluating the model using R-squared
r_squared_extended = r2_score(y, y_pred_extended)
print(f"R-squared for extended model: {r_squared_extended}")

# To plot the actual vs predicted energy production
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_extended, alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # The ideal line (y = x)
plt.title('Actual vs. Predicted Energy Production (Extended Model)')
plt.xlabel('Actual Energy Production (MWh)')
plt.ylabel('Predicted Energy Production (MWh)')
plt.grid(True)
plt.show()

# Preparing the data for predicting Energy Consumption
X_consumption = energy_data[['Storage_Efficiency_Percentage', 'Grid_Integration_Level']]
y_consumption = energy_data['Energy_Consumption_MWh']

# To train a regression model
model_consumption = LinearRegression()
model_consumption.fit(X_consumption, y_consumption)

# Predictions and performance
y_pred_consumption = model_consumption.predict(X_consumption)
r_squared_consumption = r2_score(y_consumption, y_pred_consumption)
print(f"R-squared for Energy Consumption model: {r_squared_consumption}")

# Plotting Actual vs Predicted Energy Consumption
plt.figure(figsize=(10, 6))
plt.scatter(y_consumption, y_pred_consumption, alpha=0.6)
plt.plot([min(y_consumption), max(y_consumption)], [min(y_consumption), max(y_consumption)], color='red', linestyle='--')
plt.title('Actual vs Predicted Energy Consumption (Storage & Grid Integration)')
plt.xlabel('Actual Energy Consumption (MWh)')
plt.ylabel('Predicted Energy Consumption (MWh)')
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Selecting features and target variable
X = energy_data[['Installed_Capacity_MW', 'Energy_Storage_Capacity_MWh', 'Storage_Efficiency_Percentage', 'Grid_Integration_Level']]
y = energy_data['Energy_Production_MWh']

# To split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# To create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# To train the model on the training data
rf_model.fit(X_train, y_train)

# To make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

# To calculate R-squared and MAE
r_squared_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"R-squared: {r_squared_rf}")
print(f"Mean Absolute Error: {mae_rf}")

import numpy as np

# To get the feature importances from the Random Forest model
feature_importances = rf_model.feature_importances_

# To plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.title('Feature Importance for Energy Production Prediction')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# To select features and target variable for energy consumption prediction
X_consumption = energy_data[['Storage_Efficiency_Percentage', 'Grid_Integration_Level']]
y_consumption = energy_data['Energy_Consumption_MWh']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train_consumption, X_test_consumption, y_train_consumption, y_test_consumption = train_test_split(X_consumption, y_consumption, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train_consumption)}, Test set size: {len(X_test_consumption)}")

# To create a Random Forest Regressor model for energy consumption prediction
rf_model_consumption = RandomForestRegressor(n_estimators=100, random_state=42)

# To train the model on the training data
rf_model_consumption.fit(X_train_consumption, y_train_consumption)

# To make predictions on the test set
y_pred_consumption = rf_model_consumption.predict(X_test_consumption)

# To calculate R-squared and MAE for energy consumption model
r_squared_consumption_rf = r2_score(y_test_consumption, y_pred_consumption)
mae_consumption_rf = mean_absolute_error(y_test_consumption, y_pred_consumption)

print(f"R-squared for Energy Consumption: {r_squared_consumption_rf}")
print(f"Mean Absolute Error for Energy Consumption: {mae_consumption_rf}")

# To group by type of renewable energy and calculate the total jobs created
jobs_by_energy_type = energy_data.groupby('Type_of_Renewable_Energy')['Jobs_Created'].sum()

# Alternatively, we can calculate the average number of jobs created by each energy type
avg_jobs_by_energy_type = energy_data.groupby('Type_of_Renewable_Energy')['Jobs_Created'].mean()

# Plotting the total jobs created by renewable energy type
plt.figure(figsize=(10, 6))
jobs_by_energy_type.plot(kind='bar')
plt.title('Total Jobs Created by Renewable Energy Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Total Jobs Created')
plt.xticks(rotation=45)
plt.show()

# Plotting the average jobs created by renewable energy type
plt.figure(figsize=(10, 6))
avg_jobs_by_energy_type.plot(kind='bar')
plt.title('Average Jobs Created by Renewable Energy Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Average Jobs Created')
plt.xticks(rotation=45)
plt.show()

# To calculate correlation between jobs created and other variables
correlation_jobs = energy_data[['Jobs_Created', 'Energy_Production_MWh', 'Initial_Investment_USD', 'Financial_Incentives_USD', 'GHG_Emission_Reduction_tCO2e']].corr()

import seaborn as sns

# To plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_jobs, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Jobs Created and Other Variables')
plt.show()

# To create a new column for energy production ranges
bins = [0, 100000, 200000, 300000, 400000, np.inf]  # Define ranges for energy production
labels = ['0-100k', '100k-200k', '200k-300k', '300k-400k', '400k+']  # Labels for the ranges
energy_data['Energy_Production_Range'] = pd.cut(energy_data['Energy_Production_MWh'], bins=bins, labels=labels)

# Grouping by the energy production range and sum the jobs created
jobs_by_energy_range = energy_data.groupby('Energy_Production_Range')['Jobs_Created'].sum()

# To plot the bar chart
plt.figure(figsize=(10, 6))
jobs_by_energy_range.plot(kind='bar', color='skyblue')
plt.title('Total Jobs Created Across Different Energy Production Ranges')
plt.xlabel('Energy Production Range (MWh)')
plt.ylabel('Total Jobs Created')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Grouping by renewable energy type and calculate total GHG reduction
ghg_by_energy_type = energy_data.groupby('Type_of_Renewable_Energy')['GHG_Emission_Reduction_tCO2e'].sum()

# To plot total GHG emission reduction by energy type
plt.figure(figsize=(10, 6))
ghg_by_energy_type.plot(kind='bar')
plt.title('Total GHG Emission Reduction by Renewable Energy Type')
plt.xlabel('Type of Renewable Energy')
plt.ylabel('Total GHG Emission Reduction (tCO2e)')
plt.xticks(rotation=45)
plt.show()

# To calculate correlation between jobs created and GHG emission reduction
correlation_jobs_ghg = energy_data[['Jobs_Created', 'GHG_Emission_Reduction_tCO2e']].corr()

# To plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_jobs_ghg, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation: Jobs Created and GHG Emission Reduction')
plt.show()
