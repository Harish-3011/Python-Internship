# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')

# Explore data
print(df.head())
print(df.nunique())

# Data preprocessing
print(df.info())
print(df.describe())
print(df.corr())

# Remove missing values
df = df.dropna()
print(df.info())

# Data visualization
sns.pairplot(df, x_vars=['displacement', 'horsepower', 'weight', 'acceleration'], y_vars='mpg')
sns.regplot(x='displacement', y='mpg', data=df)
plt.show()

# Define target variable y and features x
y = df['mpg']
x = df[['displacement', 'horsepower', 'weight']]

# Scale the data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_scaled = ss.fit_transform(x)

# Convert scaled data to DataFrame for inspection
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
print(x_scaled_df.describe())

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.7, random_state=2529)

# Create and train linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict and evaluate
y_pred = lr.predict(x_test)
from sklearn.metrics import mean_absolute_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

lr.fit(x_train_poly, y_train)
y_pred_poly = lr.predict(x_test_poly)

from sklearn.metrics import mean_absolute_percentage_error
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_poly))
print("R2 Score (Poly):", r2_score(y_test, y_pred_poly))