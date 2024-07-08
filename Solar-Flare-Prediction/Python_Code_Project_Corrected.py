

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Collection
data = pd.read_csv("/Users/shiveshrajsahu/Desktop/CS677/SHIVESH_RAJ_SAHU_Project/Viewing Solar Flares.CSV")

# 2. Initial Data Exploration
print(data.head())
print(data.info())
sns.pairplot(data)
plt.show()

# 3. Data Preprocessing

# Convert 'JJJ Class' into numerical representation
data['JJJ Class'] = data['JJJ Class'].str.extract('([0-9.]+)', expand=False).astype(float)

# Retaining only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Handling Missing Values
numeric_data.fillna(numeric_data.mean(), inplace=True)

# Removing outliers (simple method using z-scores)
z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
numeric_no_outliers = numeric_data[(z_scores < 3).all(axis=1)]

# Merge the processed numeric data back with the non-numeric data
non_numeric_data = data.select_dtypes(exclude=[np.number])
data = pd.concat([non_numeric_data.reset_index(drop=True), numeric_no_outliers.reset_index(drop=True)], axis=1).dropna()

# Standardization
scaler = StandardScaler()  # Initialize the scaler
numeric_cols = data.select_dtypes(include=[np.number])
numeric_scaled = scaler.fit_transform(numeric_cols)
data[numeric_cols.columns] = numeric_scaled

# Convert the date-time columns to datetime format first
date_columns = ['JJJ Start', 'JJJ Peak', 'JJJ End']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])
    
    # Now, extract features from these datetime columns
    data[col+'_year'] = data[col].dt.year
    data[col+'_month'] = data[col].dt.month
    data[col+'_day'] = data[col].dt.day
    data[col+'_hour'] = data[col].dt.hour
    data[col+'_minute'] = data[col].dt.minute

# Once I've extracted the features, I can drop the original date-time columns
data.drop(date_columns, axis=1, inplace=True)

# 4. Feature Engineering
X = data.drop('EEE', axis=1)
y = data['EEE']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Development
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

svm = SVR()
svm.fit(X_train, y_train)

nn = MLPRegressor()
nn.fit(X_train, y_train)

# Evaluation
models = [rf, svm, nn]
names = ["Random Forest", "SVM", "Neural Network"]

for model, name in zip(models, names):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse}, R2: {r2}\n")

# 7. Conclusion
# It displays:
#The first 5 rows of the dataset.
#The summary of the dataset (data types and non-null counts).
#Performance metrics (MSE and R2) for the three regression models on the test data.

#My results show:
#Random Forest: Has the best R2 score of ~0.28.
#SVM: Performs slightly worse with a negative R2, indicating it might be performing no better 
#than a horizontal straight line.
#Neural Network: Shows an incredibly high MSE and a massively negative R2. 

