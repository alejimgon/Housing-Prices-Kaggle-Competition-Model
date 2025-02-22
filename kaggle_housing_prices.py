# This script tries to solve the Kaggle Housing Prices competition using regression models
# The dataset can be found at: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# The model is trained using the training data in order to predict the process of the houses in the test data

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Load the dataset and drop the Id column
train_dataset = pd.read_csv(f'{data_folder}/train.csv')
test_dataset = pd.read_csv(f'{data_folder}/test.csv')
train_dataset = train_dataset.drop(columns=['Id'])
test_dataset = test_dataset.drop(columns=['Id'])

# Create X and y variables
X = train_dataset.drop(columns=['SalePrice'])
y = train_dataset['SalePrice']

# Identify columns with missing values
columns_with_mean_imputation = ['LotFrontage', 'MasVnrArea']
columns_with_zero_imputation = ['GarageYrBlt']

# Handling missing values with mean imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[columns_with_mean_imputation] = mean_imputer.fit_transform(X[columns_with_mean_imputation])
test_dataset[columns_with_mean_imputation] = mean_imputer.transform(test_dataset[columns_with_mean_imputation])

# Handling missing values with zero imputation
zero_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X[columns_with_zero_imputation] = zero_imputer.fit_transform(X[columns_with_zero_imputation])
test_dataset[columns_with_zero_imputation] = zero_imputer.transform(test_dataset[columns_with_zero_imputation])

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_columns)
    ],
    remainder='passthrough'
)

# Fit and transform the training data
X = column_transformer.fit_transform(X)
y = y.values

# Transform the test data
X_test = column_transformer.transform(test_dataset)

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# Split the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Parameters for Grid Search
rf_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

catboost_parameters = {
    'iterations': [50, 100, 200],
    'depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7]
}

xgboost_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0, 0.5, 1]
}

# Grid Search for RandomForestRegressor
print("Starting Grid Search for RandomForestRegressor")
rf_regressor = RandomForestRegressor(random_state=0)
rf_grid_search = GridSearchCV(estimator=rf_regressor, param_grid=rf_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_score = rf_grid_search.best_score_
rf_best_parameters = rf_grid_search.best_params_
print("RandomForest Best Score: {:.2f}".format(rf_best_score))
print("RandomForest Best Parameters:", rf_best_parameters)

# Grid Search for CatBoostRegressor
print("Starting Grid Search for CatBoostRegressor")
catboost_regressor = CatBoostRegressor(random_state=0, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_regressor, param_grid=catboost_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
catboost_best_score = catboost_grid_search.best_score_
catboost_best_parameters = catboost_grid_search.best_params_
print("CatBoost Best Score: {:.2f}".format(catboost_best_score))
print("CatBoost Best Parameters:", catboost_best_parameters)

# Grid Search for XGBRegressor
print("Starting Grid Search for XGBRegressor")
xgboost_regressor = XGBRegressor(random_state=0)
xgboost_grid_search = GridSearchCV(estimator=xgboost_regressor, param_grid=xgboost_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
xgboost_grid_search.fit(X_train, y_train)
xgboost_best_score = xgboost_grid_search.best_score_
xgboost_best_parameters = xgboost_grid_search.best_params_
print("XGBoost Best Score: {:.2f}".format(xgboost_best_score))
print("XGBoost Best Parameters:", xgboost_best_parameters)

# Compare and select the best model
best_regressor = None
best_parameters = None
best_score = float('-inf')
if rf_best_score > best_score:
    best_regressor = rf_grid_search.best_estimator_
    best_parameters = rf_best_parameters
    best_score = rf_best_score
if catboost_best_score > best_score:
    best_regressor = catboost_grid_search.best_estimator_
    best_parameters = catboost_best_parameters
    best_score = catboost_best_score
if xgboost_best_score > best_score:
    best_regressor = xgboost_grid_search.best_estimator_
    best_parameters = xgboost_best_parameters
    best_score = xgboost_best_score

# Train the best model with the best parameters
best_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = best_regressor.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Predicting the Test set results with the best model
y_test_pred = best_regressor.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_dataset.index, 'SalePrice': y_test_pred})
output.to_csv(f'{data_folder}/output/submission.csv', index=False)
print("Predictions saved to submission.csv")