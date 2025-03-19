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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Load the dataset
train_dataset = pd.read_csv(f'{data_folder}/train.csv')
test_dataset = pd.read_csv(f'{data_folder}/test.csv')

# Create X and y variables
X = train_dataset.iloc[:, 1:-1]  # Select all columns except the first (Id) and the last (SalePrice)
X_test = test_dataset.iloc[:, 1:] # Select all columns except the first (Id)
y = train_dataset.iloc[:, -1]    # Select the last column (SalePrice)

# Identify columns with missing values
columns_with_mean_imputation = ['LotFrontage', 'MasVnrArea']
columns_with_zero_imputation = ['GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                                'BsmtHalfBath', 'GarageCars', 'GarageArea']
columns_with_none_imputation = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 
                                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                                'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']

# Handling missing values with mean imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[columns_with_mean_imputation] = mean_imputer.fit_transform(X[columns_with_mean_imputation])
X_test[columns_with_mean_imputation] = mean_imputer.transform(X_test[columns_with_mean_imputation])

# Handling missing values with zero imputation
zero_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X[columns_with_zero_imputation] = zero_imputer.fit_transform(X[columns_with_zero_imputation])
X_test[columns_with_zero_imputation] = zero_imputer.transform(X_test[columns_with_zero_imputation])

# Handling missing values with 'None' imputation
none_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')
X[columns_with_none_imputation] = none_imputer.fit_transform(X[columns_with_none_imputation])
X_test[columns_with_none_imputation] = none_imputer.transform(X_test[columns_with_none_imputation])

# Identify categorical columns
categorical_columns = train_dataset.select_dtypes(include=['object']).columns.tolist()

# Add the integer columns that should be treated as categorical
categorical_columns += ['MSSubClass', 'OverallQual', 'OverallCond']

# Convert these columns to 'object' type in both training and test datasets
train_dataset[['MSSubClass', 'OverallQual', 'OverallCond']] = train_dataset[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')
test_dataset[['MSSubClass', 'OverallQual', 'OverallCond']] = test_dataset[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')

# Combine training and test datasets for encoding
combined_data = pd.concat([X, X_test], axis=0)

# Apply one-hot encoding to categorical columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
column_transformer = ColumnTransformer(transformers=[('encoder', one_hot_encoder, categorical_columns)], remainder='passthrough')

# Fit the encoder on the combined data
combined_data_encoded = column_transformer.fit_transform(combined_data)

# Split back into training and test datasets
X = combined_data_encoded[:len(X), :]
X_test = combined_data_encoded[len(X):, :]

# Split the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

#import sys
#sys.exit("Stopping script for testing purposes.")

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

mlp_parameters = {
    'hidden_layer_sizes': [(128, 64), (100, 50), (64, 32)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [200, 300, 500, 1000],
}

# Grid Search for RandomForestRegressor with k-Fold Cross Validation
print("Starting Grid Search for RandomForestRegressor with k-Fold Cross Validation")
rf_regressor = RandomForestRegressor(random_state=0)
rf_grid_search = GridSearchCV(estimator=rf_regressor, param_grid=rf_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_score = rf_grid_search.best_score_
rf_best_parameters = rf_grid_search.best_params_
print("RandomForest Best Score: {:.2f}".format(rf_best_score))
print("RandomForest Best Parameters:", rf_best_parameters)

# Grid Search for CatBoostRegressor with k-Fold Cross Validation
print("Starting Grid Search for CatBoostRegressor with k-Fold Cross Validation")
catboost_regressor = CatBoostRegressor(random_state=0, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_regressor, param_grid=catboost_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
catboost_best_score = catboost_grid_search.best_score_
catboost_best_parameters = catboost_grid_search.best_params_
print("CatBoost Best Score: {:.2f}".format(catboost_best_score))
print("CatBoost Best Parameters:", catboost_best_parameters)

# Grid Search for XGBRegressor with k-Fold Cross Validation
print("Starting Grid Search for XGBRegressor with k-Fold Cross Validation")
xgboost_regressor = XGBRegressor(random_state=0)
xgboost_grid_search = GridSearchCV(estimator=xgboost_regressor, param_grid=xgboost_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
xgboost_grid_search.fit(X_train, y_train)
xgboost_best_score = xgboost_grid_search.best_score_
xgboost_best_parameters = xgboost_grid_search.best_params_
print("XGBoost Best Score: {:.2f}".format(xgboost_best_score))
print("XGBoost Best Parameters:", xgboost_best_parameters)

# Perform Grid Search for MLPRegressor with k-Fold Cross Validation
print("Starting Grid Search for MLPRegressor with k-Fold Cross Validation")
mlp_regressor = MLPRegressor(random_state=0)
mlp_grid_search = GridSearchCV(estimator=mlp_regressor, param_grid=mlp_parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
mlp_grid_search.fit(X_train, y_train)
mlp_best_score = mlp_grid_search.best_score_
mlp_best_parameters = mlp_grid_search.best_params_
print("MLPRegressor Best Score: {:.2f}".format(mlp_best_score))
print("MLPRegressor Best Parameters:", mlp_best_parameters)

# Define the base models using the best estimators from grid search
base_models = [
    ('rf', rf_grid_search.best_estimator_),
    ('catboost', catboost_grid_search.best_estimator_),
    ('xgboost', xgboost_grid_search.best_estimator_),
    ('mlp', mlp_grid_search.best_estimator_)
]

# Define the meta-model
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the stacking regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the stacking regressor
stacking_regressor.fit(X_train, y_train)

# Evaluate the stacking regressor
y_pred_stacking = stacking_regressor.predict(X_val)
mse_stacking = mean_squared_error(y_val, y_pred_stacking)
r2_stacking = r2_score(y_val, y_pred_stacking)
print(f"Stacking Regressor Mean Squared Error: {mse_stacking}")
print(f"Stacking Regressor R-squared: {r2_stacking}")

# Predicting the Test set results with the stacking regressor
y_test_pred_stacking = stacking_regressor.predict(X_test)

# Save the predictions to a CSV file
output_stacking = pd.DataFrame({'Id': test_dataset['Id'], 'SalePrice': y_test_pred_stacking.flatten()})
output_stacking.to_csv(f'{data_folder}/output/submission_stacking.csv', index=False)
print("Stacking Regressor predictions saved to submission_stacking.csv")
