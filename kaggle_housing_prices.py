# This script tries to solve the Kaggle Housing Prices competition using regression models and neural networks
# The script have an option to run feature reduction based on Recursive Feature Elimination (RFE), Principal Component Analysis (PCA) and Autoencoder
# The models used are RandomForestRegressor, CatBoostRegressor, XGBRegressor and a simple ANN model
# We use Grid Search to find the best hyperparameters for the RandomForestRegressor, CatBoostRegressor and XGBRegressor
# The models are trained using the training data and the best model is selected based on the lowest mean squared error
# The best model is then used to predict the prices of the houses in the test data
# The dataset can be found at: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

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
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

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
categorical_columns = train_dataset.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
column_transformer = ColumnTransformer(transformers=[('encoder', one_hot_encoder, categorical_columns)], remainder='passthrough')

# Fit and transform the training data
X = column_transformer.fit_transform(X)
y = y.values

# Transform the test data
X_test = column_transformer.transform(X_test)

# Split the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Apply PCA
apply_pca = True  # Set to False to disable PCA

if apply_pca:
    n_components = 0.95  # Set the number of principal components to keep (% of variance to keep)

    print(f"Applying PCA with nº components: {n_components}")
    pca = PCA(n_components=n_components)  # n_components is the number of principal components to keep (% of variance to keep, [0, 1])
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

# Apply Recursive Feature Elimination (RFE)
apply_rfe = False  # Set to False to disable RFE

if apply_rfe:
    n_features = 70  # Set the number of features to select

    print(f"Applying Recursive Feature Elimination (RFE) with selected nº of features: {n_features}")
    model = RandomForestRegressor()
    rfe = RFE(estimator=model, n_features_to_select=n_features)  # n_features_to_select is the number of features to select
    rfe.fit(X_train, y_train)

    # Get selected features
    selected_features = rfe.support_
    X_train = X_train[:, selected_features]
    X_val = X_val[:, selected_features]
    X_test = X_test[:, selected_features]

# Apply Autoencoder for dimensionality reduction
apply_autoencoder = False  # Set to False to disable Autoencoder

if apply_autoencoder:
    encoding_dim = 60  # Set the encoding dimension (Options: 10, 20, 30, 40, 50, 60)

    print(f"Applying Autoencoder with encoding dimension: {encoding_dim}")
    input_dim = ncol  # Number of input features

    # Define the autoencoder model
    input_layer = Input(shape=(input_dim,))
        
    if encoding_dim == 10:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded2 = Dense(30, activation='relu')(encoded1)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded2)
        decoded1 = Dense(30, activation='relu')(encoded3)
        decoded2 = Dense(60, activation='relu')(decoded1)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
    elif encoding_dim == 20:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded2 = Dense(40, activation='relu')(encoded1)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded2)
        decoded1 = Dense(40, activation='relu')(encoded3)
        decoded2 = Dense(60, activation='relu')(decoded1)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
    elif encoding_dim == 30:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded2 = Dense(45, activation='relu')(encoded1)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded2)
        decoded1 = Dense(45, activation='relu')(encoded3)
        decoded2 = Dense(60, activation='relu')(decoded1)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
    elif encoding_dim == 40:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded2 = Dense(50, activation='relu')(encoded1)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded2)
        decoded1 = Dense(50, activation='relu')(encoded3)
        decoded2 = Dense(60, activation='relu')(decoded1)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
    elif encoding_dim == 50:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded2 = Dense(55, activation='relu')(encoded1)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded2)
        decoded1 = Dense(55, activation='relu')(encoded3)
        decoded2 = Dense(60, activation='relu')(decoded1)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded2)
    elif encoding_dim == 60:
        encoded1 = Dense(60, activation='relu')(input_layer)
        encoded3 = Dense(encoding_dim, activation='relu')(encoded1)
        decoded1 = Dense(60, activation='relu')(encoded3)
        decoded3 = Dense(input_dim, activation='sigmoid')(decoded1)

    autoencoder = Model(input_layer, decoded3)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=200, batch_size=32, shuffle=True, validation_data=(X_val, X_val))

    # Use the encoder part of the autoencoder to transform the data
    encoder = Model(input_layer, encoded3)
    X_train = encoder.predict(X_train)
    X_val = encoder.predict(X_val)
    X_test = encoder.predict(X_test)

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

# Train the ANN model
print("Training ANN model")
ann_regressor = tf.keras.models.Sequential()
ann_regressor.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
ann_regressor.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann_regressor.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann_regressor.add(tf.keras.layers.Dense(units=1))
ann_regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
ann_regressor.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the ANN model
y_pred_ann_regressor = ann_regressor.predict(X_val)
mse_ann_regressor = mean_squared_error(y_val, y_pred_ann_regressor)
r2_ann_regressor = r2_score(y_val, y_pred_ann_regressor)
print(f"ANN Mean Squared Error: {mse_ann_regressor}")
print(f"ANN R-squared: {r2_ann_regressor}")

# Compare and select the best model
best_regressor = None
best_parameters = None
best_score = float('inf')  # Initialize best_score to positive infinity

if mse_ann_regressor < best_score:
    best_regressor = ann_regressor
    best_parameters = None
    best_score = mse_ann_regressor
    print("Best Model found: ANN")
    print(f"Best score: {best_score}")

if abs(rf_best_score) < best_score:
    best_regressor = rf_grid_search.best_estimator_
    best_parameters = rf_best_parameters
    best_score = abs(rf_best_score)
    print("Best Model found: RandomForestRegressor")
    print(f"Best score: {best_score}")

if abs(catboost_best_score) < best_score:
    best_regressor = catboost_grid_search.best_estimator_
    best_parameters = catboost_best_parameters
    best_score = abs(catboost_best_score)
    print("Best Model found: CatBoostRegressor")
    print(f"Best score: {best_score}")

if abs(xgboost_best_score) < best_score:
    best_regressor = xgboost_grid_search.best_estimator_
    best_parameters = xgboost_best_parameters
    best_score = abs(xgboost_best_score)
    print("Best Model found: XGBRegressor")
    print(f"Best score: {best_score}")

print(f"Best Model: {type(best_regressor).__name__}")
print(f"Best Parameters: {best_parameters}")

# Predicting the Test set results with the best model
if best_regressor == ann_regressor:
    y_test_pred = ann_regressor.predict(X_test)
else:
    y_test_pred = best_regressor.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_dataset['Id'], 'SalePrice': y_test_pred.flatten()})
output.to_csv(f'{data_folder}/output/submission.csv', index=False)
print("Predictions saved to submission.csv")
