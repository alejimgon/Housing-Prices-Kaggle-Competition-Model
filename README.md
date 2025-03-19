# Housing Prices Kaggle Competition Model

## Author: Alejandro Jiménez-González

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data](#data)
- [Scripts](#scripts)
- [Usage](#usage)
- [Results](#results)
- [TODO](#todo)
- [Contributing](#contributing)
- [License](#license)

## Description
This repository contains my current models for the Housing Prices Kaggle competition. It includes two scripts:

1. **`kaggle_housing_prices.py`**: Implements regression models (`RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`) and an Artificial Neural Network (ANN) using TensorFlow. It includes options for feature reduction using `Principal Component Analysis` (PCA), `Recursive Feature Elimination` (RFE), or an `autoencoder`. The best model is selected based on the lowest mean squared error and used to predict the test set.

2. **`kaggle_housing_prices_stacking.py`**: Implements a stacking regressor that combines the predictions of multiple models (`RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`, and `MLPRegressor` from scikit-learn). It uses `GridSearchCV` to find the best hyperparameters for each model and evaluates the stacking regressor's performance.

Both scripts save the predictions in a CSV file in the `output` folder.

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/alejimgon/Housing-Prices-Kaggle-Competition-Model.git
    cd Housing-Prices-Kaggle-Competition_Model
    ```

2. **Set up the conda environment**:
    ```sh
    conda create --name housing_env python=3.12.7
    conda activate housing_env
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Data
The dataset used in this competition is provided by Kaggle and contains information about 79 features of houses.

### Download the Data
1. **Download the data from Kaggle**:
    - Go to the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) competition page on Kaggle.
    - Download the `train.csv` and `test.csv` files.

2. **Place the data files in the data directory**:
    - Create a data directory in the project root if it doesn't exist:
      ```sh
      mkdir data
      ```
    - Move the downloaded `train.csv` and `test.csv` files to the data directory.

## Scripts
### `kaggle_housing_prices.py`
This script performs the following steps:
1. **Data Preprocessing**: Handles missing values and encodes categorical variables.
2. **Feature Scaling**: Scales the features using `StandardScaler`.
3. **Feature Reduction**: If activated, reduces the feature dimensionality using PCA, RFE, or an autoencoder.
4. **Grid Search**: Uses `GridSearchCV` to find the best parameters for `RandomForestRegressor`, `CatBoostRegressor`, and `XGBRegressor`.
5. **k-Fold Cross Validation**: Evaluates the performance of the models.
6. **ANN Training**: Trains an Artificial Neural Network (ANN) using TensorFlow.
7. **Model Selection**: Compares the models and selects the best one.
8. **Prediction**: Uses the best model to predict the test set.

### `kaggle_housing_prices_stacking.py`
This script performs the following steps:
1. **Data Preprocessing**: Handles missing values and encodes categorical variables.
2. **Feature Scaling**: Scales the features using `StandardScaler`.
3. **Grid Search**: Uses `GridSearchCV` to find the best parameters for `RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`, and `MLPRegressor`.
4. **Stacking Regressor**: Combines the predictions of the base models using a `StackingRegressor` with a `RandomForestRegressor` as the meta-model.
5. **Evaluation**: Evaluates the stacking regressor's performance.
6. **Prediction**: Uses the stacking regressor to predict the test set.

## Usage
### `kaggle_housing_prices.py`
1. **Setup the desired feature reduction**:
    - Enable PCA, RFE, or autoencoder by setting the corresponding flags to `True` in the script.
    - Configure the parameters for the selected feature reduction method.

2. **Run the script**:
    ```sh
    python kaggle_housing_prices.py
    ```

### `kaggle_housing_prices_stacking.py`
1. **Run the script**:
    ```sh
    python kaggle_housing_prices_stacking.py
    ```

## Results
Both scripts output the best Mean Squared Error (MSE) and parameters for all models. The predictions are saved in a CSV file in the `output` folder:
- `kaggle_housing_prices.py`: `output/submission.csv`
- `kaggle_housing_prices_stacking.py`: `output/submission_stacking.csv`

## TODO
- Make the scripts more user-friendly (e.g., allow configuration via command-line arguments).
- Refactor the scripts for better modularity and reusability.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.