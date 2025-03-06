# Housing Prices Kaggle Competition Model

## Author: Alejandro Jiménez-González

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [TODO](#todo)
- [Contributing](#contributing)
- [License](#license)

## Description
This repository contains my current model for the Housing Prices Kaggle competition. It implements the option to reduce the dimensionality of the dataset using `Principal Component Analysis` (PCA), `Recursive Feature Elimination` (RFE), or an `autoencoder`. This allows studying how the prediction behaves when the number of features is modified under different statistics. I also use Grid Search to find the best parameters for `RandomForestRegressor`, `CatBoostRegressor`, and `XGBRegressor`. Additionally, an `Artificial Neural Network` (ANN) is trained and compared. The best model is then trained and used to predict the test set. The predictions are saved in a CSV file in the 'output' folder.

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

## Model Training
The script performs the following steps:
1. **Data Preprocessing**: Handles missing values and encodes categorical variables.
2. **Feature Scaling**: Scales the features using `StandardScaler`.
3. **Feature reduction**: If activated, it will reduce the feature dimensionality based on Principal Component Analysis (PCA), Recursive Feature Elimination (RFE), or an autoencoder. 
4. **Grid Search**: Uses `GridSearchCV` to find the best parameters for `RandomForestRegressor`, `CatBoostRegressor`, and `XGBRegressor`.
5. **k-Fold Cross Validation**: Uses `cross_val_score` to evaluate `RandomForestRegressor`, `CatBoostRegressor`, and `XGBRegressor` performance.
6. **ANN Training**: Trains an `Artificial Neural Network` (ANN) and evaluates its performance.
7. **Model Selection**: Compares the best scores and selects the best model.
8. **Model Training**: Trains the selected model with the best parameters.
9. **Prediction**: Uses the trained model to predict the test set.

## Usage
1. **Setup the desired feature reduction**
    The desired statistic needs to be turned on and the dimension parameter needs to be selected. This script allows reducing the feature dimension in three different ways:

    - To use Principal Component Analysis (PCA).
    ```python
    # Apply PCA
    apply_pca = True  # Set to False to disable PCA

    if apply_pca:
        n_components = 0.95  # Set the number of principal components to keep (% of variance to keep [0,1])
    ```
    - To use Recursive Feature Elimination (RFE).
    ```python
    # Apply Recursive Feature Elimination (RFE)
    apply_rfe = True  # Set to False to disable RFE

    if apply_rfe:
        n_features = 70  # Set the number of features to select
    ```
    - To use Autoencoder.
    ```python
    # Apply Autoencoder for dimensionality reduction
    apply_autoencoder = True  # Set to False to disable Autoencoder

    if apply_autoencoder:
        encoding_dim = 10  # Set the encoding dimension (Options: 10, 20, 30, 40, 50, 60)
    ```   

2. **Run the script**  
    To run the script, use the following command:
    ```sh
    python kaggle_housing_prices.py
    ```

## TODO
Making the script more user-friendly (option prompted using the terminal)  
Refactoring the script

## Results
The script outputs the best Mean Squared Error (MSE) and parameters for all models. It also saves the predictions in a CSV file in the 'output' folder.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
