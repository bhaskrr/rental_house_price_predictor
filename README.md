# Rental House Price Predictor Web App

This web app predicts rental house prices based on user-provided features such as the number of bedrooms, square footage, etc. It uses a RandomForestRegressor model and is built with Flask for the backend.

## Features

* Predict rental house prices based on user input.
* Real-time predictions powered by a pre-trained Random Forest model.
* User-friendly interface with dropdowns and input fields.

## Prerequisites

Prerequisites:

* Python 3.10 or above.
* Flask framework installed.
* Libraries: Scikit-learn, Pandas, Joblib and Gunicorn.
* A web browser for accessing the UI.

## Installation Guide

1. Clone the repository

   ```bash
       git clone https://github.com/bhaskrr/rental_house_price_predictor.git
       cd rental_house_price_predictor
   ```

2. Install dependencies

    ```bash
        pip install -r requirements.txt
    ```

3. Run the app

    ```bash
        gunicorn app:app
    ```

4. Open the app in a web browser:
    Visit <http://127.0.0.1:8000>.

## Usage

1. Open the app in a browser.
2. Fill in the required details:
   * Layout Type (select from dropdown).
   * Property Type (select from dropdown).
   * Furnish Type (select from dropdown).
   * Area in Square foot.
   * Number of bedrooms.
   * Number of bathrooms.
3. Click on the Get Rent button.
4. View the predicted rental price.

## Model Information

* Algorithm: RandomForestRegressor.
* Training Data: Dataset with 193011 rental listings, including features like layout type, property type, bedrooms, square footage, and rental price etc.
* Dataset Link: <https://www.kaggle.com/datasets/saisaathvik/house-rent-prices-of-metropolitan-cities-in-india/data?select=_All_Cities_Cleaned.csv>.
* Evaluation: 0.77 r2 score on test data.

## Project Structure

```python
rental_house_price_predictor/
|
|- app.py                           # Main Flask App
|- requirements.txt                 # Dependencies
|- rf_regressor.zip                 # Compressed Pre-trained RandomForestRegressor Model
|- furnish_type_encoder.joblib      # Pre-fitted Furnish Type Encoder
|- layout_type_encoder.joblib       # Pre-fitted Layout Type Encoder
|- property_type_encoder.joblib     # Pre-fitted Property Type Encoder
|- static/                          # Static Files
|   |__ index.css
|- templates                        # HTML Templates
|   |__ index.html
|- README.md                        # Documentation
```

## Known Issues and Limitations

1. Predictions are less accurate for out-of-distribution data (e.g., very large houses).

2. No support for categorical variables outside the trained dataset.
