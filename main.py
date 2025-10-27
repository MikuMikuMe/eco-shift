Creating a project like "eco-shift" involves several components, from data acquisition to model training and deployment. Below is a simplified Python program that outlines the key components of such a system. This program assumes you have a dataset with historical energy consumption and associated features (like timestamp, weather data, and occupancy). The program will focus on using a machine learning model to predict and optimize energy consumption.

This is a high-level outline and will require adaptation based on actual data sources and infrastructure.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load energy data from a CSV file.
    Args:
    - file_path (str): Path to the csv data file.
    
    Returns:
    - pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        logging.error("No data: The file is empty.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def preprocess_data(data):
    """
    Preprocess the dataset for training.
    Args:
    - data (pd.DataFrame): Raw data.
    
    Returns:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    """
    try:
        # Assuming the dataset has columns 'energy', 'timestamp', 'temperature', 'occupancy'.
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek

        # Drop non-numeric or unnecessary columns
        data.drop(['timestamp'], axis=1, inplace=True)

        # Split data into features and target
        X = data.drop('energy', axis=1)
        y = data['energy']

        logging.info("Data preprocessing completed.")
        return X, y
    except KeyError as e:
        logging.error(f"Key error - possibly missing column: {e}")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")

def train_model(X, y):
    """
    Train a Random Forest model to predict energy consumption.
    
    Args:
    - X (pd.DataFrame): Features.
    - y (pd.Series): Target variable.
    
    Returns:
    - model (RandomForestRegressor): Trained model.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model trained with Mean Squared Error: {mse}")

        # Save the model
        joblib.dump(model, 'energy_model.pkl')
        logging.info("Model saved to energy_model.pkl")

        return model
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")

def load_model(model_path):
    """
    Load a trained model from a file.
    
    Args:
    - model_path (str): Path to the model file.
    
    Returns:
    - model: Loaded model.
    """
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")

def predict_energy_consumption(model, X_future):
    """
    Predict future energy consumption.
    
    Args:
    - model: Trained machine learning model.
    - X_future (pd.DataFrame): Future features to make predictions on.
    
    Returns:
    - np.array: Predicted energy consumption.
    """
    try:
        predictions = model.predict(X_future)
        return predictions
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return None

def main():
    # Define file paths
    data_file_path = 'energy_data.csv'
    model_file_path = 'energy_model.pkl'

    # Load and preprocess data
    data = load_data(data_file_path)
    if data is not None:
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            # Train model
            model = train_model(X, y)
            
            # Example code: Load model and make predictions on new data
            if os.path.exists(model_file_path):
                model = load_model(model_file_path)

                # Create some fictional future data for prediction
                X_future = pd.DataFrame({
                    'temperature': [23, 21],
                    'occupancy': [2, 3],
                    'hour': [14, 15],
                    'day_of_week': [2, 2]
                })
                
                future_predictions = predict_energy_consumption(model, X_future)
                if future_predictions is not None:
                    logging.info(f"Future energy predictions: {future_predictions}")
        else:
            logging.error("Preprocessing failed.")
    else:
        logging.error("Data loading failed.")

if __name__ == '__main__':
    main()
```

### Key Points:

1. **Data Loading and Preprocessing:** The script includes loading data, converting timestamps, and creating new features like hour and day_of_week. Error handling is added for potential issues.

2. **Model Training:** A `RandomForestRegressor` is used to train a regression model to predict energy consumption based on input features.

3. **Model Persisting and Loading:** The trained model is saved using `joblib`. Error handling is included for file operations.

4. **Prediction:** The script includes a predictive component to demonstrate how future predictions can be made, which can later be used for optimizing energy usage.

5. **Logging:** Standard logging is used throughout for debugging and tracking the workflow execution.

This program is a starting point and can be further refined by integrating it into home automation systems, enhancing data pipelines, and using more sophisticated models and techniques.