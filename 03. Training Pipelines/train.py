import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import click

def read_dataframe(filename):
    """
    Read and preprocess the dataframe according to the specified logic
    """
    print("Reading data...")
    df = pd.read_parquet(filename)
    
    # Question 3: Print the number of records loaded
    print(f"Number of records loaded: {len(df):,}")
    
    # Calculate duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    
    # Filter duration between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical features to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Question 4: Print the size after data preparation
    print(f"Size after data preparation: {len(df):,}")
    
    return df

def prepare_features_and_target(df):
    """
    Prepare features and target for modeling
    """
    print("Preparing features and target...")
    
    # Use only pickup and dropoff locations as features
    categorical = ['PULocationID', 'DOLocationID']
    
    # Convert to dictionary format
    records = df[categorical].to_dict(orient='records')
    
    # Fit dict vectorizer
    dv = DictVectorizer()
    X = dv.fit_transform(records)
    
    # Target variable
    y = df['duration'].values
    
    return X, y, dv

def train_linear_regression(X, y):
    """
    Train linear regression model with default parameters
    """
    print("Training linear regression model...")
    
    # Train linear regression with default parameters
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Question 5: Print the intercept
    print(f"Model intercept: {lr.intercept_:.2f}")
    
    return lr

def register_model_with_mlflow(model, dv, X, y):
    """
    Register the model with MLflow and log relevant information
    """
    print("Registering model with MLflow...")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("taxi-duration-prediction")
    
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID")
        mlflow.log_param("vectorizer_type", "DictVectorizer")
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X)
        rmse_score = np.sqrt(mean_squared_error(y, y_pred))
        
        # Log metrics
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("intercept", model.intercept_)
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="taxi-duration-lr"
        )
        
        # Log the vectorizer as an artifact
        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dict_vectorizer.pkl")
        
        print(f"Model RMSE: {rmse_score:.2f}")
        print("Model registered with MLflow successfully!")
        
        # Get run info for model size
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        return run_id

def create_models_directory_and_save(model, dv):
    """
    Create models directory and save model locally
    """
    print("Saving model to local directory...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/linear_regression_model.pkl'
    vectorizer_path = 'models/dict_vectorizer.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

@click.command()
@click.option(
    "--data_url",
    default="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
    help="URL to the parquet data file"
)
def main(data_url: str):
    """
    Main function to run the complete pipeline and answer the questions
    """
    print("=" * 60)
    print("TAXI DURATION PREDICTION PIPELINE")
    print("=" * 60)
    
    # Question 3 & 4: Read and prepare data
    print("\n--- QUESTION 3 & 4: DATA LOADING AND PREPARATION ---")
    df = read_dataframe(data_url)
    
    # Prepare features and target
    print("\n--- FEATURE PREPARATION ---")
    X, y, dv = prepare_features_and_target(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Question 5: Train linear regression model
    print("\n--- QUESTION 5: TRAIN LINEAR REGRESSION ---")
    model = train_linear_regression(X, y)
    
    # Question 6: Register model with MLflow
    print("\n--- QUESTION 6: REGISTER MODEL WITH MLFLOW ---")
    run_id = register_model_with_mlflow(model, dv, X, y)
    
    # Save model locally
    print("\n--- SAVE MODEL LOCALLY ---")
    create_models_directory_and_save(model, dv)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Summary of answers
    # print("\n--- SUMMARY OF ANSWERS ---")
    # print("Question : Check the printed number of records loaded above")
    # print("Question : Check the printed size after data preparation above")
    # print("Question : Check the printed model intercept above")
    # print("Question : Check MLflow UI for model size in MLModel file")
    print(f"Question  Hint: Go to MLflow UI -> Experiments -> Run ID: {run_id}")
    # print("Then check the 'model' artifact -> MLmodel file -> model_size_bytes field")



if __name__ == '__main__':
    main()