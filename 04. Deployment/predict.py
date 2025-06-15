import pickle
import pandas as pd

with open("models/linear_regression_model.pkl", "rb") as f_model, open("models/dict_vectorizer.pkl", "rb") as f_vect:
    model = pickle.load(f_model)
    dv = pickle.load(f_vect)


# Load May 2023 data (you can download inside the container or mount locally)
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-05.parquet"
df = pd.read_parquet(url)

# Prepare features like in training
df['duration'] = (pd.to_datetime(df.lpep_dropoff_datetime) - pd.to_datetime(df.lpep_pickup_datetime)).dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)]

# Convert categorical features to string
df['PULocationID'] = df['PULocationID'].astype(str)
df['DOLocationID'] = df['DOLocationID'].astype(str)

# Create dictionary for vectorizer
dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')

# Vectorize
X = dv.transform(dicts)

# Predict duration
preds = model.predict(X)

print("Mean predicted duration:", preds.mean())
