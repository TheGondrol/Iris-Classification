import pickle
import pandas as pd

from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the trained model
with open(f"{BASE_DIR}/iris_tree_clf.pickle", "rb") as handle:
    model = pickle.load(handle)
    
# Map numeric labels to species names
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


def model_predict(iris_dict):

    # Convert dictionary to pandas DataFrame
    data = pd.DataFrame([iris_dict])

    # Make prediction
    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)[0][prediction]

    # Map prediction to species name
    species = species_mapping[prediction]

    return prediction, prediction_proba, species