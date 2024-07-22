from fastapi import FastAPI
from pydantic import BaseModel, Field

from model.classifier import model_predict


# Initialize FastAPI app
app = FastAPI()

# Define request body using Pydantic with keys containing spaces and units
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., alias="sepal length (cm)")
    sepal_width: float = Field(..., alias="sepal width (cm)")
    petal_length: float = Field(..., alias="petal length (cm)")
    petal_width: float = Field(..., alias="petal width (cm)")

# Define response model using Pydantic (optional)
class IrisResponse(BaseModel):
    prediction: int
    prediction_proba: float
    species: str


@app.post("/predict", response_model=IrisResponse)
def predict(iris: IrisRequest):

    # Convert Pydantic model to dictionary
    iris_dict = iris.dict(by_alias=True)

    # Make prediction
    prediction, prediction_proba, species = model_predict(iris_dict)

    # Return result as response model
    return IrisResponse(prediction=prediction, prediction_proba=prediction_proba, species=species)