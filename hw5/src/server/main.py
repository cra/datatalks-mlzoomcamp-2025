import os
import pathlib
import pickle
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Record(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictResponse(BaseModel):
    probability: float
    converted: bool


app = FastAPI()

# looks ugly but meh
model = {}
model_v1_path = pathlib.Path(os.getenv('MODELV1_PATH'))
model["v1"] = pickle.load(model_v1_path.open("rb"))
model_v2_path = pathlib.Path(os.getenv('MODELV2_PATH'))
model["v2"] = pickle.load(model_v2_path.open("rb"))


@app.post("/{version}/predict")
async def predict_handle(version: str, record: Record) -> PredictResponse:
    prob = model[version].predict_proba(record.model_dump())[:, 1][0]

    return PredictResponse(
        probability=prob,
        converted=prob >= 0.5,
    )


def serve():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
