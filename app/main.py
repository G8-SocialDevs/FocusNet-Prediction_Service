from fastapi import FastAPI
from app.api import prediction

app = FastAPI()

app.include_router(prediction.router, prefix="/prediction", tags=["prediction"])