from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model
from predict import predict_image

app = FastAPI(title="Smart Waste Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, classes = load_model()
