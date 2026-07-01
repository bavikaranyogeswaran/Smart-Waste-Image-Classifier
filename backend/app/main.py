from fastapi import FastAPI, UploadFile, File
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


@app.get("/")
def root():
    return {"message": "Smart Waste Classifier API is running", "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = predict_image(file.file, model, classes)
    return result
