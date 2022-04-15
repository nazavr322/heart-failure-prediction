from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    pass

@app.get("/predict")
def make_prediction(age: int, ejection_fraction: int, serum_creatinine: float):
    pred = float(model.predict_proba([[age, ejection_fraction,serum_creatinine]])[:, 1])
    pred *= 100
    return {"probability": pred}
