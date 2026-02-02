from fastapi import FastAPI
from .data import load_tracks

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/tracks")
def get_tracks():
    df = load_tracks()
    return df.head(50).to_dict(orient="records")