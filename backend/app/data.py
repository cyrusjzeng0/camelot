import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data/tracks.parquet"

def load_tracks():
    return pd.read_parquet(DATA_PATH)