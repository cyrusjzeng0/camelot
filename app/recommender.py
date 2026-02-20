import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # find directory this file is in
DATA_DIR = os.path.join(BASE_DIR, "../data") # move to data directory

def load_pickle(name: str):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        return pickle.load(f)
    
def load_indices():
    return {
        "era": load_pickle("era_index.pkl"),
        "camelot": load_pickle("camelot_index.pkl")
    }


def harmonic_neighbors(camelot: str) -> list[str]:
    key = int(camelot[:-1])
    mode = camelot[-1]
    other_mode = "A" if mode == "B" else "B"

    return [
        camelot,
        f"{((key - 2) % 12) + 1}{mode}",
        f"{(key % 12) + 1}{mode}",
        f"{key}{other_mode}",
    ]
    
def get_candidate_ids(song_id: int,
                      song_meta: dict[int, dict],
                      camelot_index: dict[str, set[int]],
                      era_index: dict[str, set[int]] | None = None,
                      era: str | None = None) -> set[int]:
    pass
    
def recommend(song_id: int,
              ) -> list:
    pass