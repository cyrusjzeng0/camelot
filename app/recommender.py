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
    
def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA_DIR, "tracks.parquet"))

def load_features() -> np.ndarray:
    return np.load(os.path.join(DATA_DIR, "features.npy"))

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
                      song_meta: pd.DataFrame,
                      camelot_index: dict[str, set[int]],
                      era_index: dict[str, set[int]] | None = None,
                      era: str | None = None) -> set[int]:
    song_camelot = song_meta.loc[song_id, "camelot"]
    song_tempo = song_meta.loc[song_id, "tempo"]
    low_bpm, high_bpm = song_tempo * 0.85, song_tempo * 1.15
    neighbor_keys = harmonic_neighbors(song_camelot)
    candidate_ids = set()

    for key in neighbor_keys:
        candidate_ids.update(camelot_index.get(key, set()))
    if era and era_index is not None:
        candidate_ids &= era_index.get(era, set())
    candidate_ids.discard(song_id) # discard() does nothing if song_id is not in set; remove() raises KeyError if song_id not in set

    if not candidate_ids:
        return set()

    candidates_df = song_meta.loc[list(candidate_ids)]
    candidates_df = candidates_df[
        (candidates_df["tempo"] >= low_bpm) &
        (candidates_df["tempo"] <= high_bpm)
    ]
    return set(candidates_df.index)


def recommend(song_id: int,
              song_meta: pd.DataFrame,
              features: np.ndarray,
              camelot_index: dict[str, set[int]],
              era_index: dict[str, set[int]] | None = None,
              era: str | None = None,
              top_k: int = 10) -> list[dict]:
    candidate_ids = get_candidate_ids(song_id, song_meta, camelot_index, era_index, era)
    if not candidate_ids:
        return []

    candidate_ids_list = list(candidate_ids)
    candidate_features = features[candidate_ids_list]
    curr_song_vec = features[song_id].reshape(1, -1)

    cos_sim = cosine_similarity(curr_song_vec, candidate_features)[0]
    euc_dist = euclidean_distances(curr_song_vec, candidate_features)[0]
    euc_sim = 1 / (1 + euc_dist)

    hybrid_sim = np.power((0.4 * cos_sim) + (0.6 * euc_sim), 3)
    hybrid_sim = np.clip(hybrid_sim, 0, 1)

    id_score_pairs = list(zip(candidate_ids_list, hybrid_sim))
    id_score_pairs.sort(key=lambda x: x[1], reverse=True)
    id_score_pairs = id_score_pairs[:top_k]

    results: list[dict] = []
    for track_id, score in id_score_pairs:
        row = song_meta.loc[track_id]
        results.append(
            {
                "id": int(track_id),
                "name": row["name"],
                "artists": row["artists"],
                "camelot": row["camelot"],
                "tempo": float(row["tempo"]),
                "era": row["era"],
                "score": float(score * 100.0),
            }
        )

    return results


def build_recommender_state():
    indices = load_indices()
    song_meta = load_metadata().set_index("id")
    features = load_features()
    return song_meta, features, indices["camelot"], indices["era"]


def main():
    print("🧮 loading recommender state... ")
    song_meta, features, camelot_index, era_index = build_recommender_state()

    while True:
        try:
            raw = input("💬 ENTER CURRENT SONG ID (or blank to exit): ")
        except EOFError:
            break

        if not raw.strip():
            print("✅ exit successful")
            break

        if not raw.isdigit():
            print("❌ please enter a numeric song id")
            continue

        song_id = int(raw)
        if song_id not in song_meta.index:
            print("❌ song id not found")
            continue

        recs = recommend(
            song_id=song_id,
            song_meta=song_meta,
            features=features,
            camelot_index=camelot_index,
            era_index=era_index,
            era=None,
            top_k=10,
        )

        if not recs:
            print("❌ no recommendations found")
            continue

        print(f"🎧 recommendations for id {song_id} ({song_meta.loc[song_id, 'name']}):")
        for r in recs:
            print(
                f"id: {r['id']}\t"
                f"name: {r['name']}\t"
                f"artists: {r['artists']}\t"
                f"camelot: {r['camelot']}\t"
                f"bpm: {r['tempo']:.1f}\t"
                f"era: {r['era']}\t"
                f"score: {r['score']:.1f}"
            )


if __name__ == "__main__":
    main() # run using "python -m app.recommender"; safe to delete pycache