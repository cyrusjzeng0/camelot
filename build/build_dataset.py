import kagglehub
import ast
import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

RAW_METADATA_COLS = ["id", "name", "artists", "key", "mode", "tempo", "year"]
FEATURE_COLS = ["energy", "valence", "danceability", "acousticness", "energy*valence"]

FINAL_METADATA_COLS = ["id", "spot_id", "name", "artists", "tempo", "year", "era", "camelot"]

# import original datset from kaggle
def load_raw_data() -> pd.DataFrame:
    # download latest version of dataset
    path = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")
    
    # import relevant features
    df = pd.read_csv(os.path.join(path, "tracks_features.csv"), usecols=RAW_METADATA_COLS+FEATURE_COLS[:-1]).dropna()

    # feature engineering to take into account interplay between features
    df["energy*valence"] = df["energy"] * df["valence"]
    
    df = df.rename(columns={"id": "spot_id"})
    
    return df

def preprocess_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # parse + filter
    df["artists"] = df["artists"].apply(ast.literal_eval)
    df = df[df["year"] >= 1955]

    # dtypes
    df[["tempo", "year"]] = df[["tempo", "year"]].astype("int16")
    for feat in FEATURE_COLS:
        if df[feat].dtype == "float64":
            df[feat] = df[feat].astype("float32")

    # feature scaling
    vibe_transformer = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("scaler", MinMaxScaler())
    ])
    df[["valence", "energy*valence"]] = vibe_transformer.fit_transform(
        df[["valence", "energy*valence"]]
    )

    # assign id col
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index.astype("int32")

    return df

def compute_camelot_era(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # == CAMELOT ==     (0 = C, 11 = B)
    major_map = {0: '8B', 1: '3B', 2: '10B', 3: '5B', 4: '12B', 5: '7B', 
             6: '2B', 7: '9B', 8: '4B', 9: '11B', 10: '6B', 11: '1B'}
    minor_map = {0: '5A', 1: '12A', 2: '7A', 3: '2A', 4: '9A', 5: '4A', 
                6: '11A', 7: '6A', 8: '1A', 9: '8A', 10: '3A', 11: '10A'}
    
    # create two helper series
    major_col = df["key"].map(major_map)
    minor_col = df["key"].map(minor_map)

    # if mode = 1, choose from major dict, otherwise choose from minor dict
    df["camelot"] = major_col.where(df["mode"] == 1, minor_col)
    
    # == ERA ==
    props = df["year"].value_counts(normalize=True).sort_index(ascending=True)
    split_year = (props.cumsum() >= 0.4).idxmax() # splits 40/60 throwback/modern in dataset
    
    df["era"] = np.where(df["year"] <= split_year, "throwback", "modern") # optimized np if-else block
    df["era"] = df["era"].astype("category") # memory efficient dtype

    df = df.drop(columns=["key", "mode"])

    return df
    
def assert_invariants(df: pd.DataFrame):
    assert df["id"].is_monotonic_increasing
    assert (df["id"].values == np.arange(len(df))).all()

def write_files(df: pd.DataFrame):
    # get the directory where build_dataset.py lives
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # ensures it goes to /camelot/data/ even if you run from root
    # script_dir is /camelot/build, so ../data is /camelot/data
    output_dir = os.path.join(script_dir, "../data")
    os.makedirs(output_dir, exist_ok=True)

    df_pq = df[FINAL_METADATA_COLS] # to store song metadata
    df_np = df[FEATURE_COLS] # to store song features

    df_pq.to_parquet(os.path.join(output_dir, "tracks.parquet"), index=False)
    
    feature_matrix = df_np.to_numpy(dtype="float32")
    np.save(os.path.join(output_dir, "features.npy"), feature_matrix)
    
def main():
    print("📊 loading raw data from kaggle... ")
    df = load_raw_data()
    print("📊 preprocessing tracks... ")
    df = preprocess_tracks(df)
    print("🧮 computing camelot and era for each song... ")
    df = compute_camelot_era(df)
    assert_invariants(df)
    print("✍️ writing tracks.parquet and features.npy... ")
    write_files(df)
    print("✅ files written!")
    
if __name__ == "__main__":
    main()