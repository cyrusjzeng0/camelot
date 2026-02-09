import pandas as pd
import pickle
import os

def build_era_index(df):
    return {
        str(era): set(df.loc[df["era"] == era, "id"])
        for era in df["era"].unique()
    }
    
def build_camelot_index(df):
    return {
        str(c): set(df.loc[df["camelot"] == c, "id"])
        for c in df["camelot"].unique()
    }
    
def save_filter_indices(
    era_index: dict[str, set[int]], 
    camelot_index: dict[str, set[int]], 
    path: str):
    
    with open(os.path.join(path, "era_index.pkl"), "wb") as f:
        pickle.dump(era_index, f)

    with open(os.path.join(path, "camelot_index.pkl"), "wb") as f:
        pickle.dump(camelot_index, f)
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) # find directory this file is in
    output_dir = os.path.join(script_dir, "../data")
    
    data = pd.read_parquet(os.path.join(output_dir, "tracks.parquet"))
    era_idx = build_era_index(data)
    camelot_idx = build_camelot_index(data)
    
    save_filter_indices(era_idx, camelot_idx, output_dir)
    
if __name__ == "__main__":
    main()