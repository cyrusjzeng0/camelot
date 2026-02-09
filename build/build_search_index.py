"""
build_search_index.py
---------------------
given partial user search query, finds top matching songs fast
"""

import pandas as pd
import os
import re # regex
from collections import defaultdict
import pickle

# load tracks.parquet and return required cols
def load_metadata(path: str) -> pd.DataFrame:
    SEARCH_COLS = ["id", "name", "artists", "year", "era"]
    df = pd.read_parquet(path)[SEARCH_COLS]
    
    return df

# lowercase, remove punctuation + whitespace
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # remove punctuation
    text = re.sub(r"\s+", " ", text) # collapse spaces
    return text.strip()

# extract normalized word tokens from each song's name + artist names
def extract_tokens(name: str, artists: list[str]) -> set[str]:
    tokens = set()
    
    # song title
    name_norm = normalize_text(name)
    tokens.update(name_norm.split())
    
    # artists
    for artist in artists:
        artist_norm = normalize_text(artist)
        tokens.update(artist_norm.split())
        
    return tokens

# build token -> set(id) inverted index for entire dataframe
def build_inverted_index(df: pd.DataFrame) -> dict[str, set[int]]:
    inverted_index = defaultdict(set)
    
    for song_id, name, artists in zip(df["id"], df["name"], df["artists"]):
        tokens = extract_tokens(name, artists)
        for token in tokens:
            inverted_index[token].add(song_id)

    return dict(inverted_index)

# map prefix -> set of full tokens. max_prefix_len caps user query (most tokens are under 10 chars in length) for entire dataframe
def build_prefix_index(tokens: set[str], min_prefix_len: int = 2, max_prefix_len: int = 10) -> dict[str, set[str]]:
    prefix_index = defaultdict(set)

    for token in tokens: # tokens = keys of inverted index dict
        token_len = len(token)
        upper = min(token_len, max_prefix_len)

        for i in range(min_prefix_len, upper + 1):
            prefix = token[:i]
            prefix_index[prefix].add(token)

    # note: output type is hashmap, but this could also be implemented using a trie for more effient data storage (current method explodes prefixes)
    return dict(prefix_index)

# combine search structures into one dict object
def assemble_search_index(inverted_index: dict[str, set[int]], prefix_index: dict[str, set[str]]) -> dict:
    return {
        "inverted_index": inverted_index,
        "prefix_index": prefix_index
    }
    
def save_search_index(search_index: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(search_index, f)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) # find directory this file is in
    output_dir = os.path.join(script_dir, "../data")
    print("📊 loading data from tracks.parquet... ")
    df = load_metadata(os.path.join(output_dir, "tracks.parquet"))
    print("👷‍♂️ building inverted index dict... ")
    inverted_index = build_inverted_index(df)

    all_tokens = set(inverted_index.keys())
    print("👷‍♂️ building prefix index dict... ")
    prefix_index = build_prefix_index(all_tokens)
    print("🧱 merging inverted index and prefix index dicts... ")
    search_index = assemble_search_index(inverted_index, prefix_index)
    print("✍️ writing search_index.pkl... ")
    save_search_index(search_index, os.path.join(output_dir, "search_index.pkl"))
    print("✅ files written!")
    
if __name__ == "__main__":
    main()