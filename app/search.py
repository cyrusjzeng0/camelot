from build.build_search_index import normalize_text
import os
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # find directory this file is in
DATA_DIR = os.path.join(BASE_DIR, "../data") # move to data directory

def load_pickle(name: str):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        return pickle.load(f)
    
def load_indices():
    return {
        "search": load_pickle("search_index.pkl"),
        "era": load_pickle("era_index.pkl"),
        "camelot": load_pickle("camelot_index.pkl")
    }

def search(
    query: str,
    search_index: dict,
    era_index: dict[str, set[int]] | None = None,
    top_k: int=10, 
    era: str = ""
) -> list[int]:
    
    normalized_q = normalize_text(query) # returns normalized string
    if not normalized_q:
        return []
    
    tokens = normalized_q.split()
    full_tokens = set(tokens[:-1])
    prefix_token = tokens[-1]

    inverted_index = search_index["inverted_index"]
    prefix_index = search_index["prefix_index"]
    
    idx_set = None
    
    for token in full_tokens:
        curr_indices = inverted_index.get(token, set())
        if idx_set is None:
            idx_set = curr_indices.copy()
        else:
            idx_set &= curr_indices # set intersection
    
    prefix_matches = prefix_index.get(prefix_token, set()) # gets set of possible full tokens associated with prefix
    prefix_idx_set = set()
    for tok in prefix_matches:
        prefix_idx_set |= inverted_index.get(tok, set()) # set union and assign back
        
    if full_tokens:
        idx_set &= prefix_idx_set
    else:
        idx_set = prefix_idx_set
    
    if idx_set is None:
        return []
        
    if era and era_index:
        idx_set &= era_index.get(era, set())
        
    return list(idx_set)[:top_k]

def main():
    indices = load_indices()
    search_idx = indices["search"]
    era_idx = indices["era"]
    ids = search(
        query="taylor sw",
        search_index=search_idx,
        era_index=era_idx,
        top_k=10,
        era="modern"
    )
    
    print(ids)
    
if __name__ == "__main__":
    main() # run using "python -m app.search"; safe to delete pycache
    