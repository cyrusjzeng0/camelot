# Camelot

DJ recommender and audience queue platform. Recommends next tracks using Camelot key compatibility, BPM range, and audio-feature similarity. Audience members can submit requests to a live session queue.

## Repo components

- **Build** (`build/`) – Offline pipeline: download Kaggle Spotify 12M dataset, deduplicate, engineer Camelot + era, write `tracks.parquet` and `features.npy`, then build search and filter indices into `data/`.
- **App** (`app/`) – Runtime:
  - **search** – Text search over track name/artists with optional era filter.
  - **recommender** – Next-track recommendations (harmonic keys, ±15% BPM, feature similarity).
  - **queue** – In-memory session + audience request queue; CLI includes a “[n]ext” flow that calls the recommender after the DJ accepts a request.
  - **profiles** – Data models for User, DJProfile, AudienceProfile (and optional ProfileManager).

## Quick start

Run everything from the **project root** (where `app/` and `build/` live).

### 1. Build data (one-time or after dataset changes)

```bash
python -m build.build_dataset    # downloads Kaggle data, writes tracks.parquet + features.npy
python -m build.build_search_index
python -m build.build_filter_indices
```

Order matters: dataset first, then search index, then filter indices.

### 2. Run the CLIs

```bash
python -m app.search       # search tracks by query
python -m app.recommender  # get next-track recommendations by song id
python -m app.queue        # create session, enqueue requests, accept/reject, [n] for recommendations
```

## Project layout

```
camelot/
├── README.md
├── app/
│   ├── search.py       # search API + CLI
│   ├── recommender.py  # recommend(), get_candidate_ids(), build_recommender_state(), CLI
│   ├── queue.py        # Session, QueueItem, QueueManager, CLI
│   └── profiles.py     # User, DJProfile, AudienceProfile, ProfileManager
├── build/
│   ├── build_dataset.py        # tracks.parquet, features.npy
│   ├── build_search_index.py   # search_index.pkl
│   └── build_filter_indices.py # era_index.pkl, camelot_index.pkl
├── data/               # generated (parquet, npy, pkl) – create and populate via build
└── notebooks/
    └── exploration.ipynb
```

## Next steps

- **Web API** – FastAPI routes for search, recommend, sessions, and queue (calling existing `app` modules).
- **Persistence** – Replace in-memory queue and profile stores with a database.
- **Auth** – DJ vs audience roles and session ownership.

---

## Additional documentation (coming soon!)

- `docs/BUILD.md` – Detailed build pipeline, column meanings, index formats.
- `docs/API.md` – Once you have an API: endpoints, request/response shapes.
- `docs/ARCHITECTURE.md` – Data flow, how queue/recommender/profiles connect.
