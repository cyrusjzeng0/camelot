from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class User:
    id: int
    display_name: str
    roles: set[str]  # e.g. {"dj"}, {"audience"}, or both
    created_at: datetime


@dataclass
class DJProfile:
    user_id: int
    bio: str | None = None
    favorite_era: str | None = None  # "modern" | "throwback" | None
    favorite_bpm_range: tuple[float, float] = (0.0, 300.0)
    favorite_energy_range: tuple[float, float] = (0.0, 1.0)
    favorite_valence_range: tuple[float, float] = (0.0, 1.0)
    favorite_danceability_range: tuple[float, float] = (0.0, 1.0)
    favorite_acousticness_range: tuple[float, float] = (0.0, 1.0)
    favorite_artists: list[tuple[str, float]] = None  # (artist_name, count or weight)
    total_sets_played: int = 0
    total_requests_taken: int = 0

    def __post_init__(self) -> None:
        if self.favorite_artists is None:
            self.favorite_artists = []


@dataclass
class AudienceProfile:
    user_id: int
    bio: str | None = None
    preferred_era: str | None = None
    preferred_bpm_range: tuple[float, float] = (0.0, 300.0)
    preferred_energy_range: tuple[float, float] = (0.0, 1.0)
    preferred_valence_range: tuple[float, float] = (0.0, 1.0)
    preferred_danceability_range: tuple[float, float] = (0.0, 1.0)
    preferred_acousticness_range: tuple[float, float] = (0.0, 1.0)
    preferred_artists: list[tuple[str, float]] = None  # (artist_name, count or weight)
    total_requests_made: int = 0
    total_requests_accepted: int = 0

    def __post_init__(self) -> None:
        if self.preferred_artists is None:
            self.preferred_artists = []
            
# === manager (makeshift database) ===

class ProfileManager:
    """
    in-memory manager for user profiles. add functions to create, update, delete, and get profiles later; also replace internal dicts with actual database layer later
    """
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._dj_profiles: Dict[int, DJProfile] = {}
        self._audience_profiles: Dict[int, AudienceProfile] = {}
        