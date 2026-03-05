from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

# Recommender integration: load and call recommend() using session.current_track_id
try:
    from app.recommender import build_recommender_state, recommend
except ImportError:
    build_recommender_state = recommend = None  # allow queue to run without recommender


# === data models ===

@dataclass
class Session:
    id: int
    dj_id: str
    created_at: datetime
    status: str = "active"  # "active" | "ended"
    current_track_id: Optional[int] = None
    era_preference: Optional[str] = None  # "modern" | "throwback" | None
    audience_ids: List[str] = None  # will be initialized in __post_init__
 
    def __post_init__(self) -> None: # creates new default list per instance; runs immediately after above auto-generated __init__()
        if self.audience_ids is None:
            self.audience_ids = []

@dataclass
class QueueItem:
    id: int
    session_id: int
    requester_id: str
    track_id: int
    created_at: datetime
    status: str = "pending"  # "pending" | "accepted" | "rejected" | "played"
    upvotes: int = 0
    note: Optional[str] = None


# === manager (makeshift database) ===

class QueueManager:
    """
    in-memory manager for DJ sessions and audience request queues.
    later, replace internal dicts with actual database layer
    """

    def __init__(self) -> None:
        self._sessions: Dict[int, Session] = {}
        self._queue_items: Dict[int, QueueItem] = {}
        self._next_session_id: int = 1
        self._next_item_id: int = 1

    # -- Session operations --

    def create_session(
        self,
        dj_id: str,
        era_preference: Optional[str] = None,
        name: Optional[str] = None,  # placeholder for future use
    ) -> Session:
        """
        creates a new active DJ session. for now, we allow multiple active sessions per DJ; can later enforce 'only one active session per DJ' if required
        """
        session_id = self._next_session_id
        self._next_session_id += 1

        session = Session(
            id=session_id,
            dj_id=dj_id,
            created_at=datetime.now(datetime.UTC),
            status="active",
            current_track_id=None,
            era_preference=era_preference,
        )
        self._sessions[session_id] = session
        return session

    def end_session(self, session_id: int) -> None:
        """mark a session as ended if it exists"""
        session = self._sessions.get(session_id)
        if session is not None:
            session.status = "ended"

    def get_session(self, session_id: int) -> Optional[Session]:
        """return a session by id, or None if not found"""
        return self._sessions.get(session_id)

    def add_audience_member(self, session_id: int, user_id: str) -> None:
        """
        track that a given audience member has joined the session;
        no-op if the session does not exist or user is already present
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        if user_id not in session.audience_ids:
            session.audience_ids.append(user_id)

    # -- Queue operations --

    def enqueue_request(
        self,
        session_id: int,
        requester_id: str,
        track_id: int,
        note: Optional[str] = None,
    ) -> QueueItem:
        """
        Add a new request to the session's queue.
        Raises ValueError if the session does not exist or is not active.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"session {session_id} does not exist")
        if session.status != "active":
            raise ValueError(f"session {session_id} is not active")

        item_id = self._next_item_id
        self._next_item_id += 1

        item = QueueItem(
            id=item_id,
            session_id=session_id,
            requester_id=requester_id,
            track_id=track_id,
            created_at=datetime.now(datetime.UTC),
            status="pending",
            upvotes=0,
            note=note,
        )
        self._queue_items[item_id] = item
        return item

    def list_queue(
        self,
        session_id: int,
        status: Optional[str] = "pending",
    ) -> List[QueueItem]:
        """
        Return queue items for a session, optionally filtered by status.
        Items are ordered by created_at ascending (oldest first).
        """
        items = [
            item
            for item in self._queue_items.values()
            if item.session_id == session_id
        ]
        if status is not None:
            items = [item for item in items if item.status == status]

        items.sort(key=lambda i: i.created_at)
        return items

    def _get_item_or_raise(self, item_id: int) -> QueueItem:
        item = self._queue_items.get(item_id)
        if item is None:
            raise ValueError(f"queue item {item_id} does not exist")
        return item

    def accept_request(self, item_id: int) -> None:
        """
        Mark a request as accepted and update the session's current_track_id.
        """
        item = self._get_item_or_raise(item_id)
        session = self._sessions.get(item.session_id)
        if session is None:
            raise ValueError(f"session {item.session_id} does not exist")

        item.status = "accepted"
        session.current_track_id = item.track_id

    def reject_request(self, item_id: int) -> None:
        """Mark a request as rejected."""
        item = self._get_item_or_raise(item_id)
        item.status = "rejected"

    def mark_played(self, item_id: int) -> None:
        """Mark a request as played."""
        item = self._get_item_or_raise(item_id)
        item.status = "played"

    def upvote_request(self, item_id: int) -> None:
        """Increment the upvote count for a request."""
        item = self._get_item_or_raise(item_id)
        item.upvotes += 1


def main() -> None:
    """
    Simple CLI to create a session, enqueue requests, accept/reject, and get
    next-track recommendations (recommender ↔ queue link).
    """
    manager = QueueManager()
    print("🎛️ creating demo session for dj 'demo_dj'...")
    session = manager.create_session(dj_id="demo_dj", era_preference=None)
    print(f"✅ created session with id={session.id}")

    # Load recommender state once so we can call recommend() after accept
    rec_state = None
    if build_recommender_state is not None and recommend is not None:
        print("🧮  Loading recommender state (one-time)...")
        rec_state = build_recommender_state()
        print("✅  Recommender ready. Use [n] after accepting a request.")
    else:
        print("ℹ️  Recommender not available; [n] will be disabled.")

    while True:
        cmd = input(
            "\nCommands: [e]nqueue, [l]ist, [a]ccept, [r]eject, [p]layed, [n]ext (recommend), [q]uit\n> "
        ).strip()

        if cmd.lower() == "q":
            print("✅ exit successful")
            break
        elif cmd.lower() == "e":
            track_raw = input("track id: ").strip()
            if not track_raw.isdigit():
                print("❌ track id must be an integer")
                continue
            track_id = int(track_raw)
            requester = input("requester id (string is fine): ").strip() or "anon"
            note = input("optional note: ").strip() or None
            item = manager.enqueue_request(
                session_id=session.id,
                requester_id=requester,
                track_id=track_id,
                note=note,
            )
            print(f"✅ enqueued item id={item.id}")
        elif cmd.lower() == "l":
            items = manager.list_queue(session_id=session.id, status="pending")
            if not items:
                print("ℹ️  no pending items")
            else:
                for item in items:
                    print(
                        f"id={item.id} track_id={item.track_id} "
                        f"requester={item.requester_id} status={item.status} "
                        f"upvotes={item.upvotes} note={item.note}"
                    )
        elif cmd.lower() == "a":
            raw = input("item id to accept: ").strip()
            if not raw.isdigit():
                print("❌ item id must be an integer")
                continue
            manager.accept_request(int(raw))
            print("✅ accepted")
        elif cmd.lower() == "r":
            raw = input("item id to reject: ").strip()
            if not raw.isdigit():
                print("❌ item id must be an integer")
                continue
            manager.reject_request(int(raw))
            print("✅ rejected")
        elif cmd.lower() == "p":
            raw = input("item id to mark played: ").strip()
            if not raw.isdigit():
                print("❌ item id must be an integer")
                continue
            manager.mark_played(int(raw))
            print("✅ marked as played")
        elif cmd.lower() == "n":
            # Next-track recommendations using session.current_track_id (set when DJ accepts)
            if rec_state is None:
                print("❌ recommender not available")
                continue
            song_meta, features, camelot_index, era_index = rec_state
            song_id = session.current_track_id
            if song_id is None:
                raw = input("No current track. Accept a request first, or enter a track id: ").strip()
                if not raw.isdigit():
                    print("❌ track id must be an integer")
                    continue
                song_id = int(raw)
            if song_id not in song_meta.index:
                print("❌ track id not found in catalog")
                continue
            recs = recommend(
                song_id=song_id,
                song_meta=song_meta,
                features=features,
                camelot_index=camelot_index,
                era_index=era_index,
                era=session.era_preference,
                top_k=10,
            )
            if not recs:
                print("❌ no recommendations found")
                continue
            print(f"🎧 Next-track recommendations for id {song_id} ({song_meta.loc[song_id, 'name']}):")
            for r in recs:
                print(
                    f"  id: {r['id']}\tname: {r['name']}\tartists: {r['artists']}\t"
                    f"camelot: {r['camelot']}\tbpm: {r['tempo']:.1f}\tera: {r['era']}\tscore: {r['score']:.1f}"
                )
        else:
            print("❌ unknown command")


if __name__ == "__main__":
    main()