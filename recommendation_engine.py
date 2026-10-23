
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# TUNETRAIL - PHASE 1: MUSIC RECOMMENDATION ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FILE_3_SEC = BASE_DIR / "features_3_sec.csv"
FILE_30_SEC = BASE_DIR / "features_30_sec.csv"
FEEDBACK_FILE = BASE_DIR / "tune_trail_feedback.json"

TOP_N = 10

# Recommendation score weights
SEED_WEIGHT = 0.65
LIKED_WEIGHT = 0.25
DISLIKED_WEIGHT = 0.35

# Artist repetition penalty (used when artist metadata exists)
ARTIST_PENALTY = 0.25


# ============================================================
# 1. LOAD AND CLEAN THE DATASETS
# ============================================================

def get_track_id(filename):
    """Return a stable song ID while collapsing only 3-second segment suffixes.

    GTZAN 30-second files look like ``blues.00000.wav`` and must keep the
    ``blues.00000`` ID. Three-second segment files look like
    ``blues.00000.0.wav`` and should collapse to ``blues.00000``.
    """
    name = Path(str(filename)).stem.lower()
    parts = name.split(".")
    if len(parts) >= 3 and parts[-1].isdigit():
        name = ".".join(parts[:-1])
    return name


def load_catalog():
    """Prefer full-track features; use 3-second features as fallback."""

    if not FILE_3_SEC.exists():
        raise FileNotFoundError(
            f"Dataset not found: {FILE_3_SEC}"
        )

    df3 = pd.read_csv(FILE_3_SEC)

    if "filename" not in df3 or "label" not in df3:
        raise ValueError(
            "The 3-second CSV must contain filename and label columns."
        )

    df3["track_id"] = df3["filename"].map(get_track_id)

    # Aggregate ten segments into one row per original song.
    numeric3 = [
        c for c in df3.select_dtypes(include=np.number).columns
        if c != "length"
    ]

    agg3 = df3.groupby("track_id")[numeric3].mean()

    labels3 = df3.groupby("track_id")["label"].first()
    agg3["label"] = labels3

    # Preserve artist metadata if a future catalog supplies it.
    if "artist" in df3.columns:
        artists3 = df3.groupby("track_id")["artist"].first()
        agg3["artist"] = artists3

    agg3 = agg3.reset_index()

    # Use the 30-second catalog as the primary source when available.
    if FILE_30_SEC.exists():
        df30 = pd.read_csv(FILE_30_SEC)

        if "filename" not in df30 or "label" not in df30:
            raise ValueError(
                "The 30-second CSV must contain filename and label columns."
            )

        df30["track_id"] = df30["filename"].map(get_track_id)
        df30 = df30.drop_duplicates("track_id", keep="first")

        # Fill songs missing from the full-track catalog using 3-second
        # aggregate features. Prefer 30-second features for matched songs.
        missing_ids = set(agg3["track_id"]) - set(df30["track_id"])
        fallback = agg3[agg3["track_id"].isin(missing_ids)]

        catalog = pd.concat(
            [df30, fallback],
            ignore_index=True,
            sort=False
        )
    else:
        catalog = agg3.copy()

    catalog = catalog.drop_duplicates("track_id").reset_index(drop=True)

    # Audio features only. Never use labels, filenames, or artist IDs
    # as numerical audio features.
    excluded = {
        "filename", "track_id", "label", "artist",
        "length", "artist_id", "track_name", "song_name"
    }

    feature_columns = [
        c for c in catalog.select_dtypes(include=np.number).columns
        if c not in excluded
    ]

    if not feature_columns:
        raise ValueError("No numerical audio features were found.")

    # Store optional metadata separately.
    if "artist" not in catalog.columns:
        catalog["artist"] = None

    catalog["artist"] = catalog["artist"].replace(
        {"": None, "nan": None, "None": None}
    )

    # Ensure every feature has usable values.
    features = catalog[feature_columns].copy()
    features = features.replace([np.inf, -np.inf], np.nan)

    # Drop columns that have no usable values.
    feature_columns = [
        c for c in feature_columns if features[c].notna().any()
    ]
    features = features[feature_columns]

    # Fill missing values with each column's median.
    features = features.fillna(features.median()).fillna(0)

    # Scale features so large-valued columns do not dominate similarity.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print(f"Catalog loaded: {len(catalog)} unique tracks")
    print(f"Audio features: {len(feature_columns)}")

    if catalog["artist"].notna().any():
        print("Artist metadata: available for some tracks")
    else:
        print(
            "Artist metadata: unavailable in GTZAN; "
            "artist-diversity penalty will activate when metadata is added."
        )

    return catalog, feature_columns, scaled_features


# ============================================================
# 2. PERSISTENT USER FEEDBACK
# ============================================================

def load_feedback():
    if not FEEDBACK_FILE.exists():
        return {
            "likes": [],
            "dislikes": [],
            "skips": []
        }

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "likes": data.get("likes", []),
            "dislikes": data.get("dislikes", []),
            "skips": data.get("skips", [])
        }

    except (json.JSONDecodeError, OSError):
        print("Warning: feedback file could not be read. Starting fresh.")
        return {
            "likes": [],
            "dislikes": [],
            "skips": []
        }


def save_feedback(feedback):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)


def record_feedback(track_id, action, feedback):
    """Save a like, dislike, or skip and remove conflicting feedback."""

    action = action.lower()

    if action not in {"like", "dislike", "skip"}:
        raise ValueError("Action must be like, dislike, or skip.")

    track_id = str(track_id)

    for category in ("likes", "dislikes", "skips"):
        feedback[category] = [
            t for t in feedback[category] if t != track_id
        ]

    category = {
        "like": "likes",
        "dislike": "dislikes",
        "skip": "skips"
    }[action]

    feedback[category].append(track_id)
    save_feedback(feedback)

    print(f"Recorded {action}: {track_id}")


# ============================================================
# 3. RECOMMENDATION ENGINE
# ============================================================

class TuneTrailRecommender:

    def __init__(self):
        (
            self.catalog,
            self.feature_columns,
            self.features
        ) = load_catalog()

        self.feedback = load_feedback()

        self.track_to_index = {
            track_id: i
            for i, track_id in enumerate(self.catalog["track_id"])
        }

        self.track_ids = self.catalog["track_id"].tolist()

        # Precompute all pairwise similarities.
        self.similarity_matrix = cosine_similarity(self.features)

        # Convert cosine similarity from [-1, 1] to [0, 1].
        self.similarity_matrix = (
            self.similarity_matrix + 1.0
        ) / 2.0

    def find_tracks(self, query, limit=15):
        """Search by track ID or genre."""

        query = query.lower().strip()

        matches = self.catalog[
            self.catalog["track_id"].str.contains(
                query, case=False, regex=False
            )
            | self.catalog["label"].astype(str).str.contains(
                query, case=False, regex=False
            )
        ]

        return matches.head(limit)[
            ["track_id", "label", "artist"]
        ]

    def get_index(self, track_id):
        track_id = str(track_id).strip().lower()

        if track_id not in self.track_to_index:
            raise ValueError(
                f"Track '{track_id}' not found. "
                "Use the search command to find valid track IDs."
            )

        return self.track_to_index[track_id]

    def recommend(
        self,
        seed_ids=None,
        top_n=TOP_N,
        diversity=0.75
    ):
        """
        Rank songs using seed similarity, feedback, and artist diversity.

        diversity:
            0.0 = no artist repetition penalty
            1.0 = strong artist repetition penalty
        """

        seed_ids = seed_ids or []

        # The user's saved feedback becomes part of their taste profile.
        liked_ids = self.feedback["likes"]
        disliked_ids = self.feedback["dislikes"]
        skipped_ids = self.feedback["skips"]

        seed_ids = list(dict.fromkeys(str(t).strip().lower() for t in seed_ids))
        liked_ids = list(dict.fromkeys(str(t).strip().lower() for t in liked_ids))
        disliked_ids = list(dict.fromkeys(str(t).strip().lower() for t in disliked_ids))
        skipped_ids = list(dict.fromkeys(str(t).strip().lower() for t in skipped_ids))

        if not seed_ids and not liked_ids:
            raise ValueError(
                "Choose at least one seed song or like a song first."
            )

        seed_indices = [
            self.get_index(t) for t in seed_ids
            if t in self.track_to_index
        ]
        liked_indices = [
            self.get_index(t) for t in liked_ids
            if t in self.track_to_index
        ]
        negative_indices = [
            self.get_index(t) for t in disliked_ids
            if t in self.track_to_index
        ]

        if not seed_indices and not liked_indices:
            raise ValueError(
                "No valid seed or liked GTZAN tracks were found."
            )

        # Keep seed similarity and saved-like similarity separate.
        seed_scores = (
            self.similarity_matrix[:, seed_indices].mean(axis=1)
            if seed_indices else np.zeros(len(self.catalog))
        )
        liked_scores = (
            self.similarity_matrix[:, liked_indices].mean(axis=1)
            if liked_indices else np.zeros(len(self.catalog))
        )

        # A disliked song lowers the ranking of similar candidates.
        negative_scores = (
            self.similarity_matrix[:, negative_indices].mean(axis=1)
            if negative_indices else np.zeros(len(self.catalog))
        )

        scores = (
            SEED_WEIGHT * seed_scores
            + LIKED_WEIGHT * liked_scores
            - DISLIKED_WEIGHT * negative_scores
        )

        positive_ids = list(dict.fromkeys(seed_ids + liked_ids))
        excluded_ids = set(
            positive_ids + disliked_ids + skipped_ids
        )

        candidates = [
            i for i, track_id in enumerate(self.track_ids)
            if track_id not in excluded_ids
        ]

        # Greedy ranking applies the diversity penalty after each choice.
        # This works only when actual artist metadata is available.
        results = []
        chosen_artists = set()

        has_artist_metadata = self.catalog["artist"].notna().any()

        while candidates and len(results) < top_n:
            best_idx = None
            best_score = -float("inf")

            for idx in candidates:
                score = float(scores[idx])

                artist = self.catalog.iloc[idx]["artist"]

                if (
                    has_artist_metadata
                    and pd.notna(artist)
                    and artist in chosen_artists
                ):
                    score -= ARTIST_PENALTY * diversity

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            row = self.catalog.iloc[best_idx]
            artist = row["artist"]

            results.append({
                "track_id": row["track_id"],
                "genre": row["label"],
                "artist": artist if pd.notna(artist) else "Unknown",
                "score": round(best_score, 4)
            })

            if pd.notna(artist):
                chosen_artists.add(artist)

            candidates.remove(best_idx)

        return pd.DataFrame(results)


# ============================================================
# 4. INTERACTIVE TERMINAL DEMO
# ============================================================

def main():
    recommender = TuneTrailRecommender()

    print("\n" + "=" * 55)
    print("          TUNETRAIL MUSIC DISCOVERY")
    print("=" * 55)

    print("""
Commands:
  search <text>           Search songs or genres
  recommend <id> [id...]  Recommend using favorite songs
  recommend               Use saved likes as preferences
  like <id>               Like a song
  dislike <id>            Dislike a song
  skip <id>               Skip a song
  feedback                Show saved feedback
  help                    Show commands
  quit                    Exit
""")

    while True:
        try:
            command = input("TuneTrail > ").strip()

            if not command:
                continue

            parts = command.split()
            action = parts[0].lower()
            args = parts[1:]

            if action in {"quit", "exit"}:
                print("Thanks for using TuneTrail!")
                break

            elif action == "help":
                print(
                    "Use search <text>, recommend <track_id>, "
                    "like <track_id>, dislike <track_id>, "
                    "skip <track_id>, feedback, or quit."
                )

            elif action == "search":
                if not args:
                    print("Example: search blues")
                    continue

                results = recommender.find_tracks(" ".join(args))

                if results.empty:
                    print("No matches found.")
                else:
                    print(results.to_string(index=False))

            elif action == "recommend":
                # Optional song IDs may be separated by spaces.
                seeds = args

                results = recommender.recommend(
                    seed_ids=seeds,
                    top_n=TOP_N,
                    diversity=0.75
                )

                print("\nRecommended songs:")
                print(results.to_string(index=False))

                if not recommender.catalog["artist"].notna().any():
                    print(
                        "\nNote: artist diversity is awaiting artist "
                        "metadata. GTZAN does not identify its artists."
                    )

            elif action in {"like", "dislike", "skip"}:
                if len(args) != 1:
                    print(f"Usage: {action} <track_id>")
                    continue

                track_id = args[0].lower()
                recommender.get_index(track_id)

                record_feedback(
                    track_id,
                    action,
                    recommender.feedback
                )

            elif action == "feedback":
                print(json.dumps(recommender.feedback, indent=2))

            else:
                print("Unknown command. Type help for options.")

        except ValueError as e:
            print("Error:", e)

        except KeyboardInterrupt:
            print("\nExiting TuneTrail.")
            break


if __name__ == "__main__":
    main()
