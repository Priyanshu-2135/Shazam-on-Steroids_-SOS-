
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from recommendation_engine import TuneTrailRecommender
from music_catalog import search_music, search_recommendations


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SOS | Shazam-on-Steroids",
    page_icon="🎧",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
FEATURES_30 = BASE_DIR / "features_30_sec.csv"
FEATURES_3 = BASE_DIR / "features_3_sec.csv"
FEEDBACK_FILE = BASE_DIR / "catalog_feedback.json"

BATCH_SIZE = 10


# ---------------------------------------------------------
# Feedback storage
# ---------------------------------------------------------

def empty_feedback():
    return {
        "likes": {},
        "dislikes": [],
        "disliked_songs": {},
        "skips": [],
    }


def load_catalog_feedback():
    if not FEEDBACK_FILE.exists():
        return empty_feedback()

    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        data.setdefault("likes", {})
        data.setdefault("dislikes", [])
        data.setdefault("disliked_songs", {})
        data.setdefault("skips", [])

        # Normalize older feedback files without losing saved feedback.
        if not isinstance(data["likes"], dict):
            data["likes"] = {}
        if not isinstance(data["dislikes"], list):
            data["dislikes"] = []
        if not isinstance(data["disliked_songs"], dict):
            data["disliked_songs"] = {}
        if not isinstance(data["skips"], list):
            data["skips"] = []

        return data

    except (OSError, json.JSONDecodeError):
        return empty_feedback()


def save_catalog_feedback(feedback):
    with FEEDBACK_FILE.open("w", encoding="utf-8") as file:
        json.dump(feedback, file, indent=2, ensure_ascii=False)


def record_feedback(song, action):
    feedback = load_catalog_feedback()
    song_id = str(song["id"])

    if action == "like":
        
        feedback["likes"][song_id] = {
            "id": song_id,
            "title": song.get("title", "Unknown title"),
            "artist": song.get("artist", "Unknown artist"),
            "album": song.get("album", "Unknown album"),
            "genre": song.get("genre", "Unknown genre"),
            "artwork": song.get("artwork", ""),
            "preview": song.get("preview", ""),
            "store_url": song.get("store_url", ""),
        }

        feedback["dislikes"] = [
            item for item in feedback["dislikes"]
            if str(item) != song_id
        ]
        feedback["disliked_songs"].pop(song_id, None)
        feedback["skips"] = [
            item for item in feedback["skips"]
            if str(item) != song_id
        ]

    elif action == "dislike":
        feedback["likes"].pop(song_id, None)
        feedback["disliked_songs"][song_id] = {
            "id": song_id,
            "title": song.get("title", "Unknown title"),
            "artist": song.get("artist", "Unknown artist"),
            "album": song.get("album", "Unknown album"),
            "genre": song.get("genre", "Unknown genre"),
        }

        if song_id not in [str(x) for x in feedback["dislikes"]]:
            feedback["dislikes"].append(song_id)

        feedback["skips"] = [
            item for item in feedback["skips"]
            if str(item) != song_id
        ]

    elif action == "skip":
        if song_id not in [str(x) for x in feedback["skips"]]:
            feedback["skips"].append(song_id)

    save_catalog_feedback(feedback)

    # Remove this song from the current visible recommendation pool.
    pool = st.session_state.get("catalog_pool", [])
    st.session_state.catalog_pool = [
        item for item in pool
        if str(item.get("id")) != song_id
    ]

    visible = st.session_state.get("catalog_results", [])
    st.session_state.catalog_results = [
        item for item in visible
        if str(item.get("id")) != song_id
    ]


# ---------------------------------------------------------
# Personalize real-catalog recommendations from user feedback
# ---------------------------------------------------------

def _normalise_text(value):
    return str(value or "").strip().casefold()


def personalize_catalog_results(results, feedback, diversity=0.75):
    """
    Re-rank catalog search results using saved likes/dislikes.

    This uses catalog metadata (genre and artist), not extracted audio
    features. It is a transparent metadata-based personalization layer.
    """
    if not results:
        return []

    likes = list(feedback.get("likes", {}).values())
    disliked_songs = list(feedback.get("disliked_songs", {}).values())
    disliked_ids = {str(value) for value in feedback.get("dislikes", [])}
    skipped_ids = {str(value) for value in feedback.get("skips", [])}

    # Ignore old feedback entries where metadata was not saved.
    liked_genres = [_normalise_text(x.get("genre")) for x in likes]
    liked_artists = [_normalise_text(x.get("artist")) for x in likes]
    disliked_genres = [_normalise_text(x.get("genre")) for x in disliked_songs]
    disliked_artists = [_normalise_text(x.get("artist")) for x in disliked_songs]

    liked_genres = [x for x in liked_genres if x and x != "unknown genre"]
    liked_artists = [x for x in liked_artists if x and x != "unknown artist"]
    disliked_genres = [x for x in disliked_genres if x and x != "unknown genre"]
    disliked_artists = [x for x in disliked_artists if x and x != "unknown artist"]

    # Frequency-weight preferences so repeated likes influence the profile more.
    from collections import Counter
    liked_genre_counts = Counter(liked_genres)
    liked_artist_counts = Counter(liked_artists)
    disliked_genre_counts = Counter(disliked_genres)
    disliked_artist_counts = Counter(disliked_artists)

    ranked = []
    chosen_artists = set()
    remaining = []

    for song in results:
        song_id = str(song.get("id", ""))
        if song_id in disliked_ids or song_id in skipped_ids:
            continue

        genre = _normalise_text(song.get("genre"))
        artist = _normalise_text(song.get("artist"))
        score = float(song.get("discovery_score", 0.0) or 0.0)

        if genre:
            score += 0.18 * liked_genre_counts.get(genre, 0)
            score -= 0.22 * disliked_genre_counts.get(genre, 0)
        if artist:
            score += 0.12 * liked_artist_counts.get(artist, 0)
            score -= 0.18 * disliked_artist_counts.get(artist, 0)

        remaining.append((score, song, artist))

    # Greedy re-ranking: preference score first, with a modest repeated-artist
    # penalty controlled by the existing diversity slider.
    while remaining:
        best_position = max(
            range(len(remaining)),
            key=lambda i: remaining[i][0] - (
                0.25 * max(0.0, min(1.0, diversity))
                if remaining[i][2] and remaining[i][2] in chosen_artists
                else 0.0
            ),
        )
        base_score, song, artist = remaining.pop(best_position)
        song = dict(song)
        song["personalized_score"] = round(
            base_score - (
                0.25 * max(0.0, min(1.0, diversity))
                if artist and artist in chosen_artists else 0.0
            ), 4
        )
        ranked.append(song)
        if artist:
            chosen_artists.add(artist)

    return ranked


# ---------------------------------------------------------
# Load the GTZAN dataset
# ---------------------------------------------------------

@st.cache_data
def load_gtzan_tracks():
    if FEATURES_30.exists():
        df = pd.read_csv(FEATURES_30)
    elif FEATURES_3.exists():
        df = pd.read_csv(FEATURES_3)
    else:
        return pd.DataFrame()

    if "filename" not in df.columns or "label" not in df.columns:
        return pd.DataFrame()

    # The 3-second file has multiple segments for each original track.
    df = df.drop_duplicates(subset=["filename"]).copy()

    df["track_id"] = df["filename"].astype(str).apply(
        lambda name: Path(name).stem
    )

    return df.reset_index(drop=True)


@st.cache_resource
def load_recommender():
    return TuneTrailRecommender()


def get_recommendation_genres(recommender, seed_ids, tracks, diversity):
    """
    Try the existing recommendation engine first.
    Use the selected GTZAN tracks' genres as a fallback.
    """
    genres = []

    if seed_ids:
        try:
            output = recommender.recommend(
                seed_ids,
                top_n=20,
                diversity=diversity,
            )

            if isinstance(output, pd.DataFrame):
                rows = output.to_dict("records")
            elif isinstance(output, list):
                rows = output
            else:
                rows = []

            for row in rows:
                if not isinstance(row, dict):
                    continue

                genre = (
                    row.get("genre")
                    or row.get("label")
                    or row.get("recommended_genre")
                )

                if genre:
                    genres.append(str(genre))

        except Exception:
            # The selected tracks' labels still provide a useful fallback.
            pass

        selected = tracks[tracks["track_id"].isin(seed_ids)]

        for genre in selected["label"].dropna().tolist():
            genres.append(str(genre))

    # Deduplicate without changing the order.
    return list(dict.fromkeys(
        genre.strip() for genre in genres if genre.strip()
    ))


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

DEFAULTS = {
    "catalog_pool": [],
    "catalog_results": [],
    "catalog_offset": 0,
    "catalog_message": "",
    "catalog_search_results": [],
    "seed_tracks": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    div[data-testid="stAudio"] {
        margin-top: 0;
        margin-bottom: 0;
    }

    .song-card {
        padding: 5px 0;
        border-bottom: 1px solid rgba(190,150,255,0.18);
    }
    .stApp {
        background: linear-gradient(135deg, #100c20 0%, #1b1230 100%);
        color: #f5f2ff;
    }

    h1, h2, h3 {
        color: #f6efff;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(190,150,255,0.2);
        padding: 14px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
    }

    .song-card {
        padding: 12px 0;
        border-bottom: 1px solid rgba(190,150,255,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------

st.title("🎧 Shazam-on-Steroids")
st.caption(
    "Discover music with machine learning, genre guidance, "
    "real catalog search, and feedback."
)

feedback = load_catalog_feedback()
tracks = load_gtzan_tracks()

with st.sidebar:
    st.header("Discovery settings")

    diversity_percent = st.slider(
        "Artist diversity",
        min_value=0,
        max_value=100,
        value=75,
        help="Higher values penalize artists already represented in a batch.",
    )

    st.divider()
    st.subheader("Your feedback")

    st.metric("Liked songs", len(feedback["likes"]))
    st.metric("Disliked songs", len(feedback["dislikes"]))
    st.metric("Skipped songs", len(feedback["skips"]))

    if st.button("Clear current recommendations", use_container_width=True):
        st.session_state.catalog_pool = []
        st.session_state.catalog_results = []
        st.session_state.catalog_offset = 0
        st.session_state.catalog_message = ""
        st.rerun()


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("GTZAN tracks", len(tracks))

with m2:
    st.metric("Saved likes", len(feedback["likes"]))

with m3:
    st.metric("Current batch", len(st.session_state.catalog_results))


# ---------------------------------------------------------
# Feedback and recommendation evaluation dashboard
# ---------------------------------------------------------

with st.expander("📊 Feedback & evaluation dashboard"):
    current_feedback = load_catalog_feedback()
    liked_items = list(current_feedback.get("likes", {}).values())
    disliked_items = list(current_feedback.get("disliked_songs", {}).values())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Likes", len(current_feedback.get("likes", {})))
    d2.metric("Dislikes", len(current_feedback.get("dislikes", [])))
    d3.metric("Skips", len(current_feedback.get("skips", [])))
    d4.metric(
        "Like / dislike ratio",
        f"{len(liked_items) / max(1, len(disliked_items)):.2f}",
    )

    genre_counts = pd.Series(
        [str(song.get("genre", "Unknown")) for song in liked_items]
    ).value_counts()
    if not genre_counts.empty:
        st.markdown("**Genres in your liked-song library**")
        st.bar_chart(genre_counts.head(10))
    else:
        st.caption("Like a few songs to see your genre-preference profile.")

    st.caption(
        "These are feedback and preference indicators, not an offline model "
        "accuracy score. Ground-truth relevance labels are not available yet."
    )


# ---------------------------------------------------------
# Real-song search
# ---------------------------------------------------------

st.header("🔎 Search real songs")

with st.form("real_music_search_form"):
    search_query = st.text_input(
        "Song, artist, album, or genre",
        placeholder="e.g. Arijit Singh, jazz, indie rock",
    )

    search_submitted = st.form_submit_button(
        "Search music",
        use_container_width=True,
    )

if search_submitted:
    if not search_query.strip():
        st.warning("Enter a song, artist, album, or genre first.")
    else:
        try:
            with st.spinner("Searching the music catalog..."):
                st.session_state.catalog_search_results = search_music(
                    search_query,
                    limit=50,
                )
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.catalog_search_results = []

search_results = st.session_state.catalog_search_results

if search_results:
    st.subheader(f"Search results ({len(search_results)})")

    for index, song in enumerate(search_results):
        with st.container():
            left, right = st.columns([1, 4])

            with left:
                if song.get("artwork"):
                    st.image(song["artwork"], width=90)
                else:
                    st.write("🎵")

            with right:
                st.markdown(f"**{song['title']}**")
                st.write(f"Artist: {song['artist']}")
                st.caption(
                    f"Album: {song['album']} · Genre: {song['genre']}"
                )

                if song.get("preview"):
                    st.audio(song["preview"], format="audio/mp4")

                if song.get("store_url"):
                    st.markdown(
                        f"[Open song in Apple Music/iTunes]({song['store_url']})"
                    )

                c1, c2, c3 = st.columns(3)

                with c1:
                    if st.button("👍 Like", key=f"search_like_{song['id']}"):
                        record_feedback(song, "like")
                        st.success("Like saved.")
                        st.rerun()

                with c2:
                    if st.button("👎 Dislike", key=f"search_dislike_{song['id']}"):
                        record_feedback(song, "dislike")
                        st.success("Dislike saved.")
                        st.rerun()

                with c3:
                    if st.button("⏭ Skip", key=f"search_skip_{song['id']}"):
                        record_feedback(song, "skip")
                        st.success("Song skipped.")
                        st.rerun()

# ---------------------------------------------------------
# Liked songs library
# ---------------------------------------------------------

st.divider()
st.header("💜 My Liked Songs")

liked_songs = list(load_catalog_feedback()["likes"].values())

if not liked_songs:
    st.info(
        "Your library is empty. Use the Like button on any song "
        "to save it here."
    )
else:
    st.caption(f"You have {len(liked_songs)} liked song(s).")

    with st.container(height=420, border=True):
        for song in liked_songs:
            with st.container():
                left, right = st.columns([1, 4])

                with left:
                    if song.get("artwork"):
                        st.image(
                            song["artwork"],
                            use_container_width=True,
                        )
                    else:
                        st.write("🎵")

                with right:
                    st.markdown(f"**{song.get('title', 'Unknown title')}**")
                    st.write(song.get("artist", "Unknown artist"))
                    st.caption(
                        f"Album: {song.get('album', 'Unknown album')} · "
                        f"Genre: {song.get('genre', 'Unknown genre')}"
                    )

                    if song.get("preview"):
                        st.audio(
                            song["preview"],
                            format="audio/mp4",
                        )

                    if song.get("store_url"):
                        st.markdown(
                            f"[Listen to song]({song['store_url']})"
                        )

                    if st.button(
                        "Remove from liked songs",
                        key=f"remove_like_{song.get('id', song.get('title'))}",
                    ):
                        feedback = load_catalog_feedback()
                        feedback["likes"].pop(
                            str(song.get("id", "")),
                            None,
                        )
                        save_catalog_feedback(feedback)
                        st.rerun()

                st.divider()


# ---------------------------------------------------------
# GTZAN seed selection
# ---------------------------------------------------------

st.divider()
st.header("🎼 Choose tracks to guide your recommendations")

seed_ids = []

if tracks.empty:
    st.error(
        "Could not load the GTZAN CSV. Make sure "
        "features_30_sec.csv or features_3_sec.csv is in the project folder."
    )
else:
    query = st.text_input(
        "Filter GTZAN tracks",
        placeholder="Search a filename or genre",
        key="gtzan_filter",
    )

    filtered = tracks.copy()

    if query.strip():
        mask = (
            filtered["filename"].astype(str).str.contains(
                query, case=False, na=False
            )
            | filtered["label"].astype(str).str.contains(
                query, case=False, na=False
            )
        )
        filtered = filtered[mask]

    options = {
        f"{row['filename']} — {row['label']}": row["track_id"]
        for _, row in filtered.head(300).iterrows()
    }

    selected_labels = st.multiselect(
        "Select one or more GTZAN tracks",
        options=list(options.keys()),
        default=[
            label for label, track_id in options.items()
            if track_id in st.session_state.seed_tracks
        ],
        help="Choose tracks whose genres you want to explore.",
    )

    seed_ids = [options[label] for label in selected_labels]
    st.session_state.seed_tracks = seed_ids

    if seed_ids:
        st.caption(f"{len(seed_ids)} seed track(s) selected.")


# ---------------------------------------------------------
# Generate recommendation pool
# ---------------------------------------------------------

st.header("✨ Recommendations")

feedback = load_catalog_feedback()

if st.button("Find real songs for me", type="primary", use_container_width=True):
    liked_genres = [
        item.get("genre", "")
        for item in feedback["likes"].values()
        if item.get("genre")
    ]

    genres = []

    if seed_ids:
        try:
            recommender = load_recommender()
            genres.extend(
                get_recommendation_genres(
                    recommender,
                    seed_ids,
                    tracks,
                    diversity_percent / 100.0,
                )
            )
        except Exception as exc:
            st.warning(
                "The recommendation engine could not be loaded. "
                "Using the selected GTZAN genres instead."
            )

        selected = tracks[tracks["track_id"].isin(seed_ids)]
        genres.extend(selected["label"].dropna().astype(str).tolist())

    # Liked real songs guide discovery even when no GTZAN tracks are selected.
    genres = liked_genres + genres
    genres = list(dict.fromkeys(
        genre.strip() for genre in genres if genre.strip()
    ))

    if not genres:
        st.warning(
            "Select at least one GTZAN track or like a real song first "
            "so SOS has a genre to explore."
        )
    else:
        try:
            with st.spinner("Building a larger recommendation pool..."):
                results = search_recommendations(
                    genres,
                    limit_per_genre=50,
                    diversity=diversity_percent / 100.0,
                )

            # Re-rank using saved likes/dislikes, then exclude blocked IDs.
            feedback = load_catalog_feedback()
            results = personalize_catalog_results(
                results,
                feedback,
                diversity=diversity_percent / 100.0,
            )
            blocked_ids = {
                str(item) for item in (
                    feedback["dislikes"] + feedback["skips"]
                )
            }

            results = [
                song for song in results
                if str(song.get("id")) not in blocked_ids
            ]

            # Keep the whole pool; display only the first batch.
            st.session_state.catalog_pool = results
            st.session_state.catalog_offset = 0
            st.session_state.catalog_results = results[:BATCH_SIZE]

            if results:
                st.session_state.catalog_message = (
                    f"Found {len(results)} eligible songs across "
                    f"{len(genres)} genre/search term(s)."
                )
            else:
                st.session_state.catalog_message = (
                    "No new songs were found. Try liking another song, "
                    "selecting a different GTZAN genre, or using Search music."
                )

        except Exception as exc:
            st.error(f"Could not build recommendations: {exc}")

        st.rerun()



# ---------------------------------------------------------
# Display current recommendation batch
# ---------------------------------------------------------

if st.session_state.catalog_message:
    st.info(st.session_state.catalog_message)

pool = st.session_state.catalog_pool
visible_songs = st.session_state.catalog_results

if pool:
    start = st.session_state.catalog_offset + 1
    end = st.session_state.catalog_offset + len(visible_songs)

    st.caption(
        f"Showing {start}–{end} of {len(pool)} songs "
        "in the current recommendation pool."
    )

if visible_songs:
    # Only this panel scrolls; the controls below remain outside it.
    with st.container(height=520, border=True):
        for song in visible_songs:
            with st.container():
                left, right = st.columns([1, 6], gap="small")                                      

                with left:
                    if song.get("artwork"):
                        st.image(
                            song["artwork"],
                            use_container_width=True,
                        )
                    else:
                        st.write("🎵")

                with right:
                    st.markdown(f"**{song['title']}**")
                    st.write(f"Artist: {song['artist']}")
                    st.caption(
                        f"Album: {song.get('album', 'Unknown album')} · "
                        f"Genre: {song.get('genre', 'Unknown genre')}"
                    )

                    st.caption(
                        f"Discovery term: "
                        f"{song.get('recommended_genre', 'N/A')} · "
                        f"Personalized score: "
                        f"{song.get('personalized_score', song.get('discovery_score', 0)):.2f}"
                    )

                    if song.get("preview"):
                        st.audio(
                            song["preview"],
                            format="audio/mp4",
                        )
                    else:
                        st.caption("No preview available.")

                    if song.get("store_url"):
                        st.markdown(
                            f"[Listen or view in Apple Music/iTunes]"
                            f"({song['store_url']})"
                        )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        if st.button(
                            "👍 Like",
                            key=f"rec_like_{song['id']}",
                        ):
                            record_feedback(song, "like")
                            st.rerun()

                    with c2:
                        if st.button(
                            "👎 Dislike",
                            key=f"rec_dislike_{song['id']}",
                        ):
                            record_feedback(song, "dislike")
                            st.rerun()

                    with c3:
                        if st.button(
                            "⏭ Skip",
                            key=f"rec_skip_{song['id']}",
                        ):
                            record_feedback(song, "skip")
                            st.rerun()

                st.divider()

    # Keep pagination outside the scrolling panel.
    next_offset = st.session_state.catalog_offset + BATCH_SIZE

    if next_offset < len(pool):
        if st.button(
            "Find another batch →",
            key="find_another_batch",
            use_container_width=True,
        ):
            st.session_state.catalog_offset = next_offset
            st.session_state.catalog_results = pool[
                next_offset:next_offset + BATCH_SIZE
            ]
            st.rerun()
    else:
        st.caption(
            "End of this recommendation pool. "
            "Select more genres to generate another pool."
        )

elif st.session_state.catalog_message:
    st.warning(
        "No songs to display. Try different genres or like another song."
    )

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------

with st.expander("How SOS recommendations work"):
    st.write(
        """
        1. GTZAN tracks provide genre guidance through the existing
           recommendation engine and dataset labels.
        2. SOS searches a real music catalog for tracks matching those
           genre/search terms.
        3. Artist diversity helps reduce repeated artists in the pool.
        4. Likes are saved locally and their genres influence future searches.
        5. Disliked and skipped catalog songs are excluded from future pools.

        Important: real catalog songs are currently found using genre-guided
        searches. Their audio is not yet extracted and compared directly
        against the GTZAN audio features.
        """
    )
