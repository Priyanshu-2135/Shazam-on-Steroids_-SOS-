
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from recommendation_engine import TuneTrailRecommender
from music_catalog import search_music, search_recommendations
from audio_processor import (
    load_audio_signal,
    extract_gtzan_features,
    predict_audio_genre,
    compute_audio_profile,
    explain_match,
)
from spotify_integration import (
    get_spotify_client,
    rip_spotify_tracks,
    build_playlist_taste_profile,
    recommend_from_spotify,
)


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
# Load Local Environment (.env - Git-Ignored)
# ---------------------------------------------------------

def load_local_env():
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            with env_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_local_env()


# ---------------------------------------------------------
# Feedback storage & Normalization
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

        if not isinstance(data["likes"], dict):
            data["likes"] = {}
        if not isinstance(data["dislikes"], list):
            data["dislikes"] = []
        if not isinstance(data["disliked_songs"], dict):
            data["disliked_songs"] = {}
        if not isinstance(data["skips"], list):
            data["skips"] = []

        # Normalize likes so every item has an 'id' attribute matching its key
        for song_id, song_data in list(data["likes"].items()):
            if isinstance(song_data, dict):
                song_data.setdefault("id", str(song_id))
            else:
                data["likes"][str(song_id)] = {
                    "id": str(song_id),
                    "title": str(song_data),
                    "artist": "Unknown artist",
                    "genre": "Unknown genre",
                }

        return data

    except (OSError, json.JSONDecodeError):
        return empty_feedback()


def save_catalog_feedback(feedback):
    with FEEDBACK_FILE.open("w", encoding="utf-8") as file:
        json.dump(feedback, file, indent=2, ensure_ascii=False)


def record_feedback(song, action):
    if not isinstance(song, dict):
        return

    song_id = str(song.get("id") or "")
    if not song_id:
        return

    feedback = load_catalog_feedback()

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
            "youtube_url": song.get("youtube_url", ""),
            "spotify_url": song.get("spotify_url", ""),
        }
        feedback["dislikes"] = [item for item in feedback["dislikes"] if str(item) != song_id]
        feedback["disliked_songs"].pop(song_id, None)
        feedback["skips"] = [item for item in feedback["skips"] if str(item) != song_id]

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
        feedback["skips"] = [item for item in feedback["skips"] if str(item) != song_id]

    elif action == "skip":
        if song_id not in [str(x) for x in feedback["skips"]]:
            feedback["skips"].append(song_id)

    save_catalog_feedback(feedback)

    # Remove song from visible pool in session state
    pool = st.session_state.get("catalog_pool", [])
    st.session_state.catalog_pool = [item for item in pool if str(item.get("id")) != song_id]

    visible = st.session_state.get("catalog_results", [])
    st.session_state.catalog_results = [item for item in visible if str(item.get("id")) != song_id]


# ---------------------------------------------------------
# Personalize real-catalog recommendations from user feedback
# ---------------------------------------------------------

def _normalise_text(value):
    return str(value or "").strip().casefold()


def personalize_catalog_results(results, feedback, diversity=0.75):
    """Re-rank catalog search results using saved likes/dislikes and artist diversity."""
    if not results:
        return []

    likes = [x for x in feedback.get("likes", {}).values() if isinstance(x, dict)]
    disliked_songs = [x for x in feedback.get("disliked_songs", {}).values() if isinstance(x, dict)]
    disliked_ids = {str(value) for value in feedback.get("dislikes", [])}
    skipped_ids = {str(value) for value in feedback.get("skips", [])}

    liked_genres = [_normalise_text(x.get("genre")) for x in likes]
    liked_artists = [_normalise_text(x.get("artist")) for x in likes]
    disliked_genres = [_normalise_text(x.get("genre")) for x in disliked_songs]
    disliked_artists = [_normalise_text(x.get("artist")) for x in disliked_songs]

    liked_genres = [x for x in liked_genres if x and x != "unknown genre"]
    liked_artists = [x for x in liked_artists if x and x != "unknown artist"]
    disliked_genres = [x for x in disliked_genres if x and x != "unknown genre"]
    disliked_artists = [x for x in disliked_artists if x and x != "unknown artist"]

    from collections import Counter
    liked_genre_counts = Counter(liked_genres)
    liked_artist_counts = Counter(liked_artists)
    disliked_genre_counts = Counter(disliked_genres)
    disliked_artist_counts = Counter(disliked_artists)

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

    ranked = []
    chosen_artists = set()

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
# Load GTZAN dataset & Recommender
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

    df = df.drop_duplicates(subset=["filename"]).copy()
    df["track_id"] = df["filename"].astype(str).apply(lambda name: Path(name).stem)
    return df.reset_index(drop=True)


@st.cache_resource
def load_recommender():
    return TuneTrailRecommender()


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

DEFAULTS = {
    "catalog_pool": [],
    "catalog_results": [],
    "catalog_offset": 0,
    "catalog_message": "",
    "catalog_search_results": [],
    "seed_tracks": [],
    "audio_analysis": None,
    "audio_features_df": None,
    "audio_raw_dict": None,
    "audio_rec_pool": [],
    "audio_record_nonce": 0,
    "spotify_client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
    "spotify_client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
    "spotify_playlist_meta": None,
    "spotify_ripped_tracks": [],
    "spotify_taste_profile": None,
    "spotify_recommendations": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0a1a 0%, #171026 100%);
        color: #f5f2ff;
    }
    h1, h2, h3 {
        color: #f7f1ff;
        font-weight: 700;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(180, 140, 255, 0.2);
        padding: 12px;
        border-radius: 10px;
    }
    .badge {
        display: inline-block;
        background: rgba(147, 51, 234, 0.25);
        border: 1px solid rgba(192, 132, 252, 0.45);
        color: #e9d5ff;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-top: 4px;
        font-weight: 500;
    }
    .badge-accent {
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.45);
        color: #bfdbfe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

feedback = load_catalog_feedback()
tracks = load_gtzan_tracks()

with st.sidebar:
    st.header("🎛️ Discovery Controls")

    diversity_percent = st.slider(
        "Artist Diversity",
        min_value=0,
        max_value=100,
        value=75,
        help="Higher values penalize artists already recommended in the pool.",
    )

    st.divider()
    st.subheader("📊 Your Taste Profile")
    st.metric("Liked Songs", len(feedback["likes"]))
    st.metric("Disliked Songs", len(feedback["dislikes"]))
    st.metric("Skipped Songs", len(feedback["skips"]))

    if st.button("Reset Recommendation Pool", use_container_width=True):
        st.session_state.catalog_pool = []
        st.session_state.catalog_results = []
        st.session_state.catalog_offset = 0
        st.session_state.catalog_message = ""
        st.session_state.audio_rec_pool = []
        st.rerun()

    st.caption("SOS | Shazam-on-Steroids v2.0")


# ---------------------------------------------------------
# Main Header & Metrics
# ---------------------------------------------------------

st.title("🎧 Shazam-on-Steroids")
st.caption("AI-powered audio fingerprinting, acoustic feature extraction, and music discovery engine.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("GTZAN Acoustic Tracks", len(tracks))
c2.metric("Saved Favorites", len(feedback["likes"]))
c3.metric("Discovery Pool", len(st.session_state.catalog_pool))
c4.metric("Active Seeds", len(st.session_state.seed_tracks))


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------

tab_shazam, tab_spotify, tab_seeds, tab_search, tab_library = st.tabs([
    "🎙️ Shazam Audio Fingerprint",
    "🟢 Spotify Playlist Importer",
    "🎼 GTZAN Guided Explorer",
    "🔎 Search Real Catalog",
    "💜 My Library & Insights",
])


# =========================================================
# TAB 1: SHAZAM AUDIO FINGERPRINTING & GENRE RECOGNITION
# =========================================================

with tab_shazam:
    st.subheader("Real-Time Audio Fingerprinting & AI Genre Classification")
    st.write(
        "Record a song snippet using your microphone or upload an audio file (MP3, WAV, M4A, OGG). "
        "SOS extracts 57 acoustic DSP features in real-time and identifies its genre and musical DNA."
    )

    nonce = st.session_state.audio_record_nonce

    col_input, col_reset = st.columns([3, 1])
    with col_input:
        input_mode = st.radio(
            "Select Audio Input Method:",
            ["🎙️ Record from Microphone", "📁 Upload Audio File"],
            horizontal=True,
            key=f"input_mode_radio_{nonce}",
        )
    with col_reset:
        st.write("")
        if st.button("🔄 Record Again / Reset", use_container_width=True, help="Clear recorded audio and reset input"):
            st.session_state.audio_record_nonce += 1
            st.session_state.audio_analysis = None
            st.session_state.audio_features_df = None
            st.session_state.audio_raw_dict = None
            st.session_state.audio_rec_pool = []
            st.rerun()

    audio_bytes = None

    if input_mode == "🎙️ Record from Microphone":
        audio_input = st.audio_input("Record audio clip (3–10 seconds)", key=f"audio_mic_{nonce}")
        if audio_input is not None:
            audio_bytes = audio_input.read()
    else:
        uploaded_file = st.file_uploader(
            "Upload song or audio snippet",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            key=f"audio_upload_{nonce}",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()

    if audio_bytes:
        st.audio(audio_bytes)

        btn_c1, btn_c2 = st.columns([2, 1])
        with btn_c1:
            analyze_clicked = st.button("🚀 Analyze Audio & Extract Fingerprint", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🗑️ Discard & Record Again", key=f"discard_rec_{nonce}", use_container_width=True):
                st.session_state.audio_record_nonce += 1
                st.session_state.audio_analysis = None
                st.session_state.audio_features_df = None
                st.session_state.audio_raw_dict = None
                st.session_state.audio_rec_pool = []
                st.rerun()

        if analyze_clicked:
            try:
                with st.spinner("Extracting 57 DSP features (MFCCs, Chroma, Spectral Centroid, Tempo)..."):
                    y, loaded_sr = load_audio_signal(audio_bytes, sr=22050, max_duration=30.0)
                    raw_feats, features_df = extract_gtzan_features(y, sr=loaded_sr)
                    genre_pred = predict_audio_genre(features_df)
                    profile = compute_audio_profile(y, raw_feats)

                st.session_state.audio_analysis = {
                    "genre_pred": genre_pred,
                    "profile": profile,
                }
                st.session_state.audio_features_df = features_df
                st.session_state.audio_raw_dict = raw_feats
                st.session_state.audio_rec_pool = []
                st.success("Audio DNA successfully extracted!")

            except Exception as e:
                st.error(f"Could not analyze audio: {e}")

    # Display Analysis Results
    if st.session_state.audio_analysis:
        analysis = st.session_state.audio_analysis
        pred = analysis["genre_pred"]
        prof = analysis["profile"]

        st.divider()
        st.markdown(f"### 🎯 Predicted Genre: **{pred['predicted_genre'].upper()}**")
        st.write(f"Model Confidence: **{round(pred['confidence'] * 100, 1)}%**")

        col_dna1, col_dna2, col_dna3, col_dna4 = st.columns(4)
        col_dna1.metric("Estimated Tempo", f"{prof['bpm']} BPM")
        col_dna2.metric("Energy Level", prof['energy'].split('(')[0].strip())
        col_dna3.metric("Timbre Tone", prof['brightness'].split('(')[0].strip())
        col_dna4.metric("Rhythm Texture", prof['texture'])

        st.markdown("#### Genre Probability Distribution")
        probs_df = pd.DataFrame(
            list(pred["probabilities"].items()),
            columns=["Genre", "Probability"]
        ).set_index("Genre")
        st.bar_chart(probs_df)

        st.divider()
        st.subheader("🔍 Discover Matching Songs for this Audio")

        rec_c1, rec_c2 = st.columns([2, 1])
        with rec_c1:
            find_clicked = st.button("⚡ Find Real Catalog Songs Matching this Sound", use_container_width=True)
        with rec_c2:
            if st.button("🔄 Record Another Song", key="rec_another_btn", use_container_width=True):
                st.session_state.audio_record_nonce += 1
                st.session_state.audio_analysis = None
                st.session_state.audio_features_df = None
                st.session_state.audio_raw_dict = None
                st.session_state.audio_rec_pool = []
                st.rerun()

        if find_clicked:
            recommender = load_recommender()
            with st.spinner("Computing vector similarity and querying music catalog..."):
                vec_recs = recommender.recommend_from_vector(
                    st.session_state.audio_raw_dict,
                    top_n=10,
                    diversity=diversity_percent / 100.0
                )

                search_genres = [pred["predicted_genre"]]
                if vec_recs is not None and not vec_recs.empty:
                    top_rec_genres = vec_recs["genre"].dropna().tolist()
                    search_genres.extend(top_rec_genres[:3])
                search_genres = list(dict.fromkeys(search_genres))

                real_catalog_songs = search_recommendations(
                    search_genres,
                    limit_per_genre=30,
                    diversity=diversity_percent / 100.0
                )

                real_catalog_songs = personalize_catalog_results(
                    real_catalog_songs,
                    feedback,
                    diversity=diversity_percent / 100.0
                )

                st.session_state.audio_rec_pool = real_catalog_songs

        audio_pool = st.session_state.audio_rec_pool
        if audio_pool:
            st.markdown(f"**Found {len(audio_pool)} matching songs based on acoustic similarity:**")
            for idx, song in enumerate(audio_pool[:8]):
                s_id = str(song.get("id") or idx)
                with st.container():
                    left, right = st.columns([1, 5], gap="medium")
                    with left:
                        if song.get("artwork"):
                            st.image(song["artwork"], use_container_width=True)
                        else:
                            st.write("🎵")
                    with right:
                        st.markdown(f"**{song['title']}** — {song['artist']}")
                        st.caption(f"Album: {song.get('album', 'N/A')} · Genre: {song.get('genre', 'N/A')}")

                        badges_html = f'<span class="badge badge-accent">🎯 {pred["predicted_genre"]} match</span>'
                        badges_html += f'<span class="badge">🥁 {prof["bpm"]} BPM Vibe</span>'
                        st.markdown(badges_html, unsafe_allow_html=True)

                        if song.get("preview"):
                            st.audio(song["preview"], format="audio/mp4")
                        else:
                            st.caption("No 30s preview available from iTunes.")

                        link_cols = st.columns([1, 1, 1])
                        with link_cols[0]:
                            if song.get("store_url"):
                                st.markdown(f"[🍎 Apple Music]({song['store_url']})")
                        with link_cols[1]:
                            if song.get("youtube_url"):
                                st.markdown(f"[▶️ YouTube Search]({song['youtube_url']})")
                        with link_cols[2]:
                            if song.get("spotify_url"):
                                st.markdown(f"[🟢 Spotify Search]({song['spotify_url']})")

                        fb1, fb2, fb3 = st.columns(3)
                        with fb1:
                            if st.button("👍 Like", key=f"audio_like_{idx}_{s_id}"):
                                record_feedback(song, "like")
                                st.success("Liked!")
                                st.rerun()
                        with fb2:
                            if st.button("👎 Dislike", key=f"audio_dislike_{idx}_{s_id}"):
                                record_feedback(song, "dislike")
                                st.rerun()
                        with fb3:
                            if st.button("⏭ Skip", key=f"audio_skip_{idx}_{s_id}"):
                                record_feedback(song, "skip")
                                st.rerun()
                st.divider()


# =========================================================
# TAB 2: SPOTIFY PLAYLIST IMPORTER & RECOMMENDER
# =========================================================

with tab_spotify:
    st.subheader("🟢 Rip Spotify Playlist & Recommend New Songs")
    st.write(
        "Import any public or shared Spotify playlist or album. "
        "SOS rips song data, extracts artist genres, synthesizes a taste profile, "
        "and recommends fresh tracks matching the playlist's musical DNA (guaranteeing no duplicates from your playlist)."
    )

    # 1. Credentials Configuration (100% Free)
    has_creds = bool(st.session_state.get("spotify_client_id") and st.session_state.get("spotify_client_secret"))
    creds_title = "✅ Spotify API Configured (100% Free Developer Tier)" if has_creds else "🔑 Spotify Developer API Setup (100% Free / Zero Billing - Click to configure)"

    with st.expander(creds_title, expanded=not has_creds):
        st.markdown(
            """
            **How to get your free credentials in 1 minute (No credit card required, completely free forever):**
            1. Log in to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) with your standard Spotify account (Free or Premium).
            2. Click **Create app**. Enter an App Name (e.g. `Shazam-on-Steroids`) and App Description.
            3. In **Redirect URIs**, enter `http://localhost:8501/callback` and click **Save**.
            4. Go to **Settings** $\\rightarrow$ click **View client secret**.
            5. Copy your **Client ID** and **Client Secret** below:
            """
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            inp_cid = st.text_input("Spotify Client ID", value=st.session_state.get("spotify_client_id", ""), type="password", key="sp_cid_input")
        with col_c2:
            inp_csec = st.text_input("Spotify Client Secret", value=st.session_state.get("spotify_client_secret", ""), type="password", key="sp_csec_input")

        if st.button("Save Spotify Credentials", key="save_sp_creds_btn"):
            st.session_state.spotify_client_id = inp_cid.strip()
            st.session_state.spotify_client_secret = inp_csec.strip()
            # Persist to local .env file (safely ignored by git)
            env_file = BASE_DIR / ".env"
            try:
                env_file.write_text(
                    f"SPOTIFY_CLIENT_ID={inp_cid.strip()}\nSPOTIFY_CLIENT_SECRET={inp_csec.strip()}\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
            st.success("Credentials saved to local .env (git-ignored) and session!")
            st.rerun()

    # 2. Playlist Input Form
    spotify_url = st.text_input(
        "Enter Spotify Playlist or Album URL / URI:",
        placeholder="e.g. https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
        key="spotify_playlist_url_input",
    )

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        max_tracks_to_rip = st.slider("Max tracks to analyze", min_value=10, max_value=100, value=50, step=10, key="sp_max_tracks_slider")
    with col_opt2:
        st.write("")
        rip_btn = st.button("📥 Rip Playlist Data", type="primary", use_container_width=True, key="rip_sp_btn")

    if rip_btn:
        if not spotify_url.strip():
            st.warning("Please enter a valid Spotify playlist URL.")
        else:
            try:
                with st.spinner("Connecting to Spotify and ripping tracks..."):
                    sp = None
                    if st.session_state.get("spotify_client_id") and st.session_state.get("spotify_client_secret"):
                        try:
                            sp = get_spotify_client(
                                st.session_state.spotify_client_id,
                                st.session_state.spotify_client_secret
                            )
                        except Exception:
                            sp = None

                    meta, ripped = rip_spotify_tracks(sp, spotify_url.strip(), max_tracks=max_tracks_to_rip)
                    taste = build_playlist_taste_profile(ripped)

                st.session_state.spotify_playlist_meta = meta
                st.session_state.spotify_ripped_tracks = ripped
                st.session_state.spotify_taste_profile = taste
                st.session_state.spotify_recommendations = []
                st.success(f"Successfully ripped {len(ripped)} songs from '{meta['title']}'!")

            except Exception as e:
                st.error(f"Failed to rip Spotify playlist: {e}")

    # 3. Display Ripped Playlist Details
    if st.session_state.get("spotify_playlist_meta") and st.session_state.get("spotify_ripped_tracks"):
        meta = st.session_state.spotify_playlist_meta
        ripped = st.session_state.spotify_ripped_tracks
        taste = st.session_state.spotify_taste_profile

        st.divider()
        top_l, top_r = st.columns([1, 4])
        with top_l:
            if meta.get("artwork"):
                st.image(meta["artwork"], use_container_width=True)
            else:
                st.write("🎵")
        with top_r:
            st.markdown(f"### {meta.get('title', 'Spotify Playlist')}")
            if meta.get("owner"):
                st.write(f"**Curator / Artist:** {meta['owner']}")
            if meta.get("description"):
                st.caption(meta["description"])
            st.write(f"**Analyzed Tracks:** {len(ripped)} / {meta.get('total_tracks', len(ripped))}")

        # Taste Profile Metrics
        d1, d2, d3 = st.columns(3)
        d1.metric("Dominant Styles", len(taste.get("dominant_genres", [])))
        d2.metric("Unique Artists", len(taste.get("top_artists", [])))
        d3.metric("Avg Song Popularity", f"{taste.get('avg_popularity', 0)} / 100")

        # Genres Pills
        if taste.get("dominant_genres"):
            genres_html = "**Dominant Musical Styles:** " + " ".join([f'<span class="badge">{g}</span>' for g in taste["dominant_genres"][:6]])
            st.markdown(genres_html, unsafe_allow_html=True)

        with st.expander(f"📋 View Ripped Playlist Songs ({len(ripped)})"):
            rip_df = pd.DataFrame([
                {
                    "Title": t["title"],
                    "Artist": t["artist"],
                    "Album": t["album"],
                    "Popularity": t["popularity"],
                    "Genres": ", ".join(t.get("genres", [])[:2]) or "N/A"
                }
                for t in ripped
            ])
            st.dataframe(rip_df, use_container_width=True)

        st.divider()
        st.subheader("✨ Recommendations for this Playlist")
        st.caption("SOS searches for new songs matching this playlist's genre profile, guaranteeing no duplicate songs from your playlist.")

        if st.button("🚀 Generate New Recommendations", type="primary", use_container_width=True, key="gen_sp_recs_btn"):
            with st.spinner("Finding matching songs and applying taste reranking..."):
                try:
                    spotify_recs = recommend_from_spotify(
                        ripped,
                        search_recommendations_fn=search_recommendations,
                        personalize_fn=personalize_catalog_results,
                        feedback=feedback,
                        top_n=15,
                        diversity=diversity_percent / 100.0
                    )
                    st.session_state.spotify_recommendations = spotify_recs
                    if not spotify_recs:
                        st.info("No new unique songs found. Try adjusting the diversity slider or checking other playlists.")
                except Exception as exc:
                    st.error(f"Could not build recommendations: {exc}")

        # Render Spotify Recommendations
        sp_recs = st.session_state.get("spotify_recommendations", [])
        if sp_recs:
            st.markdown(f"**Discovered {len(sp_recs)} new songs matching this playlist's vibe:**")
            for idx, song in enumerate(sp_recs):
                s_id = str(song.get("id") or idx)
                with st.container():
                    left, right = st.columns([1, 5], gap="medium")
                    with left:
                        if song.get("artwork"):
                            st.image(song["artwork"], use_container_width=True)
                        else:
                            st.write("🎵")
                    with right:
                        st.markdown(f"**{song['title']}** — {song['artist']}")
                        st.caption(f"Album: {song.get('album', 'N/A')} · Genre: {song.get('genre', 'N/A')}")

                        badges_html = f'<span class="badge badge-accent">🎯 Playlist Vibe Match</span>'
                        if song.get("recommended_genre"):
                            badges_html += f'<span class="badge">🎸 {song["recommended_genre"]}</span>'
                        st.markdown(badges_html, unsafe_allow_html=True)

                        if song.get("preview"):
                            st.audio(song["preview"], format="audio/mp4")
                        else:
                            st.caption("No 30s preview available from iTunes.")

                        lcols = st.columns(3)
                        with lcols[0]:
                            if song.get("store_url"):
                                st.markdown(f"[🍎 Apple Music]({song['store_url']})")
                        with lcols[1]:
                            if song.get("youtube_url"):
                                st.markdown(f"[▶️ YouTube Search]({song['youtube_url']})")
                        with lcols[2]:
                            if song.get("spotify_url"):
                                st.markdown(f"[🟢 Spotify Search]({song['spotify_url']})")

                        fb1, fb2, fb3 = st.columns(3)
                        with fb1:
                            if st.button("👍 Like", key=f"sp_like_{idx}_{s_id}"):
                                record_feedback(song, "like")
                                st.success("Added to liked library!")
                                st.rerun()
                        with fb2:
                            if st.button("👎 Dislike", key=f"sp_dislike_{idx}_{s_id}"):
                                record_feedback(song, "dislike")
                                st.rerun()
                        with fb3:
                            if st.button("⏭ Skip", key=f"sp_skip_{idx}_{s_id}"):
                                record_feedback(song, "skip")
                                st.rerun()
                st.divider()


# =========================================================
# TAB 3: GTZAN GUIDED EXPLORER
# =========================================================

with tab_seeds:
    st.subheader("Explore Musical Directions via GTZAN Seeds")

    if tracks.empty:
        st.error("GTZAN dataset not found. Ensure features_30_sec.csv exists.")
    else:
        query_filter = st.text_input(
            "Filter GTZAN catalog tracks",
            placeholder="Search filename or genre (e.g., rock, blues.00000)",
            key="gtzan_filter_box",
        )

        filtered = tracks.copy()
        if query_filter.strip():
            mask = (
                filtered["filename"].astype(str).str.contains(query_filter, case=False, na=False)
                | filtered["label"].astype(str).str.contains(query_filter, case=False, na=False)
            )
            filtered = filtered[mask]

        options = {
            f"{row['filename']} — {row['label']}": row["track_id"]
            for _, row in filtered.head(200).iterrows()
        }

        selected_labels = st.multiselect(
            "Select seed tracks to guide recommendations:",
            options=list(options.keys()),
            default=[
                label for label, track_id in options.items()
                if track_id in st.session_state.seed_tracks
            ],
        )

        seed_ids = [options[label] for label in selected_labels]
        st.session_state.seed_tracks = seed_ids

        col_seed_act1, col_seed_act2 = st.columns(2)

        with col_seed_act1:
            if st.button("✨ Discover Real Songs from Seeds", type="primary", use_container_width=True):
                recommender = load_recommender()
                genres = []
                if seed_ids:
                    try:
                        rec_df = recommender.recommend(seed_ids, top_n=20, diversity=diversity_percent / 100.0)
                        genres.extend(rec_df["genre"].dropna().tolist())
                    except Exception:
                        pass
                    selected = tracks[tracks["track_id"].isin(seed_ids)]
                    genres.extend(selected["label"].dropna().astype(str).tolist())

                liked_genres = [
                    item.get("genre", "") for item in feedback["likes"].values()
                    if isinstance(item, dict) and item.get("genre")
                ]
                genres = list(dict.fromkeys([g.strip() for g in (liked_genres + genres) if g.strip()]))

                if not genres:
                    st.warning("Please choose at least one seed or like a song first.")
                else:
                    with st.spinner("Finding songs across real catalog..."):
                        results = search_recommendations(
                            genres,
                            limit_per_genre=40,
                            diversity=diversity_percent / 100.0
                        )
                        results = personalize_catalog_results(
                            results,
                            feedback,
                            diversity=diversity_percent / 100.0
                        )
                        blocked = {str(item) for item in (feedback["dislikes"] + feedback["skips"])}
                        results = [s for s in results if str(s.get("id")) not in blocked]

                        st.session_state.catalog_pool = results
                        st.session_state.catalog_offset = 0
                        st.session_state.catalog_results = results[:BATCH_SIZE]
                        st.session_state.catalog_message = f"Generated {len(results)} songs across {len(genres)} genre(s)."
                    st.rerun()

        with col_seed_act2:
            if st.button("🔍 View Acoustic Matches in GTZAN", use_container_width=True):
                if not seed_ids:
                    st.warning("Select seed tracks first.")
                else:
                    recommender = load_recommender()
                    top_gtzan = recommender.recommend(seed_ids, top_n=10, diversity=diversity_percent / 100.0)
                    st.dataframe(top_gtzan)

    if st.session_state.catalog_message:
        st.info(st.session_state.catalog_message)

    visible_songs = st.session_state.catalog_results
    pool = st.session_state.catalog_pool

    if visible_songs:
        st.caption(f"Showing batch {st.session_state.catalog_offset + 1}–{st.session_state.catalog_offset + len(visible_songs)} of {len(pool)} songs.")
        with st.container(height=520, border=True):
            for idx, song in enumerate(visible_songs):
                s_id = str(song.get("id") or idx)
                with st.container():
                    left, right = st.columns([1, 6], gap="small")
                    with left:
                        if song.get("artwork"):
                            st.image(song["artwork"], use_container_width=True)
                        else:
                            st.write("🎵")
                    with right:
                        st.markdown(f"**{song['title']}**")
                        st.write(f"Artist: {song['artist']}")
                        st.caption(f"Album: {song.get('album', 'Unknown')} · Genre: {song.get('genre', 'Unknown')}")
                        st.caption(f"Search Term: {song.get('recommended_genre', 'N/A')} · Score: {song.get('personalized_score', song.get('discovery_score', 0)):.2f}")

                        if song.get("preview"):
                            st.audio(song["preview"], format="audio/mp4")
                        else:
                            st.caption("No preview clip available.")

                        l1, l2, l3 = st.columns(3)
                        with l1:
                            if song.get("store_url"):
                                st.markdown(f"[Apple Music]({song['store_url']})")
                        with l2:
                            if song.get("youtube_url"):
                                st.markdown(f"[YouTube Search]({song['youtube_url']})")
                        with l3:
                            if song.get("spotify_url"):
                                st.markdown(f"[Spotify Search]({song['spotify_url']})")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("👍 Like", key=f"cat_like_{idx}_{s_id}"):
                                record_feedback(song, "like")
                                st.rerun()
                        with c2:
                            if st.button("👎 Dislike", key=f"cat_dislike_{idx}_{s_id}"):
                                record_feedback(song, "dislike")
                                st.rerun()
                        with c3:
                            if st.button("⏭ Skip", key=f"cat_skip_{idx}_{s_id}"):
                                record_feedback(song, "skip")
                                st.rerun()
                st.divider()

        next_offset = st.session_state.catalog_offset + BATCH_SIZE
        if next_offset < len(pool):
            if st.button("Find another batch →", key="seed_batch_next", use_container_width=True):
                st.session_state.catalog_offset = next_offset
                st.session_state.catalog_results = pool[next_offset:next_offset + BATCH_SIZE]
                st.rerun()


# =========================================================
# TAB 3: SEARCH REAL CATALOG
# =========================================================

with tab_search:
    st.subheader("Search Apple iTunes Music Catalog")

    with st.form("real_catalog_search_form"):
        search_query = st.text_input(
            "Song, artist, or genre",
            placeholder="e.g. Coldplay, synthwave, Arijit Singh, Daft Punk"
        )
        submitted = st.form_submit_button("Search Catalog", use_container_width=True)

    if submitted:
        if not search_query.strip():
            st.warning("Enter a search term first.")
        else:
            try:
                with st.spinner("Searching real songs..."):
                    results = search_music(search_query, limit=50)
                    st.session_state.catalog_search_results = results
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state.catalog_search_results = []

    search_items = st.session_state.catalog_search_results
    if search_items:
        st.markdown(f"**Found {len(search_items)} songs:**")
        for idx, song in enumerate(search_items):
            s_id = str(song.get("id") or idx)
            with st.container():
                left, right = st.columns([1, 5])
                with left:
                    if song.get("artwork"):
                        st.image(song["artwork"], width=90)
                    else:
                        st.write("🎵")
                with right:
                    st.markdown(f"**{song['title']}** — {song['artist']}")
                    st.caption(f"Album: {song.get('album', 'N/A')} · Genre: {song.get('genre', 'N/A')}")

                    if song.get("preview"):
                        st.audio(song["preview"], format="audio/mp4")
                    else:
                        st.caption("No 30s preview available.")

                    link_col1, link_col2 = st.columns(2)
                    with link_col1:
                        if song.get("store_url"):
                            st.markdown(f"[Open in Apple Music]({song['store_url']})")
                    with link_col2:
                        if song.get("youtube_url"):
                            st.markdown(f"[Search on YouTube]({song['youtube_url']})")

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        if st.button("👍 Like", key=f"s_like_{idx}_{s_id}"):
                            record_feedback(song, "like")
                            st.success("Added to library!")
                            st.rerun()
                    with sc2:
                        if st.button("👎 Dislike", key=f"s_dislike_{idx}_{s_id}"):
                            record_feedback(song, "dislike")
                            st.rerun()
                    with sc3:
                        if st.button("⏭ Skip", key=f"s_skip_{idx}_{s_id}"):
                            record_feedback(song, "skip")
                            st.rerun()
            st.divider()


# =========================================================
# TAB 4: MY LIBRARY & PREFERENCE INSIGHTS
# =========================================================

with tab_library:
    st.subheader("💜 Your Saved Favorites")
    liked_songs = list(load_catalog_feedback()["likes"].values())

    if not liked_songs:
        st.info("Your liked songs library is empty. Use the 👍 Like button on any song to save it here.")
    else:
        st.caption(f"You have {len(liked_songs)} saved track(s).")
        with st.container(height=400, border=True):
            for idx, song in enumerate(liked_songs):
                s_id = str(song.get("id") or idx)
                with st.container():
                    left, right = st.columns([1, 5])
                    with left:
                        if song.get("artwork"):
                            st.image(song["artwork"], use_container_width=True)
                        else:
                            st.write("🎵")
                    with right:
                        st.markdown(f"**{song.get('title', 'Unknown')}** — {song.get('artist', 'Unknown')}")
                        st.caption(f"Genre: {song.get('genre', 'Unknown')} · Album: {song.get('album', 'Unknown')}")
                        if song.get("preview"):
                            st.audio(song["preview"], format="audio/mp4")

                        if st.button("Remove from Library", key=f"lib_rm_{idx}_{s_id}"):
                            fb = load_catalog_feedback()
                            fb["likes"].pop(s_id, None)
                            save_catalog_feedback(fb)
                            st.rerun()
                st.divider()

    st.divider()
    st.subheader("📈 Feedback & Genre Preference Dashboard")
    current_fb = load_catalog_feedback()
    likes_count = len(current_fb.get("likes", {}))
    dislikes_count = len(current_fb.get("dislikes", []))
    skips_count = len(current_fb.get("skips", []))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Likes", likes_count)
    d2.metric("Total Dislikes", dislikes_count)
    d3.metric("Total Skips", skips_count)
    d4.metric("Like/Dislike Ratio", f"{likes_count / max(1, dislikes_count):.2f}")

    liked_items = list(current_fb.get("likes", {}).values())
    genre_series = pd.Series([s.get("genre", "Unknown") for s in liked_items if isinstance(s, dict)]).value_counts()
    if not genre_series.empty:
        st.markdown("**Dominant Genres in Your Library:**")
        st.bar_chart(genre_series.head(8))
    else:
        st.caption("Like songs across different styles to see your genre preference distribution.")
