
import requests
import streamlit as st

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


@st.cache_data(ttl=1800, show_spinner=False)
def search_music(query, limit=50, country="IN"):
    """Search Apple's public iTunes catalog for real songs."""
    query = str(query).strip()

    if not query:
        return []

    params = {
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": min(max(int(limit), 1), 50),
        "country": country,
    }

    try:
        response = requests.get(
            ITUNES_SEARCH_URL,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach the music catalog. "
            "Check your internet connection and try again."
        ) from exc

    songs = []

    for item in payload.get("results", []):
        if item.get("kind") != "song":
            continue

        title = item.get("trackName")
        artist = item.get("artistName")
        track_id = item.get("trackId")

        if not title or not artist or not track_id:
            continue

        songs.append({
            "id": str(track_id),
            "title": title,
            "artist": artist,
            "album": item.get("collectionName", "Unknown album"),
            "genre": item.get("primaryGenreName", "Unknown genre"),
            "artwork": item.get("artworkUrl100", ""),
            "preview": item.get("previewUrl", ""),
            "store_url": item.get("trackViewUrl", ""),
        })

    return songs


def search_recommendations(genres, limit_per_genre=50, diversity=0.75):
    """
    Build a pool of real songs using genre/search terms.

    The diversity score favours artists that have appeared less often.
    This is genre-guided discovery, not direct audio-feature similarity.
    """
    results = []
    seen_track_ids = set()
    artist_counts = {}

    # Preserve genre order while removing duplicate search terms.
    search_terms = []
    seen_terms = set()

    for genre in genres:
        term = str(genre).strip()

        if not term:
            continue

        normalized = term.casefold()

        if normalized not in seen_terms:
            search_terms.append(term)
            seen_terms.add(normalized)

    for term in search_terms:
        try:
            songs = search_music(term, limit=limit_per_genre)
        except RuntimeError:
            # One failed search should not prevent other genres from loading.
            continue

        for song in songs:
            track_id = song.get("id")

            if not track_id or track_id in seen_track_ids:
                continue

            artist_key = song["artist"].casefold()
            repeat_count = artist_counts.get(artist_key, 0)

            # Penalize repeated artists while retaining their songs.
            score = 1.0 - float(diversity) * 0.35 * repeat_count

            song["discovery_score"] = max(0.0, score)
            song["recommended_genre"] = term

            results.append(song)
            seen_track_ids.add(track_id)
            artist_counts[artist_key] = repeat_count + 1

    results.sort(
        key=lambda song: song["discovery_score"],
        reverse=True,
    )

    return results
