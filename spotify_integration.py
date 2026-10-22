"""
Spotify Integration Module for Shazam-on-Steroids (SOS).
Handles Spotify playlist and track ripping, genre & artist extraction,
taste profile synthesis, and intelligent recommendation generation.
Supports both official Web API and 100% free public embed extraction
(no Spotify Premium required, zero billing).
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional, Set

import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


def get_spotify_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
) -> spotipy.Spotify:
    """
    Initialize a Spotify API client using Client Credentials flow.
    """
    cid = (client_id or os.getenv("SPOTIFY_CLIENT_ID", "")).strip()
    csecret = (client_secret or os.getenv("SPOTIFY_CLIENT_SECRET", "")).strip()

    if not cid or not csecret:
        raise ValueError(
            "Spotify Client ID and Client Secret are required. "
            "Please provide them in the UI or set SPOTIFY_CLIENT_ID "
            "and SPOTIFY_CLIENT_SECRET environment variables."
        )

    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=csecret)
    return spotipy.Spotify(auth_manager=auth_manager)


def parse_spotify_uri_or_url(input_str: str) -> Tuple[str, str]:
    """
    Extract the item type (playlist, album, track) and its Spotify ID from a URL or URI.
    """
    clean_str = input_str.strip()

    # Regex for open.spotify.com URLs
    url_match = re.search(r"open\.spotify\.com/(playlist|album|track)/([a-zA-Z0-9]+)", clean_str)
    if url_match:
        return url_match.group(1), url_match.group(2)

    # Regex for spotify:type:id URIs
    uri_match = re.search(r"spotify:(playlist|album|track):([a-zA-Z0-9]+)", clean_str)
    if uri_match:
        return uri_match.group(1), uri_match.group(2)

    # Default to playlist if only alphanumeric ID is provided
    if re.match(r"^[a-zA-Z0-9]{22}$", clean_str):
        return "playlist", clean_str

    raise ValueError(
        f"Invalid Spotify URL or ID: '{input_str}'. "
        "Please provide a valid Spotify playlist link (e.g. https://open.spotify.com/playlist/...)."
    )


def rip_spotify_public(url_or_id: str, max_tracks: int = 100) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Rips public Spotify playlist or album metadata and tracklist using
    Spotify's public web embed pipeline. 100% free, requires NO API keys
    and NO active Spotify Premium subscription.
    """
    item_type, item_id = parse_spotify_uri_or_url(url_or_id)
    embed_url = f"https://open.spotify.com/embed/{item_type}/{item_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(embed_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Spotify public embed: {exc}") from exc

    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text)
    if not match:
        raise ValueError("Could not parse Spotify playlist structure. Make sure the playlist is public or shared.")

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed Spotify playlist JSON.") from exc

    entity = (
        data.get("props", {})
        .get("pageProps", {})
        .get("state", {})
        .get("data", {})
        .get("entity", {})
    )

    cover = ""
    if entity.get("coverArt", {}).get("sources"):
        cover = entity["coverArt"]["sources"][0].get("url", "")

    meta = {
        "id": item_id,
        "type": item_type,
        "title": entity.get("title") or entity.get("name", "Spotify Playlist"),
        "description": entity.get("subtitle", ""),
        "owner": ", ".join([a.get("name", "") for a in entity.get("authors", [])]) if entity.get("authors") else entity.get("subtitle", "Spotify"),
        "artwork": cover,
        "total_tracks": len(entity.get("trackList", [])),
    }

    tracks = []
    track_list = entity.get("trackList", [])[:max_tracks]

    for t in track_list:
        t_uri = t.get("uri", "")
        t_id = t_uri.split(":")[-1] if ":" in t_uri else t.get("uid", "")
        artists_str = t.get("subtitle", "Unknown Artist").replace("\xa0", " ").strip()
        primary_artist = artists_str.split(",")[0].strip() if artists_str else "Unknown"

        # Direct MP3 audio preview from Spotify CDN
        preview = ""
        if t.get("audioPreview") and t["audioPreview"].get("url"):
            preview = t["audioPreview"]["url"]

        tracks.append({
            "id": t_id,
            "title": t.get("title", "Unknown Title"),
            "artist": artists_str,
            "primary_artist": primary_artist,
            "album": meta["title"],
            "artwork": cover,
            "preview_url": preview,
            "popularity": 70,
            "duration_ms": t.get("duration", 0),
            "genres": [],
            "spotify_url": f"https://open.spotify.com/track/{t_id}",
        })

    return meta, tracks


def rip_spotify_tracks(
    sp: Optional[spotipy.Spotify],
    url_or_id: str,
    max_tracks: int = 100
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Rips Spotify songs. Tries the official Spotify Web API first; if Spotify
    rejects with 403 (e.g. recent Spotify Premium requirement on dev accounts),
    it transparently falls back to the 100% free public embed ripper.
    """
    # If no Spotify client was provided, use public embed directly
    if sp is None:
        return rip_spotify_public(url_or_id, max_tracks=max_tracks)

    item_type, item_id = parse_spotify_uri_or_url(url_or_id)

    container_meta = {
        "id": item_id,
        "type": item_type,
        "title": "Unknown Collection",
        "description": "",
        "owner": "",
        "artwork": "",
        "total_tracks": 0,
    }

    raw_items = []

    try:
        if item_type == "playlist":
            playlist_data = sp.playlist(item_id, fields="name,description,owner.display_name,images,tracks.total")
            container_meta["title"] = playlist_data.get("name", "Spotify Playlist")
            container_meta["description"] = playlist_data.get("description", "")
            container_meta["owner"] = playlist_data.get("owner", {}).get("display_name", "")
            imgs = playlist_data.get("images", [])
            if imgs:
                container_meta["artwork"] = imgs[0].get("url", "")
            container_meta["total_tracks"] = playlist_data.get("tracks", {}).get("total", 0)

            # Paginate through tracks
            results = sp.playlist_items(item_id, additional_types=("track",), limit=min(max_tracks, 100))
            while results and len(raw_items) < max_tracks:
                for item in results.get("items", []):
                    track = item.get("track")
                    if track and track.get("id"):
                        raw_items.append(track)
                    if len(raw_items) >= max_tracks:
                        break

                if results.get("next") and len(raw_items) < max_tracks:
                    results = sp.next(results)
                else:
                    break

        elif item_type == "album":
            album_data = sp.album(item_id)
            container_meta["title"] = album_data.get("name", "Spotify Album")
            container_meta["owner"] = ", ".join([a["name"] for a in album_data.get("artists", [])])
            imgs = album_data.get("images", [])
            if imgs:
                container_meta["artwork"] = imgs[0].get("url", "")
            container_meta["total_tracks"] = album_data.get("total_tracks", 0)

            results = album_data.get("tracks", {})
            for item in results.get("items", [])[:max_tracks]:
                if item and item.get("id"):
                    item["album"] = {"name": container_meta["title"], "images": imgs}
                    raw_items.append(item)

        elif item_type == "track":
            track = sp.track(item_id)
            container_meta["title"] = track.get("name", "Spotify Track")
            raw_items.append(track)
            container_meta["total_tracks"] = 1

    except SpotifyException as exc:
        # If Spotify API blocks due to Premium requirement or token error, fall back to public embed
        print(f"Spotify Web API error ({exc.http_status}): {exc.msg}. Using public embed ripper...")
        return rip_spotify_public(url_or_id, max_tracks=max_tracks)
    except Exception as exc:
        print(f"Spotify API error: {exc}. Falling back to public embed ripper...")
        return rip_spotify_public(url_or_id, max_tracks=max_tracks)

    # Extract Artist IDs to fetch genres in fast batches of 50
    artist_ids = set()
    for track in raw_items:
        artists = track.get("artists", [])
        if artists and artists[0].get("id"):
            artist_ids.add(artists[0]["id"])

    artist_genre_map = {}
    artist_id_list = list(artist_ids)
    for i in range(0, len(artist_id_list), 50):
        chunk = artist_id_list[i : i + 50]
        try:
            artists_resp = sp.artists(chunk)
            for a in artists_resp.get("artists", []):
                if a and a.get("id"):
                    artist_genre_map[a["id"]] = a.get("genres", [])
        except Exception:
            pass

    # Format into standardized song records
    tracks = []
    for track in raw_items:
        artist_objs = track.get("artists", [])
        artist_names = [a.get("name", "Unknown") for a in artist_objs]
        primary_artist_id = artist_objs[0].get("id") if artist_objs else None
        genres = artist_genre_map.get(primary_artist_id, []) if primary_artist_id else []

        album_obj = track.get("album", {})
        album_name = album_obj.get("name", "Unknown Album")
        artwork = ""
        if album_obj.get("images"):
            artwork = album_obj["images"][0].get("url", "")

        tracks.append({
            "id": track["id"],
            "title": track.get("name", "Unknown Title"),
            "artist": ", ".join(artist_names),
            "primary_artist": artist_names[0] if artist_names else "Unknown",
            "album": album_name,
            "artwork": artwork,
            "preview_url": track.get("preview_url") or "",
            "popularity": track.get("popularity", 50),
            "duration_ms": track.get("duration_ms", 0),
            "genres": genres,
            "spotify_url": track.get("external_urls", {}).get("spotify", f"https://open.spotify.com/track/{track['id']}"),
        })

    return container_meta, tracks


def build_playlist_taste_profile(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesize high-level taste profile from ripped Spotify tracks.
    Returns dominant genres, top artists, and mapped GTZAN/catalog genres.
    """
    if not tracks:
        return {
            "dominant_genres": [],
            "top_artists": [],
            "search_genres": [],
            "avg_popularity": 0,
            "track_count": 0,
        }

    all_genres = [g.lower() for t in tracks for g in t.get("genres", []) if g]

    # If genres are not yet enriched, look up top 6 unique artists in iTunes
    if not all_genres:
        seen_artists = list(dict.fromkeys([t.get("primary_artist", "") for t in tracks if t.get("primary_artist")]))
        for artist in seen_artists[:6]:
            try:
                itunes_resp = requests.get(
                    "https://itunes.apple.com/search",
                    params={"term": artist, "media": "music", "entity": "song", "limit": 1},
                    timeout=5,
                )
                if itunes_resp.status_code == 200:
                    results = itunes_resp.json().get("results", [])
                    if results and results[0].get("primaryGenreName"):
                        all_genres.append(results[0]["primaryGenreName"].lower())
            except Exception:
                pass

    genre_series = pd.Series(all_genres).value_counts() if all_genres else pd.Series(dtype=int)
    dominant_genres = genre_series.head(8).index.tolist() if not genre_series.empty else []

    all_artists = [t.get("primary_artist", "Unknown") for t in tracks]
    artist_series = pd.Series(all_artists).value_counts()
    top_artists = artist_series.head(5).index.tolist()

    avg_popularity = round(float(pd.Series([t.get("popularity", 50) for t in tracks]).mean()), 1)

    # Core genres mapped to catalog
    core_genres = ["rock", "pop", "hiphop", "jazz", "blues", "classical", "metal", "disco", "reggae", "country", "electronic", "indie", "r&b", "folk"]
    mapped_genres = []

    for g in dominant_genres:
        for core in core_genres:
            if core in g and core not in mapped_genres:
                mapped_genres.append(core)

    search_genres = list(dict.fromkeys(mapped_genres + dominant_genres[:4]))
    if not search_genres:
        search_genres = top_artists[:4]

    return {
        "dominant_genres": dominant_genres if dominant_genres else ["Pop / Mixed"],
        "top_artists": top_artists,
        "search_genres": search_genres,
        "avg_popularity": avg_popularity,
        "track_count": len(tracks),
    }


def recommend_from_spotify(
    playlist_tracks: List[Dict[str, Any]],
    search_recommendations_fn,
    personalize_fn,
    feedback: Dict[str, Any],
    top_n: int = 15,
    diversity: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Recommends new songs based on a Spotify playlist:
      1. Synthesizes dominant genres from ripped tracks
      2. Searches real music catalog using target genres
      3. Reranks with user taste feedback & diversity penalty
      4. Strictly filters out all songs already in the Spotify playlist
    """
    profile = build_playlist_taste_profile(playlist_tracks)
    search_terms = profile["search_genres"]

    if not search_terms:
        return []

    raw_candidates = search_recommendations_fn(
        search_terms,
        limit_per_genre=35,
        diversity=diversity
    )

    ranked = personalize_fn(
        raw_candidates,
        feedback,
        diversity=diversity
    )

    existing_titles = {t["title"].strip().casefold() for t in playlist_tracks}
    existing_ids = {str(t["id"]).strip().casefold() for t in playlist_tracks}
    blocked_feedback = {str(x) for x in (feedback.get("dislikes", []) + feedback.get("skips", []))}

    new_recommendations = []
    for cand in ranked:
        title_norm = cand.get("title", "").strip().casefold()
        cand_id = str(cand.get("id", "")).strip().casefold()

        if title_norm in existing_titles or cand_id in existing_ids or cand_id in blocked_feedback:
            continue

        new_recommendations.append(cand)
        if len(new_recommendations) >= top_n:
            break

    return new_recommendations
