"""Small Spotify Web API integration for the local SOS Streamlit app.

Uses Spotify OAuth authorization-code flow and the Spotify Web API directly.
Credentials and cached tokens stay in local files and should never be committed.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
TOKEN_FILE = BASE_DIR / ".spotify_token.json"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-private playlist-modify-public"


def spotify_configured() -> bool:
    return bool(CLIENT_ID and CLIENT_SECRET)


def get_authorize_url() -> str:
    import streamlit as st
    if not spotify_configured():
        raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.")
    state = secrets.token_urlsafe(24)
    st.session_state["spotify_oauth_state"] = state
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "show_dialog": "true",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _save_token(token: dict) -> None:
    token["obtained_at"] = int(time.time())
    TOKEN_FILE.write_text(json.dumps(token, indent=2), encoding="utf-8")


def _read_token() -> dict | None:
    try:
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _exchange_token(data: dict) -> dict:
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(raw).decode("ascii"),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=20)
    if not response.ok:
        raise RuntimeError(f"Spotify token request failed ({response.status_code}): {response.text[:400]}")
    token = response.json()
    _save_token(token)
    return token


def finish_authorization(code: str, returned_state: str | None) -> bool:
    import streamlit as st
    expected_state = st.session_state.get("spotify_oauth_state")
    if not code or not expected_state or not returned_state or returned_state != expected_state:
        return False
    _exchange_token({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
    })
    st.session_state.pop("spotify_oauth_state", None)
    return True


def get_spotify_token() -> str | None:
    token = _read_token()
    if not token:
        return None
    expires_at = int(token.get("obtained_at", 0)) + int(token.get("expires_in", 0))
    if time.time() >= expires_at - 60:
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            try:
                TOKEN_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        refreshed = _exchange_token({"grant_type": "refresh_token", "refresh_token": refresh_token})
        if "refresh_token" not in refreshed:
            refreshed["refresh_token"] = refresh_token
            _save_token(refreshed)
        token = refreshed
    return token.get("access_token")


def _api_request(method: str, path: str, token: str, **kwargs):
    response = requests.request(
        method,
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=25,
        **kwargs,
    )
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"Spotify API request failed ({response.status_code}): {detail}")
    return response.json() if response.content else {}


def spotify_search_track(title: str, artist: str, token: str) -> dict | None:
    query = f'track:"{title}" artist:"{artist}"' if artist else f'track:"{title}"'
    data = _api_request("GET", "/search", token, params={"q": query, "type": "track", "limit": 5})
    items = data.get("tracks", {}).get("items", [])
    if not items:
        # Relax exact field search if punctuation/metadata differs.
        fallback = f"{title} {artist}".strip()
        data = _api_request("GET", "/search", token, params={"q": fallback, "type": "track", "limit": 5})
        items = data.get("tracks", {}).get("items", [])
    if not items:
        return None
    wanted_artist = artist.casefold().strip()
    wanted_title = title.casefold().strip()
    def score(item):
        name = item.get("name", "").casefold()
        artists = [a.get("name", "").casefold() for a in item.get("artists", [])]
        return (2 if name == wanted_title else 0) + (2 if wanted_artist and wanted_artist in artists else 0)
    best = max(items, key=score)
    return {
        "uri": best.get("uri"),
        "name": best.get("name", title),
        "artist": ", ".join(a.get("name", "") for a in best.get("artists", [])),
    }


def create_spotify_playlist(name: str, track_uris: list[str], token: str, public: bool = False) -> dict:
    playlist = _api_request(
        "POST",
        "/me/playlists",
        token,
        json={"name": name, "description": "Created by Shazam-on-Steroids (SOS)", "public": public},
    )
    # Spotify accepts up to 100 items per add request.
    for start in range(0, len(track_uris), 100):
        _api_request(
            "POST",
            f"/playlists/{playlist['id']}/items",
            token,
            json={"uris": track_uris[start:start + 100]},
        )
    return {"id": playlist.get("id"), "url": playlist.get("external_urls", {}).get("spotify")}
