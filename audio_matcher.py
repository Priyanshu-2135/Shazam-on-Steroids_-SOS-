"""Extract audio features from short catalog previews and compare them to GTZAN seeds.

Preview URLs are third-party catalog previews, not full-track downloads. Some URLs
may be unavailable or undecodable; those songs are returned without an audio score.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import requests


def extract_preview_features(preview_url: str, timeout: int = 12) -> dict[str, float] | None:
    """Download and extract mean Librosa features from a preview URL."""
    if not preview_url:
        return None

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Audio matching needs librosa. Install it with: pip install librosa soundfile"
        ) from exc

    try:
        response = requests.get(
            preview_url,
            timeout=timeout,
            headers={"User-Agent": "SOS-Music-Recommender/1.0"},
        )
        response.raise_for_status()
        if not response.content:
            return None

        # A temporary file lets librosa/audioread use supported codecs such as AAC.
        suffix = ".m4a" if ".m4a" in preview_url.lower() or ".mp4" in preview_url.lower() else ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
            temp.write(response.content)
            temp.flush()
            y, sr = librosa.load(temp.name, sr=22050, mono=True, duration=25.0)

        if y.size < sr:
            return None

        features: dict[str, float] = {}
        features["chroma_stft"] = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        features["rms"] = float(np.mean(librosa.feature.rms(y=y)))
        features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.asarray(tempo).reshape(-1)[0])

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for index, value in enumerate(np.mean(mfcc, axis=1), start=1):
            features[f"mfcc{index}"] = float(value)

        # Approximate harmonic/percussive components for GTZAN's related columns.
        harmonic, percussive = librosa.effects.hpss(y)
        features["harmony"] = float(np.mean(np.abs(harmonic)))
        features["perceptr"] = float(np.mean(np.abs(percussive)))

        return {key: value for key, value in features.items() if np.isfinite(value)}
    except Exception:
        # Preview access/codec failures should not crash the whole Streamlit app.
        return None


def rank_songs_by_audio(
    songs: list[dict[str, Any]],
    seed_track_ids: list[str],
    recommender,
    max_songs: int = 30,
    progress_callback=None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add audio_similarity scores and rank candidates against selected GTZAN seeds."""
    if not seed_track_ids:
        return songs, {"scored": 0, "failed": 0, "skipped": len(songs)}

    catalog = recommender.catalog.copy()
    feature_columns = [
        column for column in recommender.feature_columns
        if column in catalog.columns
    ]
    if not feature_columns:
        return songs, {"scored": 0, "failed": 0, "skipped": len(songs)}

    valid_seed_rows = catalog[catalog["track_id"].astype(str).isin(set(seed_track_ids))]
    if valid_seed_rows.empty:
        return songs, {"scored": 0, "failed": 0, "skipped": len(songs)}

    seed_profile = valid_seed_rows[feature_columns].mean(numeric_only=True)
    reference = catalog[feature_columns].replace([np.inf, -np.inf], np.nan)
    means = reference.mean().fillna(0.0)
    stds = reference.std(ddof=0).replace(0, np.nan).fillna(1.0)
    seed_z = ((seed_profile - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    seed_norm = float(np.linalg.norm(seed_z))

    output = [dict(song) for song in songs]
    scored_count = 0
    failed_count = 0
    candidates = output[:max_songs]
    total = len(candidates)

    for position, song in enumerate(candidates, start=1):
        features = extract_preview_features(str(song.get("preview", "")))
        if not features:
            song["audio_similarity"] = None
            failed_count += 1
        else:
            common = [column for column in feature_columns if column in features]
            if not common:
                song["audio_similarity"] = None
                failed_count += 1
            else:
                candidate_raw = np.array([features[column] for column in common], dtype=float)
                seed_common = np.array([seed_profile[column] for column in common], dtype=float)
                mean_common = np.array([means[column] for column in common], dtype=float)
                std_common = np.array([stds[column] for column in common], dtype=float)
                candidate_z = (candidate_raw - mean_common) / std_common
                seed_z_common = (seed_common - mean_common) / std_common
                candidate_norm = float(np.linalg.norm(candidate_z))
                seed_norm_common = float(np.linalg.norm(seed_z_common))
                if candidate_norm == 0 or seed_norm_common == 0:
                    similarity = 0.0
                else:
                    cosine = float(np.dot(candidate_z, seed_z_common) / (candidate_norm * seed_norm_common))
                    similarity = max(0.0, min(100.0, (cosine + 1.0) * 50.0))
                song["audio_similarity"] = round(similarity, 1)
                scored_count += 1

        if progress_callback:
            progress_callback(position / max(total, 1), position, total)

    # Audio-scored songs first. Songs without decodable previews remain available after them.
    output.sort(
        key=lambda song: (
            song.get("audio_similarity") is not None,
            song.get("audio_similarity") if song.get("audio_similarity") is not None else -1,
            float(song.get("personalized_score", song.get("discovery_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    return output, {"scored": scored_count, "failed": failed_count, "skipped": max(0, len(songs) - total)}
