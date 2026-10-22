"""
Audio processing module for Shazam-on-Steroids (SOS).
Handles microphone input, audio file upload, DSP feature extraction
compatible with the GTZAN dataset, and real-time genre prediction.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
import librosa

# Expected 57 GTZAN audio feature columns in exact order
EXPECTED_COLUMNS = [
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var',
    'perceptr_mean', 'perceptr_var',
    'tempo',
    'mfcc1_mean', 'mfcc1_var',
    'mfcc2_mean', 'mfcc2_var',
    'mfcc3_mean', 'mfcc3_var',
    'mfcc4_mean', 'mfcc4_var',
    'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var',
    'mfcc7_mean', 'mfcc7_var',
    'mfcc8_mean', 'mfcc8_var',
    'mfcc9_mean', 'mfcc9_var',
    'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var',
    'mfcc12_mean', 'mfcc12_var',
    'mfcc13_mean', 'mfcc13_var',
    'mfcc14_mean', 'mfcc14_var',
    'mfcc15_mean', 'mfcc15_var',
    'mfcc16_mean', 'mfcc16_var',
    'mfcc17_mean', 'mfcc17_var',
    'mfcc18_mean', 'mfcc18_var',
    'mfcc19_mean', 'mfcc19_var',
    'mfcc20_mean', 'mfcc20_var'
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "music_genre_model.pkl"


def load_audio_signal(
    audio_source: Union[str, Path, io.BytesIO, bytes],
    sr: int = 22050,
    max_duration: float = 30.0
) -> Tuple[np.ndarray, int]:
    """
    Load an audio signal from a file path, BytesIO, or raw bytes into a mono
    waveform at target sample rate sr.
    """
    temp_file_path = None
    try:
        if isinstance(audio_source, (str, Path)):
            file_path = str(audio_source)
        elif isinstance(audio_source, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_source)
                tmp.flush()
                temp_file_path = tmp.name
                file_path = temp_file_path
        elif hasattr(audio_source, "read"):
            content = audio_source.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(content)
                tmp.flush()
                temp_file_path = tmp.name
                file_path = temp_file_path
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

        # Load audio (downmix to mono, resample to target sr)
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True, duration=max_duration)

        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            raise ValueError("The audio signal is silent or empty. Please check your recording or file.")

        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=35)
        if len(y) == 0:
            raise ValueError("The audio signal has no audible sound after trimming silence.")

        return y, loaded_sr

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def extract_gtzan_features(y: np.ndarray, sr: int = 22050) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Extract all 57 audio DSP features matching the GTZAN dataset representation.
    """
    feats: Dict[str, float] = {}

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats['chroma_stft_mean'] = float(np.mean(chroma))
    feats['chroma_stft_var'] = float(np.var(chroma))

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_var'] = float(np.var(rms))

    # 3. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats['spectral_centroid_mean'] = float(np.mean(spec_cent))
    feats['spectral_centroid_var'] = float(np.var(spec_cent))

    # 4. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
    feats['spectral_bandwidth_var'] = float(np.var(spec_bw))

    # 5. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats['rolloff_mean'] = float(np.mean(rolloff))
    feats['rolloff_var'] = float(np.var(rolloff))

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    feats['zero_crossing_rate_mean'] = float(np.mean(zcr))
    feats['zero_crossing_rate_var'] = float(np.var(zcr))

    # 7. Harmonic & Percussive Components
    harm, perc = librosa.effects.hpss(y)
    feats['harmony_mean'] = float(np.mean(harm))
    feats['harmony_var'] = float(np.var(harm))
    feats['perceptr_mean'] = float(np.mean(perc))
    feats['perceptr_var'] = float(np.var(perc))

    # 8. Tempo (BPM)
    tempo = librosa.feature.tempo(y=y, sr=sr)
    feats['tempo'] = float(np.atleast_1d(tempo)[0])

    # 9. MFCCs 1 through 20
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        feats[f'mfcc{i}_mean'] = float(np.mean(mfccs[i - 1]))
        feats[f'mfcc{i}_var'] = float(np.var(mfccs[i - 1]))

    df = pd.DataFrame([feats])[EXPECTED_COLUMNS]
    return feats, df


def predict_audio_genre(
    features_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Predict the genre and probability distribution using the trained Random Forest model.
    """
    model_file = Path(model_path or DEFAULT_MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"Genre model file not found at: {model_file}")

    saved = joblib.load(model_file)
    model = saved["model"]
    feature_cols = saved.get("feature_columns", EXPECTED_COLUMNS)

    # Ensure columns match model expectations
    clean_df = features_df[feature_cols].copy()

    prediction = str(model.predict(clean_df)[0])
    probabilities = model.predict_proba(clean_df)[0]
    classes = [str(c) for c in model.classes_]

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probabilities)
    }
    # Sort descending by probability
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    confidence = sorted_probs.get(prediction, 0.0)

    return {
        "predicted_genre": prediction,
        "confidence": confidence,
        "probabilities": sorted_probs,
        "top_3": list(sorted_probs.items())[:3]
    }


def compute_audio_profile(y: np.ndarray, feats: Dict[str, float]) -> Dict[str, Any]:
    """
    Extract human-readable qualitative descriptors from raw acoustic features.
    """
    tempo = feats.get("tempo", 120.0)
    rms_mean = feats.get("rms_mean", 0.1)
    spec_cent = feats.get("spectral_centroid_mean", 2000.0)
    harmony_var = feats.get("harmony_var", 0.005)
    perceptr_var = feats.get("perceptr_var", 0.005)

    # Energy qualitative descriptor
    if rms_mean > 0.18:
        energy_desc = "High Energy (Intense / Loud)"
    elif rms_mean > 0.08:
        energy_desc = "Moderate Energy"
    else:
        energy_desc = "Low Energy (Mellow / Ambient)"

    # Brightness / Timbre qualitative descriptor
    if spec_cent > 2800:
        brightness_desc = "Very Bright & Crisp (Treble-heavy / Percussive)"
    elif spec_cent > 1800:
        brightness_desc = "Balanced Frequency Distribution"
    else:
        brightness_desc = "Warm & Deep (Bass-heavy / Acoustic)"

    # Harmonic vs Rhythmic balance
    tot = harmony_var + perceptr_var + 1e-6
    harm_ratio = harmony_var / tot
    if harm_ratio > 0.6:
        feel_desc = "Harmonically Rich / Melodic"
    elif harm_ratio < 0.4:
        feel_desc = "Rhythm & Beat Driven"
    else:
        feel_desc = "Balanced Harmony & Rhythm"

    return {
        "bpm": round(tempo, 1),
        "energy": energy_desc,
        "brightness": brightness_desc,
        "texture": feel_desc,
        "rms": round(rms_mean, 4),
        "centroid": round(spec_cent, 1),
    }


def explain_match(query_features: Dict[str, float], candidate_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare query audio features with a candidate track to generate explainable AI (XAI)
    match percentages and readable badges.
    """
    # 1. Tempo match
    t_q = float(query_features.get("tempo", 120.0))
    t_c = float(candidate_features.get("tempo", 120.0))
    tempo_diff = abs(t_q - t_c)
    # Check half-time / double-time alignment
    tempo_diff_half = abs(t_q - 2 * t_c)
    tempo_diff_double = abs(2 * t_q - t_c)
    effective_diff = min(tempo_diff, tempo_diff_half, tempo_diff_double)
    tempo_match = max(0, min(100, int((1.0 - effective_diff / max(t_q, t_c, 1)) * 100)))

    # 2. Energy match (RMS)
    rms_q = float(query_features.get("rms_mean", 0.1))
    rms_c = float(candidate_features.get("rms_mean", 0.1))
    rms_ratio = min(rms_q, rms_c) / (max(rms_q, rms_c) + 1e-6)
    energy_match = max(0, min(100, int(rms_ratio * 100)))

    # 3. Brightness match (Spectral Centroid)
    cent_q = float(query_features.get("spectral_centroid_mean", 2000.0))
    cent_c = float(candidate_features.get("spectral_centroid_mean", 2000.0))
    cent_ratio = min(cent_q, cent_c) / (max(cent_q, cent_c) + 1e-6)
    brightness_match = max(0, min(100, int(cent_ratio * 100)))

    # 4. Badges
    badges = []
    if tempo_match >= 85:
        badges.append(f"🥁 {round(t_c)} BPM Match ({tempo_match}%)")
    elif tempo_match >= 70:
        badges.append(f"🥁 Similar Pace ({round(t_c)} BPM)")

    if energy_match >= 85:
        badges.append(f"⚡ Energy Match ({energy_match}%)")

    if brightness_match >= 85:
        badges.append(f"🎸 Timbre Tone Match ({brightness_match}%)")

    return {
        "tempo_match_pct": tempo_match,
        "energy_match_pct": energy_match,
        "brightness_match_pct": brightness_match,
        "badges": badges,
        "tempo": round(t_c, 1),
    }
