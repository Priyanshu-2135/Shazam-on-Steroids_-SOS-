# Shazam-on-Steroids (SOS) 🎧

**Shazam-on-Steroids (SOS)** is an advanced music recommendation, audio fingerprinting, and genre classification platform. It combines real-time DSP audio feature extraction (MFCCs, Chroma, Spectral Centroid, Tempo), machine learning trained on the GTZAN audio dataset, and live music catalog exploration with personalized taste learning.

---

## ✨ Key Features

1. **🎙️ Real-Time Audio Fingerprinting ("True Shazam")**
   - Record directly from your microphone or upload audio files (MP3, WAV, M4A, OGG, FLAC).
   - Extracts all 57 GTZAN-compatible acoustic DSP features via `librosa`.
   - Real-time genre prediction with confidence percentages and probability distributions.
   - Computes musical DNA: Estimated Tempo (BPM), Energy Level, Timbre Tone, and Rhythmic Texture.

2. **🟢 Spotify Playlist Ripper & Recommender (100% Free)**
   - Rip song data, artist genres, album artwork, and popularity from any Spotify playlist or album URL.
   - Synthesizes a **Musical Taste Profile** (dominant musical styles, top artists, popularity metrics).
   - Generates new song recommendations matching the playlist's vibe while strictly filtering out duplicate songs already in your playlist.
   - Built on Spotify's free developer tier (zero billing, no credit card required).

3. **⚡ Vector Similarity & Explainable AI (XAI) Discovery**
   - Direct audio-to-catalog matching using vector cosine similarity.
   - Explainable AI match cards: provides exact reasons why a track was recommended (e.g. `🥁 128 BPM Match`, `⚡ High Energy`, `🎸 Timbre Match`).
   - Artist diversity controls to avoid artist repetition and encourage discovery.

4. **🔎 Live Catalog Search & Streaming Previews**
   - Integrated with Apple's iTunes music catalog (no API keys required).
   - In-app 30-second audio streaming with multi-country failover.
   - Direct one-click fallback links to full songs on YouTube and Spotify.

5. **💜 Continuous Personal Taste Learning**
   - Like, Dislike, and Skip feedback loop stored persistently.
   - Dynamic re-ranking algorithm weighting saved preferences.
   - Visual taste analytics and genre preference charts.

---

## 🚀 Quick Start

### 1. Automated Setup
Clone or open the repository, then run the automated setup script:
```bash
./setup.sh
```
This script automatically:
* Creates an isolated virtual environment (`.venv/`)
* Upgrades `pip`
* Installs all dependencies from `requirements.txt`

### 2. Launch the Application
Start the Streamlit interface with:
```bash
./run.sh
```
Or manually via:
```bash
source .venv/bin/activate
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
SOS/
├── audio_processor.py         # DSP audio extraction & genre prediction engine
├── recommendation_engine.py   # Vector similarity engine, XAI metrics & CLI demo
├── music_catalog.py           # iTunes API catalog search & streaming fallback
├── app.py                     # Interactive Streamlit web application
├── music.py                   # Model training script (Random Forest on GTZAN)
├── music_demo.py              # CLI genre prediction demo
├── music_genre_model.pkl      # Pre-trained Random Forest genre model
├── features_30_sec.csv        # GTZAN dataset 30-second acoustic features
├── features_3_sec.csv         # GTZAN dataset 3-second acoustic features
├── catalog_feedback.json      # Persistent user feedback (likes, dislikes, skips)
├── requirements.txt           # Python dependencies
├── setup.sh                   # Environment setup script
├── run.sh                     # Launch script
└── .gitignore                 # Environment and cache ignore rules
```

---

## 🛠️ Tech Stack
* **Language & Frameworks**: Python 3.10+, Streamlit, Librosa, Scikit-Learn, NumPy, Pandas
* **Machine Learning**: Random Forest Classifier (10 GTZAN genres, `GroupShuffleSplit` evaluation)
* **Audio DSP**: MFCCs, Chroma STFT, Spectral Centroid, Bandwidth, Rolloff, Zero Crossing Rate, RMS Energy, HPSS Harmony/Percussive separation, Tempo
