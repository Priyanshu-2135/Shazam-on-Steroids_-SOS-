SOS - Audio-feature matching update
===================================

Files in this package
---------------------
app.py                    Streamlit app with audio matching integrated
recommendation_engine.py  Personalized GTZAN recommender and corrected track IDs
audio_matcher.py          Downloads catalog previews, extracts audio features,
                          and ranks the candidates against selected GTZAN tracks

Install dependencies in the project's activated virtual environment
--------------------------------------------------------------------
python -m pip install librosa soundfile requests

If preview decoding fails for .m4a/.mp4 previews on Windows, install FFmpeg and
make sure ffmpeg.exe is on PATH. The app continues to work with genre and
feedback-based ranking if preview audio cannot be decoded.

Run
---
python -m streamlit run app.py

How to use
----------
1. Select one or more GTZAN seed tracks in the app.
2. Leave "Rank by audio similarity" enabled in the sidebar.
3. Click "Find real songs for me".
4. SOS finds catalog candidates using genre searches, then analyzes up to 30
   preview URLs and ranks decodable previews against the selected GTZAN audio
   feature profile.
5. Songs with a score show "Audio similarity to your GTZAN seeds".

Features extracted from preview audio
-------------------------------------
- Tempo
- RMS energy
- Chroma
- Spectral centroid, bandwidth, and rolloff
- Zero-crossing rate
- MFCC 1-20
- Approximate harmonic/percussive energy

Important limitations
---------------------
- This compares short catalog previews, not full commercial tracks.
- Catalog search is still genre-guided; audio features re-rank up to 30 of the
  candidates rather than searching the entire streaming catalog.
- A preview may be missing, blocked, or encoded with a codec unavailable on
  the computer. Such songs remain in the results but may have no audio score.
- The displayed 0-100 value is a relative cosine-similarity score, not a
  calibrated probability or guarantee that a person will like the song.
- Audio matching is applied only when GTZAN seed tracks are selected. If no
  seeds are selected, the app uses saved feedback and genre-guided discovery.
