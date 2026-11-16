
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from ipaddress import ip_address

import librosa
import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="BeatPilot Audio Analysis")

API_KEY = os.getenv("BEATPILOT_API_KEY", "")
ALLOWED_HOSTS = {
    h.strip().lower()
    for h in os.getenv("ALLOWED_AUDIO_HOSTS", "").split(",")
    if h.strip()
}

MAX_BYTES = 25 * 1024 * 1024
MAX_DURATION = 900


class AnalysisRequest(BaseModel):
    track_id: str
    audio_url: HttpUrl


def check_url(url):
    parsed = urlparse(str(url))
    host = (parsed.hostname or "").lower()

    if parsed.scheme != "https" or not host:
        raise HTTPException(400, "A valid HTTPS audio URL is required")

    if host not in ALLOWED_HOSTS:
        raise HTTPException(403, "Audio storage host is not allowed")

    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        for item in addresses:
            ip = ip_address(item[4][0])
            if not ip.is_global:
                raise HTTPException(403, "Private network addresses are blocked")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not validate audio host")


def download_audio(url, destination):
    check_url(url)

    total = 0
    try:
        with requests.get(
            str(url),
            stream=True,
            timeout=(10, 60),
            allow_redirects=False
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise HTTPException(400, "URL did not return an audio file")

            with open(destination, "wb") as file:
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(413, "Audio file exceeds 25 MB")
                    file.write(chunk)

    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(400, "Could not download audio file")

    if total == 0:
        raise HTTPException(400, "Audio file is empty")


def estimate_key(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    profile = np.mean(chroma, axis=1)

    if not np.all(np.isfinite(profile)) or np.std(profile) < 1e-8:
        return None, None

    major = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
         2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
         2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]

    scores = []
    for i in range(12):
        rotated = np.roll(profile, -i)
        scores.append((np.corrcoef(rotated, major)[0, 1], names[i], "major"))
        scores.append((np.corrcoef(rotated, minor)[0, 1], names[i], "minor"))

    scores = [s for s in scores if np.isfinite(s[0])]
    if not scores:
        return None, None

    best = max(scores, key=lambda s: s[0])
    return f"{best[1]} {best[2]}", round(float(best[0]), 3)


@app.get("/health")
def health():
    return {"status": "ok", "service": "BeatPilot Audio Analysis"}


@app.post("/analyze")
def analyze(
    request: AnalysisRequest,
    x_api_key: str = Header(default="")
):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp:
            temp_path = temp.name

        download_audio(str(request.audio_url), temp_path)

        y, sr = librosa.load(temp_path, sr=22050, mono=True)

        duration = len(y) / sr
        if duration < 10:
            raise HTTPException(400, "Audio must be at least 10 seconds")
        if duration > MAX_DURATION:
            raise HTTPException(400, "Audio exceeds 15 minutes")

        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, units="frames"
        )
        bpm = float(np.asarray(tempo).reshape(-1)[0])

        if not np.isfinite(bpm) or bpm <= 0:
            bpm = None

        key, key_confidence = estimate_key(y, sr)

        rms = librosa.feature.rms(y=y)[0]
        average_rms = float(np.mean(rms))
        energy = round(float(np.clip(average_rms * 100, 0, 100)), 1)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_positions = [
            round(float(t), 3) for t in beat_times[:500]
        ]

        return {
            "track_id": request.track_id,
            "duration": round(duration, 2),
            "bpm": round(bpm, 2) if bpm else None,
            "bpm_confidence": None,
            "key": key,
            "key_confidence": key_confidence,
            "camelot": None,
            "energy": energy,
            "energy_method": "rough_rms_estimate",
            "danceability": None,
            "loudness": None,
            "beat_positions": beat_positions,
            "analysis_version": "0.1.0"
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "Audio analysis failed")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
