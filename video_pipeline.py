# video_pipeline.py
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import imageio_ffmpeg
from openai import OpenAI


SCRIPT_EXTS = {".txt", ".md", ".json"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".webm", ".flac"}


# ----------------------------
# OS / ffmpeg helpers
# ----------------------------

def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + (p.stderr or "")
            + "\n\nSTDOUT:\n"
            + (p.stdout or "")
        )

def safe_slug(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:max_len] if s else "chapter")

def extract_int_prefix(name: str) -> Optional[int]:
    base = os.path.basename(name)
    m = re.search(r"(^|\b)(\d{1,3})(\b|_|\s|-)", base)
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None

def get_media_duration_seconds(path: str) -> float:
    """
    Uses `ffmpeg -i` stderr parsing (works even without ffprobe).
    """
    ff = ffmpeg_exe()
    p = subprocess.run([ff, "-i", path], capture_output=True, text=True)
    txt = (p.stderr or "") + "\n" + (p.stdout or "")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
    if not m:
        return 0.0
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss


# ----------------------------
# ZIP + file discovery
# ----------------------------

def extract_zip_to_temp(zip_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (workdir, extract_dir)
    """
    workdir = tempfile.mkdtemp(prefix="uappress_video_")
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(extract_dir)

    return workdir, extract_dir

def find_files(root_dir: str) -> Tuple[List[str], List[str]]:
    scripts, audios = [], []
    for r, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            path = os.path.join(r, f)
            if ext in SCRIPT_EXTS:
                scripts.append(path)
            elif ext in AUDIO_EXTS:
                audios.append(path)
    return scripts, audios

def read_script_file(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".json":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ["text", "chapter_text", "content", "script"]:
                if k in data and isinstance(data[k], str):
                    return data[k].strip()
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

def best_match_pairs(scripts: List[str], audios: List[str]) -> List[Dict]:
    """
    Pair script + audio by chapter number if possible; fallback to token overlap.
    """
    audio_by_num: Dict[int, List[str]] = {}
    for a in audios:
        n = extract_int_prefix(a)
        if n is not None:
            audio_by_num.setdefault(n, []).append(a)

    def token_score(script_path: str, audio_path: str) -> int:
        sname = os.path.splitext(os.path.basename(script_path).lower())[0]
        aname = os.path.splitext(os.path.basename(audio_path).lower())[0]
        stoks = set(re.split(r"[^a-z0-9]+", sname)); stoks.discard("")
        atoks = set(re.split(r"[^a-z0-9]+", aname)); atoks.discard("")
        return len(stoks.intersection(atoks))

    pairs = []
    used_audio = set()

    for s in sorted(scripts):
        sn = extract_int_prefix(s)
        best_a = None

        if sn is not None and sn in audio_by_num:
            candidates = [a for a in audio_by_num[sn] if a not in used_audio]
            if candidates:
                best_a = max(candidates, key=lambda a: token_score(s, a))

        if best_a is None:
            candidates = [a for a in audios if a not in used_audio]
            if candidates:
                best_a = max(candidates, key=lambda a: token_score(s, a))

        if best_a:
            used_audio.add(best_a)
            base = os.path.splitext(os.path.basename(s))[0]
            pairs.append({
                "chapter_no": sn,
                "title_guess": base,
                "script_path": s,
                "audio_path": best_a,
            })

    pairs.sort(key=lambda p: (p["chapter_no"] if p["chapter_no"] is not None else 9999, p["title_guess"].lower()))
    return pairs


# ----------------------------
# Scene planning (text -> JSON scenes)
# ----------------------------

SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration chapter into a list of short visual scenes for AI video generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt, on_screen_text(optional). "
    "Style: cinematic documentary b-roll and reenactment vibes; realistic lighting; camera movement notes. "
    "Avoid brand names, copyrighted characters, celebrity likeness, and explicit violence/gore. "
    "Keep prompts concise (1–3 sentences)."
)

def plan_scenes(
    client: OpenAI,
    chapter_title: str,
    chapter_text: str,
    *,
    max_scenes: int,
    seconds_per_scene: int,
    model: str = "gpt-5-mini",
) -> List[Dict]:
    payload = {
        "chapter_title": chapter_title,
        "max_scenes": max_scenes,
        "seconds_per_scene": seconds_per_scene,
        "chapter_text": chapter_text,
        "output": "STRICT_JSON_LIST_ONLY",
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )

    text = (resp.output_text or "").strip()
    # Extract JSON list if wrapped
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
    if m:
        text = m.group(1)

    scenes = json.loads(text)
    out: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        out.append({
            "scene": int(sc.get("scene", i)),
            "seconds": int(sc.get("seconds", seconds_per_scene)),
            "prompt": str(sc["prompt"]).strip(),
            "on_screen_text": (str(sc["on_screen_text"]).strip() if sc.get("on_screen_text") else None),
        })
    return out


# ----------------------------
# Video generation (Sora) - sync polling
# ----------------------------

def generate_video_clip(
    client: OpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,
    out_path: str,
    poll_every_s: int = 2,
) -> str:
    """
    Creates video job, polls until completed, downloads MP4 bytes to out_path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    video = client.videos.create(
        model=model,
        prompt=prompt,
        seconds=str(seconds),
        size=size,
    )

    while getattr(video, "status", None) in ("queued", "in_progress"):
        time.sleep(poll_every_s)
        video = client.videos.retrieve(video.id)

    if getattr(video, "status", None) != "completed":
        err = getattr(getattr(video, "error", None), "message", "Video generation failed")
        raise RuntimeError(f"Video failed: {err}")

    content = client.videos.download_content(video.id, variant="video")
    # Stainless binary response has write_to_file
    content.write_to_file(out_path)
    return out_path


# ----------------------------
# Transcription -> SRT (always)
# ----------------------------

def transcribe_audio_to_srt(
    client: OpenAI,
    audio_path: str,
    *,
    model: str = "whisper-1",
    language: str = "en",
) -> str:
    """
    Returns SRT string. Uses whisper-1 for reliable SRT output.
    """
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="srt",
            language=language,
        )

    # SDK may return a string or an object depending on version; handle both.
    if isinstance(tr, str):
        return tr
    if hasattr(tr, "text") and isinstance(tr.text, str):
