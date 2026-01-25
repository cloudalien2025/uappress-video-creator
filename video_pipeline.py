# video_pipeline.py
from __future__ import annotations

import asyncio
import os
import re
import json
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import imageio_ffmpeg
from openai import OpenAI, AsyncOpenAI


# ----------------------------
# utils
# ----------------------------

def ffmpeg_path() -> str:
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
    """
    Best-effort chapter number extraction:
      "01 - Intro.mp3" -> 1
      "Chapter 12 ...txt" -> 12
      "1_intro.txt" -> 1
    """
    base = os.path.basename(name)
    m = re.search(r"(^|\b)(\d{1,3})(\b|_|\s|-)", base)
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None


# ----------------------------
# finding inputs inside ZIP
# ----------------------------

SCRIPT_EXTS = {".txt", ".md", ".json"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}

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
        # allow a few formats: {"text": "..."} or {"chapter_text":"..."} etc.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ["text", "chapter_text", "content", "script"]:
                if k in data and isinstance(data[k], str):
                    return data[k]
        # fallback
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

def best_match_pairs(scripts: List[str], audios: List[str]) -> List[Dict]:
    """
    Attempts to pair script + audio by:
      1) shared chapter number prefix
      2) similar filename tokens
    Returns list of dicts: {chapter_no, title_guess, script_path, audio_path}
    """
    # index audios by chapter number if possible
    audio_by_num: Dict[int, List[str]] = {}
    for a in audios:
        n = extract_int_prefix(a)
        if n is not None:
            audio_by_num.setdefault(n, []).append(a)

    def score(script_path: str, audio_path: str) -> int:
        sname = os.path.splitext(os.path.basename(script_path).lower())[0]
        aname = os.path.splitext(os.path.basename(audio_path).lower())[0]
        stoks = set(re.split(r"[^a-z0-9]+", sname))
        atoks = set(re.split(r"[^a-z0-9]+", aname))
        stoks.discard(""); atoks.discard("")
        return len(stoks.intersection(atoks))

    pairs = []
    used_audio = set()

    # First pass: match by number
    for s in sorted(scripts):
        sn = extract_int_prefix(s)
        best_a = None
        if sn is not None and sn in audio_by_num:
            # pick the first unused, else best token match
            candidates = [a for a in audio_by_num[sn] if a not in used_audio]
            if candidates:
                best_a = max(candidates, key=lambda a: score(s, a))

        # Second pass: best token match overall
        if best_a is None:
            candidates = [a for a in audios if a not in used_audio]
            if candidates:
                best_a = max(candidates, key=lambda a: score(s, a))

        if best_a:
            used_audio.add(best_a)
            base = os.path.splitext(os.path.basename(s))[0]
            pairs.append({
                "chapter_no": sn,
                "title_guess": base,
                "script_path": s,
                "audio_path": best_a
            })

    # Sort: numeric first, then name
    def sort_key(p):
        return (p["chapter_no"] if p["chapter_no"] is not None else 9999, p["title_guess"].lower())
    pairs.sort(key=sort_key)
    return pairs


# ----------------------------
# scene planning
# ----------------------------

SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration chapter into a list of short visual scenes for AI video generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt, on_screen_text(optional). "
    "Style: cinematic documentary b-roll and reenactment vibes; realistic lighting; camera movement notes. "
    "Avoid brand names, copyrighted characters, celebrity likeness, and explicit violence/gore. "
    "No medical claims. No hateful or sexual content. Keep prompts concise."
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
        "output": "STRICT_JSON_LIST_ONLY"
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(payload)}
        ],
    )
    text = resp.output_text.strip()

    # Extract JSON list if model wrapped it
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
# video generation
# ----------------------------

async def generate_clip_bytes(
    client: AsyncOpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,
) -> bytes:
    """
    Uses OpenAI Sora Video API.
    Async job: create_and_poll -> download_content (MP4 bytes). :contentReference[oaicite:1]{index=1}
    """
    video = await client.videos.create_and_poll(
        model=model,
        prompt=prompt,
        seconds=str(seconds),
        size=size,
    )
    if video.status != "completed":
        raise RuntimeError(f"Video job did not complete: status={video.status} id={video.id}")

    content = client.videos.download_content(video_id=video.id)
    return content.read()

def write_bytes(path: str, b: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


# ----------------------------
# ffmpeg assembly
# ----------------------------

def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    ffmpeg = ffmpeg_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # concat demuxer list
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for p in mp4_paths:
            tf.write(f"file '{p}'\n")
        list_path = tf.name

    try:
        run_cmd([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path

def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    ffmpeg = ffmpeg_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    run_cmd([
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ])
    return out_path

def reencode_if_needed(in_path: str, out_path: str) -> str:
    """
    Optional safety net if concat-copy fails due to codec mismatch.
    Re-encodes to H.264/AAC for consistent concatenation.
    """
    ffmpeg = ffmpeg_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_cmd([
        ffmpeg, "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ])
    return out_path

def build_chapter_mp4(scene_mp4s: List[str], mp3_path: str, out_dir: str, chapter_slug: str) -> str:
    stitched = os.path.join(out_dir, f"{chapter_slug}_stitched.mp4")
    final = os.path.join(out_dir, f"{chapter_slug}_final.mp4")
    concat_mp4s(scene_mp4s, stitched)
    mux_audio(stitched, mp3_path, final)
    return final

def build_full_documentary(chapter_mp4s: List[str], out_path: str) -> str:
    return concat_mp4s(chapter_mp4s, out_path)
