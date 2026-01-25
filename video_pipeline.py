# video_pipeline.py
from __future__ import annotations

import io
import os
import re
import json
import time
import zipfile
import base64
import tempfile
import subprocess
import hashlib
from typing import Dict, List, Optional, Tuple

import imageio_ffmpeg
from openai import OpenAI


# ----------------------------
# Cache
# ----------------------------
# NOTE: Streamlit Cloud usually persists filesystem across reruns, but may reset on rebuilds.
# This cache speeds up testing and prevents repeated OpenAI + ffmpeg work for identical scenes.
CACHE_DIR = os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _cache_key(*parts: str) -> str:
    s = "||".join([str(p).strip() for p in parts if p is not None])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


# ----------------------------
# Extensions / discovery
# ----------------------------

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
    "You convert a documentary narration chapter into a list of short visual scenes for AI generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt, on_screen_text(optional). "
    "Style: cinematic documentary b-roll / reenactment vibes; realistic lighting; camera movement notes. "
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
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
    if m:
        text = m.group(1)

    scenes = json.loads(text)
    out: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        out.append({
            # Robust: always trust loop index, not model-provided labels
            "scene": i,
            "seconds": int(sc.get("seconds", seconds_per_scene)),
            "prompt": str(sc.get("prompt", "")).strip(),
            "on_screen_text": (
                str(sc.get("on_screen_text")).strip()
                if sc.get("on_screen_text") is not None
                else None
            ),
        })
    return out


# ----------------------------
# Transcription -> SRT
# ----------------------------

def transcribe_audio_to_srt(
    client: OpenAI,
    audio_path: str,
    *,
    model: str = "whisper-1",
    language: str = "en",
) -> str:
    """
    Returns SRT string.
    """
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="srt",
            language=language,
        )

    if isinstance(tr, str):
        return tr
    if hasattr(tr, "text") and isinstance(tr.text, str):
        return tr.text
    return str(tr)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------
# SRT shifting / merging
# ----------------------------

def srt_time_to_seconds(t: str) -> float:
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def seconds_to_srt_time(x: float) -> str:
    if x < 0:
        x = 0.0
    hh = int(x // 3600)
    x -= hh * 3600
    mm = int(x // 60)
    x -= mm * 60
    ss = int(x)
    ms = int(round((x - ss) * 1000))
    if ms >= 1000:
        ss += 1
        ms -= 1000
    if ss >= 60:
        mm += 1
        ss -= 60
    if mm >= 60:
        hh += 1
        mm -= 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def shift_srt(srt_text: str, offset_seconds: float) -> str:
    def repl(match: re.Match) -> str:
        start = match.group(1)
        end = match.group(2)
        s = seconds_to_srt_time(srt_time_to_seconds(start) + offset_seconds)
        e = seconds_to_srt_time(srt_time_to_seconds(end) + offset_seconds)
        return f"{s} --> {e}"

    pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
    return pattern.sub(repl, srt_text)


def renumber_srt_blocks(srt_text: str) -> str:
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.S)
    out_blocks = []
    n = 1
    for b in blocks:
        lines = b.strip().splitlines()
        if not lines:
            continue
        if re.fullmatch(r"\d+", (lines[0].strip() or "")):
            lines = lines[1:]
        out_blocks.append(str(n) + "\n" + "\n".join(lines))
        n += 1
    return "\n\n".join(out_blocks) + "\n"


# ----------------------------
# ffmpeg assembly: concat, mux, subtitles
# ----------------------------

def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for p in mp4_paths:
            tf.write(f"file '{p}'\n")
        list_path = tf.name

    try:
        run_cmd([ff, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_cmd([
        ff, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        out_path
    ])
    return out_path


def _escape_for_ffmpeg_filter(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_in_subtitles(video_path: str, srt_path: str, out_path: str) -> str:
    """
    Burns subtitles into video (always visible). Re-encodes.
    """
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    srt_escaped = _escape_for_ffmpeg_filter(srt_path)

    run_cmd([
        ff, "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_escaped}'",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ])
    return out_path


def embed_srt_softsubs(video_path: str, srt_path: str, out_path: str) -> str:
    """
    Keep signature for app.py compatibility. Burns subtitles in.
    """
    return burn_in_subtitles(video_path, srt_path, out_path)


def reencode_mp4(in_path: str, out_path: str) -> str:
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_cmd([
        ff, "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ])
    return out_path


# ----------------------------
# Image -> Ken Burns MP4 generation (with retries + safe sizing + CACHE)
# ----------------------------

def _parse_size(size: str) -> Tuple[int, int]:
    if "x" not in size:
        return 1280, 720
    w, h = size.lower().split("x", 1)
    try:
        return int(w), int(h)
    except Exception:
        return 1280, 720


def _best_image_size_for_video(w: int, h: int) -> str:
    """
    gpt-image-1 supported sizes:
    - 1024x1024
    - 1536x1024 (landscape)
    - 1024x1536 (portrait)
    - auto
    """
    if w >= h:
        return "1536x1024"
    return "1024x1536"


def generate_video_clip(
    client: OpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,     # kept for compatibility with app.py (ignored here)
    out_path: str,
) -> str:
    """
    Generates a scene clip WITHOUT OpenAI video endpoint.
    Flow:
      prompt -> OpenAI image -> ffmpeg Ken Burns motion -> mp4

    Includes:
    - Cache reuse (PNG + MP4)
    - Image retries + fallback to size="auto"
    - Consistent MP4 encoding for concat stability
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _ensure_dir(CACHE_DIR)

    # Final prompt should be stable (cache key depends on it)
    img_prompt = (
        prompt
        + "\n\nStyle notes: cinematic documentary b-roll, realistic lighting, "
          "photorealistic, no text overlays, no logos, no watermarks."
    )

    # ----------------------------
    # CACHE: reuse MP4 if already generated for same prompt/seconds/size
    # ----------------------------
    cache_id = _cache_key("gpt-image-1", img_prompt, str(seconds), size)
    cached_png = os.path.join(CACHE_DIR, f"{cache_id}.png")
    cached_mp4 = os.path.join(CACHE_DIR, f"{cache_id}.mp4")

    if os.path.exists(cached_mp4):
        if not os.path.exists(out_path):
            _copy_file(cached_mp4, out_path)
        # Optional: also copy cached PNG next to the clip for transparency/debugging
        png_path = out_path.replace(".mp4", ".png")
        if os.path.exists(cached_png) and not os.path.exists(png_path):
            _copy_file(cached_png, png_path)
        return out_path

    # If MP4 not cached but PNG is, we'll reuse it and only run ffmpeg
    png_path = out_path.replace(".mp4", ".png")

    W, H = _parse_size(size)
    img_size = _best_image_size_for_video(W, H)

    if os.path.exists(cached_png) and not os.path.exists(png_path):
        _copy_file(cached_png, png_path)

    if not os.path.exists(png_path):
        # ✅ Retry image generation (handles intermittent 429/5xx)
        img = None
        last_err = None
        for attempt in range(1, 4):
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size=img_size,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

        # ✅ Fallback to size="auto" if size-specific fails
        if img is None and last_err is not None:
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="auto",
                )
                last_err = None
            except Exception as e:
                raise RuntimeError(f"Image generation failed (gpt-image-1). {type(e).__name__}: {e}") from e

        # Decode b64
        try:
            b64 = img.data[0].b64_json
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            raise RuntimeError(f"Could not decode image bytes. {type(e).__name__}: {e}")

        # Save image next to clip + into cache
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        with open(cached_png, "wb") as f:
            f.write(img_bytes)

    # 2) Animate image into video (Ken Burns zoom/pan)
    ff = ffmpeg_exe()
    fps = 30
    frames = max(1, int(seconds * fps))

    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"zoompan=z='min(zoom+0.0008,1.08)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={W}x{H},"
        f"fps={fps},"
        f"format=yuv420p"
    )

    # ✅ Consistent encoding (reduces concat failures)
    run_cmd([
        ff, "-y",
        "-loop", "1",
        "-i", png_path,
        "-t", str(seconds),
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-g", "60",
        "-movflags", "+faststart",
        out_path
    ])

    # Save MP4 into cache
    _copy_file(out_path, cached_mp4)

    return out_path


# ----------------------------
# Packaging outputs
# ----------------------------

def zip_dir(dir_path: str, zip_path: str) -> str:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                z.write(full, rel)
    return zip_path
