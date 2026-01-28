# ============================
# PART 1/5 — Core Setup, Cache, ffmpeg Helpers, ZIP Discovery, Pairing
# ============================
# video_pipeline.py (Tier 1 optimized) — Shared helpers for UAPpress video apps
#
# Incorporates the new plan:
# ✅ Cache dir is read dynamically (Streamlit sets env after import sometimes)
# ✅ Cache pruning to prevent Streamlit disk bloat
# ✅ No “shake” (hard disabled)
# ✅ Keeps compatibility helpers used by app.py / video_creator.py
# ✅ Pairing supports Intro / Chapters / Outro
#
# NOTE: Subtitle functions still exist for compatibility, but the video creator app
# will not call them (CapCut will handle subtitles).

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
from typing import Dict, List, Optional, Tuple, Set

import imageio_ffmpeg
from openai import OpenAI


# ----------------------------
# Cache (Tier 1 optimized)
# ----------------------------
def _get_cache_dir() -> str:
    # DO NOT freeze cache dir at import time
    return os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _cache_key(*parts: str) -> str:
    s = "||".join([str(p).strip() for p in parts if p is not None])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _prune_cache(cache_dir: str) -> None:
    """
    Evict oldest files if cache grows beyond limit.
    Prevents Streamlit Cloud restarts/crashes due to disk bloat.

    Env:
      UAPPRESS_CACHE_MAX_MB (default 1200)
      UAPPRESS_CACHE_PRUNE_MIN_FREE_MB (default 150)
    """
    max_mb = _safe_int_env("UAPPRESS_CACHE_MAX_MB", 1200)
    min_free_mb = _safe_int_env("UAPPRESS_CACHE_PRUNE_MIN_FREE_MB", 150)
    if max_mb <= 0:
        return

    try:
        _ensure_dir(cache_dir)
        files: List[Tuple[float, int, str]] = []
        total = 0

        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                stt = os.stat(path)
                size = int(stt.st_size)
                mtime = float(stt.st_mtime)
            except Exception:
                continue
            total += size
            files.append((mtime, size, path))

        max_bytes = max_mb * 1024 * 1024
        if total <= max_bytes:
            return

        files.sort(key=lambda x: x[0])  # oldest first

        target = max(0, max_bytes - (min_free_mb * 1024 * 1024))
        for _, size, path in files:
            try:
                os.remove(path)
                total -= size
            except Exception:
                pass
            if total <= target:
                break
    except Exception:
        return


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
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:max_len] if s else "segment")


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


# ----------------------------
# Pairing helpers (Intro / Chapter N / Outro)
# ----------------------------
def _is_intro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("intro") or b == "intro" or " intro" in b or "_intro" in b or "-intro" in b


def _is_outro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("outro") or b == "outro" or " outro" in b or "_outro" in b or "-outro" in b


def _chapter_no_from_name(name: str) -> Optional[int]:
    base = os.path.splitext(os.path.basename(name))[0].lower()
    m = re.search(r"(?:chapter|ch)[\s_\-]*0*(\d{1,3})\b", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m2 = re.search(r"\b0*(\d{1,3})\b", base)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def segment_label(p: dict) -> str:
    if _is_intro_name(p.get("title_guess", "")) or _is_intro_name(p.get("script_path", "")) or _is_intro_name(p.get("audio_path", "")):
        return "INTRO"
    if _is_outro_name(p.get("title_guess", "")) or _is_outro_name(p.get("script_path", "")) or _is_outro_name(p.get("audio_path", "")):
        return "OUTRO"
    n = p.get("chapter_no")
    if n is not None:
        return f"CHAPTER {n}"
    return "SEGMENT"


def pair_segments(scripts: List[str], audios: List[str]) -> List[Dict]:
    """
    Creates pairs:
      - intro script ↔ intro audio
      - outro script ↔ outro audio
      - chapter_N script ↔ chapter_N audio
    Falls back to token overlap.
    """

    def norm_tokens(s: str) -> Set[str]:
        s = (s or "").lower()
        toks = set(re.split(r"[^a-z0-9]+", s))
        toks.discard("")
        return toks

    def token_score(sp: str, ap: str) -> int:
        return len(norm_tokens(sp).intersection(norm_tokens(ap)))

    intro_script = next((s for s in scripts if _is_intro_name(s)), None)
    outro_script = next((s for s in scripts if _is_outro_name(s)), None)
    intro_audio = next((a for a in audios if _is_intro_name(a)), None)
    outro_audio = next((a for a in audios if _is_outro_name(a)), None)

    scripts_left = [s for s in scripts if s not in {intro_script, outro_script}]
    audios_left = [a for a in audios if a not in {intro_audio, outro_audio}]

    audio_by_no: Dict[int, List[str]] = {}
    for a in audios_left:
        n = _chapter_no_from_name(a)
        if n is not None:
            audio_by_no.setdefault(n, []).append(a)

    used_audio = set()
    pairs: List[Dict] = []

    if intro_script and intro_audio:
        pairs.append(
            {
                "chapter_no": 0,
                "title_guess": os.path.splitext(os.path.basename(intro_script))[0],
                "script_path": intro_script,
                "audio_path": intro_audio,
            }
        )
        used_audio.add(intro_audio)

    if outro_script and outro_audio:
        pairs.append(
            {
                "chapter_no": 9998,
                "title_guess": os.path.splitext(os.path.basename(outro_script))[0],
                "script_path": outro_script,
                "audio_path": outro_audio,
            }
        )
        used_audio.add(outro_audio)

    for s in sorted(scripts_left):
        sn = _chapter_no_from_name(s)
        chosen = None

        if sn is not None and sn in audio_by_no:
            cands = [a for a in audio_by_no[sn] if a not in used_audio]
            if cands:
                chosen = max(cands, key=lambda a: token_score(s, a))

        if chosen is None:
            cands = [a for a in audios_left if a not in used_audio]
            if cands:
                chosen = max(cands, key=lambda a: token_score(s, a))

        if chosen:
            used_audio.add(chosen)
            pairs.append(
                {
                    "chapter_no": sn,
                    "title_guess": os.path.splitext(os.path.basename(s))[0],
                    "script_path": s,
                    "audio_path": chosen,
                }
            )

    def sort_key(p: Dict):
        n = p.get("chapter_no")
        if n == 0:
            return (-1, p["title_guess"].lower())
        if n == 9998:
            return (999999, p["title_guess"].lower())
        if n is None:
            return (500000, p["title_guess"].lower())
        return (int(n), p["title_guess"].lower())

    pairs.sort(key=sort_key)
    return pairs

# ============================
# PART 2/5 — Scene Planning (Text → JSON) + Transcription (Audio → SRT)
# ============================

# ----------------------------
# Scene planning (text -> JSON scenes)
# ----------------------------
SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration segment into a list of short visual scenes for AI generation. "
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
    """
    Uses Responses API to return a STRICT JSON list of scene objects.
    """
    payload = {
        "chapter_title": chapter_title,
        "max_scenes": int(max_scenes),
        "seconds_per_scene": int(seconds_per_scene),
        "chapter_text": chapter_text,
        "output": "STRICT_JSON_LIST_ONLY",
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    text = (resp.output_text or "").strip()

    # Defensive extraction if the model wraps JSON in extra text
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
    if m:
        text = m.group(1)

    try:
        scenes = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Scene planner did not return valid JSON. {type(e).__name__}: {e}\n\nRAW:\n{text[:2000]}")

    if not isinstance(scenes, list):
        raise RuntimeError("Scene planner returned JSON but it is not a list.")

    out: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        if not isinstance(sc, dict):
            continue
        out.append(
            {
                "scene": i,
                "seconds": int(sc.get("seconds", seconds_per_scene)),
                "prompt": str(sc.get("prompt", "")).strip(),
                "on_screen_text": (
                    str(sc.get("on_screen_text")).strip()
                    if sc.get("on_screen_text") is not None
                    else None
                ),
            }
        )

    # Sanity: remove empty prompts
    out = [s for s in out if s.get("prompt")]
    if not out:
        raise RuntimeError("Scene planner returned no usable prompts.")
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
    NOTE: For the new pipeline, you may not *use* SRTs (CapCut),
    but we keep this function for optional workflows + compatibility.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="srt",
            language=language,
        )

    # OpenAI SDK can return str or object depending on version
    if isinstance(tr, str):
        return tr
    if hasattr(tr, "text") and isinstance(tr.text, str):
        return tr.text
    return str(tr)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

# ============================
# PART 3/5 — SRT Utilities + ffmpeg Assembly (Concat / Mux / Subs / Re-encode)
# ============================
# NOTE:
# - Your new "Generate Videos" flow will NOT burn subtitles (CapCut).
# - We keep subtitle helpers here for compatibility and optional use.
# - In your new app/video_creator, you’ll simply avoid calling embed_srt_softsubs().

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
    return pattern.sub(repl, srt_text or "")


def renumber_srt_blocks(srt_text: str) -> str:
    blocks = re.split(r"\n\s*\n", (srt_text or "").strip(), flags=re.S)
    out_blocks = []
    n = 1
    for b in blocks:
        lines = b.strip().splitlines()
        if not lines:
            continue
        # Drop existing index line if present
        if re.fullmatch(r"\d+", (lines[0].strip() or "")):
            lines = lines[1:]
        out_blocks.append(str(n) + "\n" + "\n".join(lines))
        n += 1
    return "\n\n".join(out_blocks) + "\n"


# ----------------------------
# ffmpeg assembly: concat, mux, subtitles
# ----------------------------
def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    """
    Fast path: concat demuxer + stream copy.
    If inputs differ in codecs/timebase, caller should re-encode first.
    """
    if not mp4_paths:
        raise ValueError("concat_mp4s: no input mp4_paths")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        list_path = tf.name
        for p in mp4_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"concat_mp4s: missing input: {p}")
            # concat demuxer format
            tf.write(f"file '{p.replace(\"'\", \"'\\\\''\")}'\n")

    try:
        run_cmd([ff, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """
    Adds/overwrites audio onto the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"mux_audio: missing video_path: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"mux_audio: missing audio_path: {audio_path}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_cmd(
        [
            ff, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            out_path,
        ]
    )
    return out_path


def _escape_for_ffmpeg_filter(path: str) -> str:
    # needed for subtitles filter on Windows-like paths + colons
    return (path or "").replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_in_subtitles(video_path: str, srt_path: str, out_path: str) -> str:
    """
    Burns subtitles into video (always visible). Re-encodes.
    (Legacy / optional — your new workflow will avoid this.)
    """
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    srt_escaped = _escape_for_ffmpeg_filter(srt_path)

    run_cmd(
        [
            ff, "-y",
            "-i", video_path,
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:v", "libx264",
            "-preset", os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
            "-crf", os.environ.get("UAPPRESS_X264_CRF", "23"),
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path,
        ]
    )
    return out_path


def embed_srt_softsubs(video_path: str, srt_path: str, out_path: str) -> str:
    """
    Kept for app compatibility: historically this burned subs in.
    Your new video creator should NOT call this (CapCut subtitles).
    """
    return burn_in_subtitles(video_path, srt_path, out_path)


def reencode_mp4(in_path: str, out_path: str) -> str:
    """
    Safe re-encode for compatibility (use when concat-copy fails).
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"reencode_mp4: missing input: {in_path}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_cmd(
        [
            ff, "-y",
            "-i", in_path,
            "-c:v", "libx264",
            "-preset", os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
            "-crf", os.environ.get("UAPPRESS_X264_CRF", "23"),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path,
        ]
    )
    return out_path

# ============================
# PART 4/5 — Image → Ken Burns MP4 (No Shake) + Tier-1 Caching + Disk Safety
# ============================

def _parse_size(size: str) -> Tuple[int, int]:
    if "x" not in (size or ""):
        return 1280, 720
    w, h = str(size).lower().split("x", 1)
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
    return "1536x1024" if w >= h else "1024x1536"


def _build_motion_vf(W: int, H: int, fps: int, frames: int) -> str:
    """
    Continuous motion for full duration (fixes "stops early then static").

    - NO ROTATION / SHAKE (hard disabled)
    - Optional subtle grain (cheap, ffmpeg-only)

    Env knobs:
      UAPPRESS_ZOOM_START (default 1.00)
      UAPPRESS_ZOOM_END   (default 1.08)
      UAPPRESS_MOTION_GRAIN (0/1, default 1)
    """
    z0 = _safe_float_env("UAPPRESS_ZOOM_START", 1.00)
    z1 = _safe_float_env("UAPPRESS_ZOOM_END", 1.08)
    z0 = max(1.0, min(1.25, z0))
    z1 = max(z0, min(1.35, z1))

    grain = _safe_int_env("UAPPRESS_MOTION_GRAIN", 1) == 1

    # Smooth linear zoom over full frames using on (output frame index)
    zoom_expr = f"{z0}+({z1}-{z0})*on/max(1\\,{frames-1})"

    # Center crop tracking (no drift, no shake)
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={W}x{H},"
        f"fps={fps},"
        f"format=yuv420p"
    )

    if grain:
        vf += ",noise=alls=10:allf=t+u"

    return vf


def generate_video_clip(
    client: OpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,  # kept for compatibility with app.py (ignored here)
    out_path: str,
) -> str:
    """
    Generates a scene clip WITHOUT OpenAI video endpoint.
    Flow:
      prompt -> OpenAI image -> ffmpeg motion -> mp4

    Tier 1 optimizations:
    - Continuous motion for full duration (no static tail)
    - Hard-disable shake/rotation (zero drift)
    - Lower default FPS (env configurable) for speed
    - Faster x264 preset+crf+tune stillimage
    - Cache eviction to avoid Streamlit disk crashes
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cache_dir = _get_cache_dir()
    _ensure_dir(cache_dir)
    _prune_cache(cache_dir)

    # Stable prompt (cache key depends on it)
    img_prompt = (
        (prompt or "").strip()
        + "\n\nStyle notes: cinematic documentary b-roll, realistic lighting, photorealistic, "
          "no text overlays, no logos, no watermarks."
    )

    seconds = max(1, int(seconds))
    W, H = _parse_size(size)
    img_size = _best_image_size_for_video(W, H)

    # FPS defaults lower for speed (override with env)
    fps = _safe_int_env("UAPPRESS_KB_FPS", 15)  # default 15 (faster than 30)
    fps = max(10, min(30, fps))
    frames = max(1, int(seconds * fps))

    # ----------------------------
    # CACHE: reuse MP4 if already generated for same prompt/seconds/size/fps
    # ----------------------------
    cache_id = _cache_key("gpt-image-1", img_prompt, str(seconds), str(size), f"fps={fps}")
    cached_png = os.path.join(cache_dir, f"{cache_id}.png")
    cached_mp4 = os.path.join(cache_dir, f"{cache_id}.mp4")

    if os.path.exists(cached_mp4):
        if not os.path.exists(out_path):
            _copy_file(cached_mp4, out_path)
        # also copy png beside output for debugging, if present
        png_path = out_path.replace(".mp4", ".png")
        if os.path.exists(cached_png) and not os.path.exists(png_path):
            _copy_file(cached_png, png_path)
        return out_path

    # If MP4 not cached but PNG is cached, reuse it and only run ffmpeg
    png_path = out_path.replace(".mp4", ".png")
    if os.path.exists(cached_png) and not os.path.exists(png_path):
        _copy_file(cached_png, png_path)

    if not os.path.exists(png_path):
        # Retry image generation (handles intermittent 429/5xx)
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

        # Fallback to size="auto"
        if img is None:
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="auto",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Image generation failed (gpt-image-1). {type(e).__name__}: {e}"
                ) from e

        # Decode base64 image payload
        try:
            b64 = img.data[0].b64_json
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            raise RuntimeError(f"Could not decode image bytes. {type(e).__name__}: {e}") from e

        # Save image next to clip + into cache
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        with open(cached_png, "wb") as f:
            f.write(img_bytes)

    # Animate image into video
    ff = ffmpeg_exe()
    vf = _build_motion_vf(W, H, fps=fps, frames=frames)

    preset = os.environ.get("UAPPRESS_X264_PRESET", "veryfast")
    crf = os.environ.get("UAPPRESS_X264_CRF", "23")

    run_cmd(
        [
            ff, "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-loop", "1",
            "-i", png_path,
            "-t", str(seconds),
            "-vf", vf,
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", crf,
            "-tune", "stillimage",
            "-profile:v", "high",
            "-level", "4.1",
            "-pix_fmt", "yuv420p",
            "-g", str(max(30, fps * 2)),
            "-movflags", "+faststart",
            out_path,
        ]
    )

    # Save MP4 into cache + prune again
    _copy_file(out_path, cached_mp4)
    _prune_cache(cache_dir)

    return out_path

# ============================
# PART 5/5 — Packaging Outputs (ZIP) + (Optional) Cleanup Helper
# ============================

def zip_dir(dir_path: str, zip_path: str) -> str:
    """
    Zip a directory to a file path (used by some legacy flows).
    """
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                z.write(full, rel)
    return zip_path


def safe_rmtree(path: str) -> None:
    """
    Best-effort cleanup (used by the new sequential generator after each upload).
    Never throws — cleanup should not crash the run.
    """
    try:
        if not path:
            return
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn))
                    except Exception:
                        pass
            try:
                os.rmdir(path)
            except Exception:
                pass
        elif os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        return
