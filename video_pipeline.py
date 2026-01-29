# ============================
# PART 1/5 — Core Setup, Cache, ffmpeg Helpers, ZIP Discovery, Pairing
# ============================
# video_pipeline.py (Tier 1 optimized) — Shared helpers for UAPpress video apps
#
# ✅ Cache dir is read dynamically (Streamlit sets env after import sometimes)
# ✅ Cache pruning to prevent Streamlit disk bloat
# ✅ No “shake” (hard disabled — zoom-only handled later)
# ✅ Pairing supports Intro / Chapters / Outro
# ✅ PATCH: extract_zip_to_temp now accepts ZIP PATH (str) OR ZIP BYTES (bytes) safely

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
from typing import Dict, List, Optional, Tuple, Set, Union

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
def extract_zip_to_temp(zip_input: Union[str, bytes, bytearray]) -> Tuple[str, str]:
    """
    Returns (workdir, extract_dir)

    ✅ PATCH:
    Accepts either:
      - zip_input as ZIP FILE PATH (str)
      - zip_input as ZIP BYTES (bytes/bytearray) for legacy callers
    """
    workdir = tempfile.mkdtemp(prefix="uappress_video_")
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    # Path-based (new)
    if isinstance(zip_input, str):
        if not os.path.isfile(zip_input):
            raise FileNotFoundError(f"ZIP file not found: {zip_input}")
        with zipfile.ZipFile(zip_input, "r") as z:
            z.extractall(extract_dir)
        return workdir, extract_dir

    # Bytes-based (legacy)
    if isinstance(zip_input, (bytes, bytearray)):
        with zipfile.ZipFile(io.BytesIO(zip_input), "r") as z:
            z.extractall(extract_dir)
        return workdir, extract_dir

    raise TypeError(f"extract_zip_to_temp: unsupported input type: {type(zip_input)}")


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
# PART 2/5 — Scene Planning (Text → JSON)
#   (NO Whisper / NO SRT — CapCut handles captions)
# ============================

SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration segment into a list of short visual scenes for AI generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt. "
    "Style: cinematic documentary b-roll / reenactment vibes; realistic lighting; subtle motion notes only. "
    "Avoid brand names, copyrighted characters, celebrity likeness, and explicit violence/gore. "
    "Keep prompts concise (1–3 sentences)."
)

def _extract_json_list(text: str) -> str:
    t = (text or "").strip()
    # strip code fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)

    i = t.find("[")
    j = t.rfind("]")
    if i != -1 and j != -1 and j > i:
        return t[i : j + 1]
    return t

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
        "chapter_title": str(chapter_title or "").strip(),
        "max_scenes": int(max_scenes),
        "seconds_per_scene": int(seconds_per_scene),
        "chapter_text": str(chapter_text or "").strip(),
        "output": "STRICT_JSON_LIST_ONLY",
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    text = ""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        text = resp.output_text
    else:
        text = str(resp)

    raw = (text or "").strip()
    json_str = _extract_json_list(raw)

    try:
        scenes = json.loads(json_str)
    except Exception as e:
        raise RuntimeError(
            f"Scene planner did not return valid JSON. {type(e).__name__}: {e}\n\nRAW:\n{raw[:1200]}"
        )

    if not isinstance(scenes, list):
        raise RuntimeError("Scene planner returned JSON but it is not a list.")

    out: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        if not isinstance(sc, dict):
            continue

        prompt = str(sc.get("prompt", "")).strip()
        if not prompt:
            continue

        sec = sc.get("seconds", seconds_per_scene)
        try:
            sec_i = int(sec)
        except Exception:
            sec_i = int(seconds_per_scene)
        sec_i = max(1, sec_i)

        out.append({"scene": i, "seconds": sec_i, "prompt": prompt})

    if not out:
        raise RuntimeError("Scene planner returned no usable prompts.")
    return out

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

# ============================
# PART 3/5 — Segment Assembly (Scene → Clips → Concat → Audio)
# ============================

def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    """
    Concatenate MP4 clips using ffmpeg concat demuxer with stream copy.
    Assumes all MP4 clips were generated with consistent codec/settings.
    """
    if not mp4_paths:
        raise ValueError("concat_mp4s: no input mp4_paths")

    for p in mp4_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"concat_mp4s: missing input: {p}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # concat demuxer list file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        list_path = tf.name
        for p in mp4_paths:
            # Correct escaping for concat file format
            safe_p = p.replace("'", "'\\''")
            tf.write(f"file '{safe_p}'\n")

    try:
        run_cmd([
            ff, "-y",
            "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """
    Mux narration audio onto video WITHOUT cutting narration.

    ✅ Key fix: removes `-shortest` (which can truncate audio if video is slightly shorter).

    Behavior:
      - Copies video stream (no re-encode)
      - Encodes audio to AAC for broad compatibility
      - Adds a tiny audio pad to absorb timestamp edge cases (harmless if not needed)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    run_cmd([
        ff, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        # Small pad to guard against timestamp rounding; does NOT chop audio.
        "-af", "apad=pad_dur=2",
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path


def _allocate_scene_seconds(
    total_seconds: int,
    n_scenes: int,
    *,
    min_scene: int,
    max_scene: int,
) -> List[int]:
    """
    Distribute total_seconds across n_scenes with per-scene bounds.
    Returns a list (length n_scenes) that sums to total_seconds (best effort).

    This exists to prevent narration cutoffs by ensuring the SUM(scene_seconds)
    matches the target duration (narration duration, or padded narration duration).
    """
    total_seconds = max(1, int(total_seconds))
    n_scenes = max(1, int(n_scenes))
    min_scene = max(1, int(min_scene))
    max_scene = max(min_scene, int(max_scene))

    # Start equal split
    base = total_seconds // n_scenes
    secs = [base] * n_scenes

    # Remainder distribution
    rem = total_seconds - sum(secs)
    i = 0
    while rem > 0:
        secs[i % n_scenes] += 1
        rem -= 1
        i += 1

    # Cap at max_scene by redistributing overflow
    for i in range(n_scenes):
        if secs[i] > max_scene:
            overflow = secs[i] - max_scene
            secs[i] = max_scene
            j = 1
            while overflow > 0 and j <= n_scenes:
                k = (i + j) % n_scenes
                room = max_scene - secs[k]
                if room > 0:
                    add = min(room, overflow)
                    secs[k] += add
                    overflow -= add
                j += 1

    # Raise to min_scene by borrowing from largest
    for i in range(n_scenes):
        if secs[i] < min_scene:
            need = min_scene - secs[i]
            secs[i] = min_scene
            while need > 0:
                donor = max(range(n_scenes), key=lambda d: secs[d])
                if secs[donor] <= min_scene:
                    break
                take = min(need, secs[donor] - min_scene)
                secs[donor] -= take
                need -= take

    # Final nudge to match total_seconds if possible within bounds
    diff = total_seconds - sum(secs)
    if diff > 0:
        for i in range(n_scenes):
            if diff == 0:
                break
            if secs[i] < max_scene:
                secs[i] += 1
                diff -= 1
    elif diff < 0:
        diff = -diff
        for i in range(n_scenes):
            if diff == 0:
                break
            if secs[i] > min_scene:
                secs[i] -= 1
                diff -= 1

    return secs

# ============================
# PART 4/5 — Image → Ken Burns MP4 (ZOOM ONLY)
# ============================
# ✅ NO Streamlit UI here (NO st.columns / sliders / buttons)
# ✅ Zoom-only Ken Burns (no shake, no drift, no noise)
# ✅ Tier-1 caching (png + mp4) + cache pruning
# ✅ Robust image generation retries

def _parse_size(size: str) -> Tuple[int, int]:
    if "x" not in (size or ""):
        return 1280, 720
    w, h = str(size).lower().split("x", 1)
    try:
        return int(w), int(h)
    except Exception:
        return 1280, 720


def _best_image_size_for_video(w: int, h: int) -> str:
    # gpt-image-1 supported sizes: 1024x1024, 1536x1024, 1024x1536, auto
    return "1536x1024" if int(w) >= int(h) else "1024x1536"


def _build_motion_vf(W: int, H: int, fps: int, frames: int) -> str:
    """
    ZOOM ONLY.
    No shake, no drift, no noise.
    """
    z0 = _safe_float_env("UAPPRESS_ZOOM_START", 1.00)
    z1 = _safe_float_env("UAPPRESS_ZOOM_END", 1.08)

    z0 = max(1.0, min(1.25, float(z0)))
    z1 = max(z0, min(1.35, float(z1)))

    # smooth linear zoom over all frames using output frame index 'on'
    zoom_expr = f"{z0}+({z1}-{z0})*on/max(1\\,{max(1, frames-1)})"
    # dead-center crop tracking (no drift)
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    return (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={W}x{H},"
        f"fps={fps},"
        f"format=yuv420p"
    )


def generate_video_clip(
    client: OpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,   # kept for compatibility (ignored; we use gpt-image-1)
    out_path: str,
) -> str:
    """
    OpenAI image → zoom-only Ken Burns → MP4

    Caching:
      - stores {cache_id}.png and {cache_id}.mp4 inside UAPPRESS_CACHE_DIR
      - if cached mp4 exists, copies it to out_path (fast)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cache_dir = _get_cache_dir()
    _ensure_dir(cache_dir)
    _prune_cache(cache_dir)

    seconds = max(1, int(seconds))
    W, H = _parse_size(size)

    fps = _safe_int_env("UAPPRESS_KB_FPS", 15)
    fps = max(10, min(30, int(fps)))
    frames = max(1, int(seconds * fps))

    img_prompt = (
        (prompt or "").strip()
        + "\n\nCinematic documentary b-roll, realistic lighting, photorealistic, "
          "no text, no logos, no watermarks."
    )

    cache_id = _cache_key("gpt-image-1", img_prompt, str(seconds), str(size), f"fps={fps}")
    png_path = out_path.replace(".mp4", ".png")
    cached_png = os.path.join(cache_dir, f"{cache_id}.png")
    cached_mp4 = os.path.join(cache_dir, f"{cache_id}.mp4")

    # Fast path: cached mp4
    if os.path.exists(cached_mp4):
        if not os.path.exists(out_path):
            _copy_file(cached_mp4, out_path)
        if os.path.exists(cached_png) and not os.path.exists(png_path):
            _copy_file(cached_png, png_path)
        return out_path

    # If png cached, reuse it
    if os.path.exists(cached_png) and not os.path.exists(png_path):
        _copy_file(cached_png, png_path)

    # Generate png if needed
    if not os.path.exists(png_path):
        img = None
        last_err = None
        for attempt in range(1, 4):
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size=_best_image_size_for_video(W, H),
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

        if img is None:
            # final fallback: auto size
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

        try:
            img_bytes = base64.b64decode(img.data[0].b64_json)
        except Exception as e:
            raise RuntimeError(f"Could not decode image bytes. {type(e).__name__}: {e}") from e

        with open(png_path, "wb") as f:
            f.write(img_bytes)
        with open(cached_png, "wb") as f:
            f.write(img_bytes)

    ff = ffmpeg_exe()
    vf = _build_motion_vf(W, H, fps=fps, frames=frames)

    run_cmd([
        ff, "-y",
        "-hide_banner", "-loglevel", "error",
        "-loop", "1",
        "-i", png_path,
        "-t", str(seconds),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
        "-crf", os.environ.get("UAPPRESS_X264_CRF", "23"),
        "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-g", str(max(30, fps * 2)),
        "-movflags", "+faststart",
        out_path,
    ])

    _copy_file(out_path, cached_mp4)
    _prune_cache(cache_dir)

    return out_path

# ============================
# PART 5/5 — Segment Orchestrator (ONE SEGMENT → ONE MP4)
# Auto-fit to narration duration
# ============================

from typing import Dict, List, Optional
import os
import tempfile
import shutil

from openai import OpenAI

# REQUIRES (already defined earlier in video_pipeline.py):
# - get_media_duration_seconds()
# - plan_scenes()
# - generate_video_clip()
# - concat_mp4s()
# - mux_audio()
# - read_script_file()
# - _allocate_scene_seconds()


def render_segment_mp4(
    *,
    pair: Dict,
    extract_dir: str,   # kept for compatibility
    out_path: str,
    zoom_strength: float = 1.06,
    fps: int = 15,
    width: int = 1280,
    height: int = 720,
    max_scenes: int = 10,
    min_scene_seconds: int = 4,
    max_scene_seconds: int = 10,
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
) -> str:
    """
    Orchestrates:
      script → scene planning → auto-fit timing → image clips → concat → mux audio

    Auto-fit logic:
      - Measure narration duration from audio
      - Estimate scene count (capped)
      - Allocate per-scene seconds within min/max bounds
    """

    script_path = pair.get("script_path")
    audio_path = pair.get("audio_path")

    if not script_path or not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing script: {script_path}")
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing audio: {audio_path}")

    # OpenAI client
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # Read script
    chapter_text = read_script_file(script_path)
    chapter_title = pair.get("title_guess") or os.path.basename(script_path)

    # -------------------------------------------------
    # Auto-fit: narration duration
    # -------------------------------------------------
    narration_sec = int(round(get_media_duration_seconds(audio_path)))
    narration_sec = max(1, narration_sec)

    min_scene_seconds = max(1, int(min_scene_seconds))
    max_scene_seconds = max(min_scene_seconds, int(max_scene_seconds))
    max_scenes = max(1, int(max_scenes))

    target_avg = int((min_scene_seconds + max_scene_seconds) / 2)
    est_scenes = int(round(narration_sec / max(1, target_avg)))
    est_scenes = max(1, min(max_scenes, est_scenes))

    # -------------------------------------------------
    # Scene planning (LLM)
    # -------------------------------------------------
    planned = plan_scenes(
        client,
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        max_scenes=est_scenes,
        seconds_per_scene=target_avg,
        model=model,
    )

    planned = planned[:max_scenes]
    if not planned:
        raise RuntimeError("Scene planner returned no usable prompts.")

    # -------------------------------------------------
    # Allocate seconds per scene to match narration
    # -------------------------------------------------
    alloc = _allocate_scene_seconds(
        narration_sec,
        len(planned),
        min_scene=min_scene_seconds,
        max_scene=max_scene_seconds,
    )

    scenes: List[Dict] = []
    for i, sc in enumerate(planned):
        prompt = str(sc.get("prompt") or "").strip()
        if not prompt:
            continue
        scenes.append(
            {
                "scene": i + 1,
                "seconds": int(alloc[i]),
                "prompt": prompt,
            }
        )

    if not scenes:
        raise RuntimeError("No usable scenes after allocation.")

    # -------------------------------------------------
    # Ken Burns (zoom-only)
    # -------------------------------------------------
    os.environ["UAPPRESS_ZOOM_START"] = "1.00"
    os.environ["UAPPRESS_ZOOM_END"] = f"{max(1.01, min(1.20, float(zoom_strength))):.4f}"
    os.environ["UAPPRESS_KB_FPS"] = str(int(max(10, min(30, fps))))

    size = f"{int(width)}x{int(height)}"

    # Per-segment temp directory
    seg_work = tempfile.mkdtemp(prefix="uappress_seg_")
    clips: List[str] = []

    try:
        # -------------------------------------------------
        # Generate scene clips
        # -------------------------------------------------
        for sc in scenes:
            clip_path = os.path.join(
                seg_work, f"scene_{int(sc['scene']):02d}.mp4"
            )
            generate_video_clip(
                client,
                prompt=sc["prompt"],
                seconds=int(sc["seconds"]),
                size=size,
                model="ignored",
                out_path=clip_path,
            )
            clips.append(clip_path)

        if not clips:
            raise RuntimeError("No scene clips were generated.")

        # -------------------------------------------------
        # Concat → silent MP4
        # -------------------------------------------------
        silent_mp4 = os.path.join(seg_work, "segment_silent.mp4")
        concat_mp4s(clips, silent_mp4)

        # -------------------------------------------------
        # Mux narration audio
        # -------------------------------------------------
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        mux_audio(silent_mp4, audio_path, out_path)

        return out_path

    finally:
        # ✅ SAFE CLEANUP — no custom helper, no NameError
        shutil.rmtree(seg_work, ignore_errors=True)
