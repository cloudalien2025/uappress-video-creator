# SECTION 1 — Module Imports (single source of truth)
# video_pipeline.py — Shared helpers for UAPpress video apps
#
# NOTE: Consolidated imports here to avoid duplicate mid-file imports.
# This is a readability-only change (no behavior change).

from __future__ import annotations

import io
import os
import re
import json
import time
import zipfile
import tempfile
import subprocess
import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any

import imageio_ffmpeg
from openai import OpenAI


# SECTION 2 — Cache Management (Tier 1 optimized)
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


# SECTION 3 — Extensions / Discovery
SCRIPT_EXTS = {".txt", ".md", ".json"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".webm", ".flac"}


# SECTION 4 — OS / ffmpeg Helpers
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


# SECTION 5 — ZIP Extraction + File Discovery
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


# SECTION 6 — Segment Pairing Helpers (Intro / Chapters / Outro)
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


# SECTION 7 — Scene Planning (Text → JSON) (NO Whisper / NO SRT)
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


# SECTION 8 — Video Concatenation (clips → segment)
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
        run_cmd(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "+genpts",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


# SECTION 9 — Audio Muxing (no narration cutoffs)
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

    run_cmd(
        [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
            # Small pad to guard against timestamp rounding; does NOT chop audio.
            "-af",
            "apad=pad_dur=2",
            "-movflags",
            "+faststart",
            out_path,
        ]
    )
    return out_path


# SECTION 10 — Scene Duration Allocation (prevents audio cutoffs)
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


# SECTION 11 — Sora Brand Intro/Outro (Config)
SORA_DEFAULT_MODEL = os.getenv("SORA_MODEL", "sora-2-pro")
SORA_DEFAULT_SECONDS = os.getenv("SORA_SECONDS", "8")  # allowed: "4","8","12"
SORA_DEFAULT_SIZE = os.getenv("SORA_SIZE", "1280x720")  # allowed: see app docs
SORA_POLL_INTERVAL_S = float(os.getenv("SORA_POLL_INTERVAL_S", "2.5"))
SORA_POLL_TIMEOUT_S = float(os.getenv("SORA_POLL_TIMEOUT_S", "900"))  # 15 min


# SECTION 12 — Sora Brand Intro/Outro (Data Model)
@dataclass(frozen=True)
class BrandIntroOutroSpec:
    brand_name: str
    channel_or_series: str
    tagline: str
    episode_title: str  # kept for backward compatibility; ignored when global_mode=True
    visual_style: str = (
        "Radar/FLIR surveillance aesthetic, serious investigative tone. "
        "Monochrome/low-saturation, subtle scanlines, HUD overlays, gridlines, "
        "bearing ticks, altitude/velocity readouts, minimal telemetry numbers, "
        "soft glow, restrained film grain. Slow camera drift. "
        "No aliens, no monsters, no bright neon sci-fi, no cheesy explosions. "
        "Clean premium typography, stable and readable."
    )
    palette: str = "deep charcoal, soft white, muted amber accents"
    logo_text: Optional[str] = None
    intro_music_cue: str = "subtle systems hum, low synth bed, minimal, tense but restrained"
    outro_music_cue: str = "subtle systems hum, low synth bed, minimal, calm resolve"
    cta_line: str = "Subscribe for more investigations."
    sponsor_line: Optional[str] = None
    aspect: str = "landscape"  # "landscape" or "portrait"
    global_mode: bool = True   # if True, DO NOT include episode title in intro


# SECTION 13 — Sora Prompt Builders
def _sanitize_filename(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "clip"


def _normalize_seconds(v: Optional[Union[str, int]]) -> str:
    if v is None:
        return str(SORA_DEFAULT_SECONDS)
    s = str(v).strip()
    if s not in ("4", "8", "12"):
        return str(SORA_DEFAULT_SECONDS)
    return s


def _resolve_size(spec: BrandIntroOutroSpec, explicit_size: Optional[str] = None) -> str:
    if explicit_size:
        return explicit_size
    if spec.aspect.lower().startswith("p"):
        return "720x1280"
    return "1280x720"


def build_sora_brand_intro_prompt(spec: BrandIntroOutroSpec, *, seconds: str) -> str:
    logo_text = spec.logo_text or spec.brand_name

    lines = [
        "Create a short branded INTRO bumper for a serious investigative documentary YouTube channel.",
        f"Length: {seconds} seconds.",
        f"Style: {spec.visual_style}",
        f"Color palette: {spec.palette}",
        "",
        "Visual language requirements (Radar/FLIR HUD):",
        "- FLIR / radar scope UI feeling, telemetry overlays, HUD ticks, gridlines, bearing marks, minimal numbers.",
        "- Subtle scanlines, restrained grain, soft glow; slow drift (no fast cuts).",
        "- Serious tone: declassified / surveillance vibe. Avoid neon sci-fi and cheesy effects.",
        "",
        "On-screen text (clean, modern, readable, stable):",
        f"1) '{logo_text}' (primary)",
        f"2) '{spec.channel_or_series}' (secondary)",
    ]

    if not spec.global_mode:
        ep = (spec.episode_title or "").strip()
        if ep:
            lines.append(f"3) Episode title (briefly): '{ep}'")

    lines += [
        "",
        "Typography rules:",
        "- Keep text large, stable, and spelled correctly.",
        "- Minimal, premium broadcast typography; no gimmicky fonts.",
        "",
        "Audio:",
        f"- Add synced audio that matches: {spec.intro_music_cue}.",
        "- No voiceover.",
        "",
        "Deliver a polished broadcast-ready intro bumper.",
    ]
    return "\n".join(lines)


def build_sora_brand_outro_prompt(spec: BrandIntroOutroSpec, *, seconds: str) -> str:
    logo_text = spec.logo_text or spec.brand_name

    lines = [
        "Create a short branded OUTRO bumper for a serious investigative documentary YouTube channel.",
        f"Length: {seconds} seconds.",
        f"Style: {spec.visual_style}",
        f"Color palette: {spec.palette}",
        "",
        "Visual language requirements (Radar/FLIR HUD):",
        "- FLIR / radar scope UI feeling, telemetry overlays, HUD ticks, gridlines, bearing marks, minimal numbers.",
        "- Subtle scanlines, restrained grain, soft glow; slow drift; gentle fade to black at end.",
        "- Serious tone: avoid neon sci-fi, avoid cheesy effects.",
        "",
        "On-screen text (clean, modern, readable, stable):",
        f"1) '{logo_text}' (primary)",
        f"2) '{spec.cta_line}' (secondary)",
    ]
    if spec.sponsor_line:
        lines.append(f"3) Sponsor line (brief, tasteful): '{spec.sponsor_line}'")

    lines += [
        "",
        "Typography rules:",
        "- Keep text large, stable, and spelled correctly.",
        "- Minimal, premium broadcast typography.",
        "",
        "Audio:",
        f"- Add synced audio that matches: {spec.outro_music_cue}.",
        "- No voiceover.",
        "",
        "Deliver a polished broadcast-ready outro bumper.",
    ]
    return "\n".join(lines)


# SECTION 14 — Sora Job Lifecycle (create → poll → download)
def _sora_create_video_job(
    client,
    prompt: str,
    *,
    model: Optional[str] = None,
    seconds: Optional[Union[str, int]] = None,
    size: Optional[str] = None,
    input_reference_path: Optional[str] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "model": (model or SORA_DEFAULT_MODEL),
        "seconds": _normalize_seconds(seconds),
        "size": (size or SORA_DEFAULT_SIZE),
    }

    if input_reference_path:
        with open(input_reference_path, "rb") as f:
            kwargs["input_reference"] = f
            return client.videos.create(**kwargs)

    return client.videos.create(**kwargs)


def _sora_poll_until_done(client, video_id: str) -> Any:
    deadline = time.time() + SORA_POLL_TIMEOUT_S

    while time.time() < deadline:
        job = client.videos.retrieve(video_id)
        status = getattr(job, "status", None) or (job.get("status") if isinstance(job, dict) else None)

        if status in ("completed", "succeeded"):
            return job
        if status in ("failed", "canceled", "cancelled", "error"):
            raise RuntimeError(f"Sora video job {video_id} ended with status={status}")

        time.sleep(SORA_POLL_INTERVAL_S)

    raise TimeoutError(f"Sora video job {video_id} timed out after {SORA_POLL_TIMEOUT_S:.0f}s")


def _sora_download_mp4(client, video_id: str, out_path: Path) -> Path:
    """
    ✅ FIX (required):
    The OpenAI Python SDK has had variants where `client.videos.content(video_id)` is absent.
    We try the known method names in order for compatibility on Streamlit Cloud.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try common SDK shapes without assuming one exact method.
    content = None
    videos = getattr(client, "videos", None)

    if videos is not None:
        for meth_name in ("content", "download_content", "retrieve_content"):
            meth = getattr(videos, meth_name, None)
            if callable(meth):
                try:
                    content = meth(video_id)
                    break
                except AttributeError:
                    # Inline fix note: method exists on object in some SDKs but may raise; try next.
                    continue

    if content is None:
        raise AttributeError("OpenAI SDK client.videos has no supported content download method")

    # SDK may return bytes, a stream, or a response-like object depending on version.
    if isinstance(content, (bytes, bytearray)):
        out_path.write_bytes(content)
        return out_path

    read = getattr(content, "read", None)
    if callable(read):
        out_path.write_bytes(content.read())
        return out_path

    raw = getattr(content, "content", None)
    if isinstance(raw, (bytes, bytearray)):
        out_path.write_bytes(raw)
        return out_path

    raise TypeError("Unexpected return type from Sora content download method")


# SECTION 15 — Public API: Generate Sora Brand Intro/Outro Clips
def generate_sora_brand_intro_outro(
    client,
    spec: BrandIntroOutroSpec,
    output_dir: Union[str, Path],
    *,
    model: Optional[str] = None,
    # Backward-compatible single seconds/size:
    seconds: Optional[Union[str, int]] = None,
    size: Optional[str] = None,
    # NEW: separate controls
    intro_seconds: Optional[Union[str, int]] = None,
    outro_seconds: Optional[Union[str, int]] = None,
    intro_size: Optional[str] = None,
    outro_size: Optional[str] = None,
    intro_reference_image: Optional[str] = None,
    outro_reference_image: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Generates:
      - <brand>-intro.mp4
      - <brand>-outro.mp4

    Returns (intro_path, outro_path)

    Controls:
      - If intro_seconds/outro_seconds are provided, they take precedence.
      - Else falls back to `seconds`, else env default.
      - If intro_size/outro_size are provided, they take precedence.
      - Else falls back to `size`, else aspect-based default.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve seconds
    if intro_seconds is None:
        intro_seconds = seconds
    if outro_seconds is None:
        outro_seconds = seconds

    intro_s = _normalize_seconds(intro_seconds)
    outro_s = _normalize_seconds(outro_seconds)

    # Resolve sizes
    base_size = size  # may be None
    intro_sz = _resolve_size(spec, explicit_size=(intro_size or base_size))
    outro_sz = _resolve_size(spec, explicit_size=(outro_size or base_size))

    # --- Intro ---
    intro_prompt = build_sora_brand_intro_prompt(spec, seconds=intro_s)
    intro_job = _sora_create_video_job(
        client,
        intro_prompt,
        model=model,
        seconds=intro_s,
        size=intro_sz,
        input_reference_path=intro_reference_image,
    )
    intro_id = getattr(intro_job, "id", None) or (intro_job.get("id") if isinstance(intro_job, dict) else None)
    if not intro_id:
        raise RuntimeError("Sora intro job did not return an id")
    _sora_poll_until_done(client, intro_id)

    intro_name = f"{_sanitize_filename(spec.brand_name)}-intro.mp4"
    intro_path = out_dir / intro_name
    _sora_download_mp4(client, intro_id, intro_path)

    # --- Outro ---
    outro_prompt = build_sora_brand_outro_prompt(spec, seconds=outro_s)
    outro_job = _sora_create_video_job(
        client,
        outro_prompt,
        model=model,
        seconds=outro_s,
        size=outro_sz,
        input_reference_path=outro_reference_image,
    )
    outro_id = getattr(outro_job, "id", None) or (outro_job.get("id") if isinstance(outro_job, dict) else None)
    if not outro_id:
        raise RuntimeError("Sora outro job did not return an id")
    _sora_poll_until_done(client, outro_id)

    outro_name = f"{_sanitize_filename(spec.brand_name)}-outro.mp4"
    outro_path = out_dir / outro_name
    _sora_download_mp4(client, outro_id, outro_path)

    return intro_path, outro_path


# SECTION 16 — Final Assembly + Optional Sora GLOBAL Intro/Outro Injection
ENABLE_SORA_BRAND_CLIPS = os.getenv("ENABLE_SORA_BRAND_CLIPS", "0").strip() == "1"
SORA_BRAND_ASSETS_SUBDIR = os.getenv("SORA_BRAND_ASSETS_SUBDIR", "brand_assets")  # stable folder inside output_dir


def _run_cmd(cmd: Union[str, list]) -> None:
    p = subprocess.run(
        cmd if isinstance(cmd, list) else shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{p.stdout}")


def _ffmpeg_concat_videos_demuxer(
    video_paths: Tuple[Path, ...],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        list_file = Path(td) / "concat_list.txt"
        lines = [f"file '{p.resolve().as_posix()}'" for p in video_paths]
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = (
            f"ffmpeg -y -hide_banner -loglevel error "
            f"-f concat -safe 0 -i {list_file.as_posix()} "
            f"-c copy {out_path.as_posix()}"
        )
        _run_cmd(cmd)

    return out_path


def _sanitize_filename_local(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "brand"


def _global_brand_filenames(brand_name: str) -> Tuple[str, str]:
    slug = _sanitize_filename_local(brand_name)
    return (f"{slug}_GLOBAL_INTRO.mp4", f"{slug}_GLOBAL_OUTRO.mp4")


def _get_brand_assets_dir(output_dir: Union[str, Path]) -> Path:
    return Path(output_dir) / SORA_BRAND_ASSETS_SUBDIR


def _maybe_generate_or_reuse_global_brand_clips(
    client,
    *,
    output_dir: Union[str, Path],
    brand_spec,
    model: Optional[str] = None,
    intro_seconds: Optional[Union[str, int]] = None,
    outro_seconds: Optional[Union[str, int]] = None,
    size: Optional[str] = None,
    intro_size: Optional[str] = None,
    outro_size: Optional[str] = None,
    intro_reference_image: Optional[str] = None,
    outro_reference_image: Optional[str] = None,
    force_regen: bool = False,
) -> Tuple[Path, Path]:
    brand_dir = _get_brand_assets_dir(output_dir)
    brand_dir.mkdir(parents=True, exist_ok=True)

    intro_name, outro_name = _global_brand_filenames(getattr(brand_spec, "brand_name", "brand"))
    intro_path = brand_dir / intro_name
    outro_path = brand_dir / outro_name

    if (not force_regen) and intro_path.exists() and outro_path.exists():
        return intro_path, outro_path

    intro_tmp, outro_tmp = generate_sora_brand_intro_outro(
        client,
        brand_spec,
        brand_dir,
        model=model,
        size=size,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        intro_size=intro_size,
        outro_size=outro_size,
        intro_reference_image=intro_reference_image,
        outro_reference_image=outro_reference_image,
    )

    # Rename to GLOBAL stable filenames
    try:
        Path(intro_tmp).replace(intro_path)
    except Exception:
        intro_path.write_bytes(Path(intro_tmp).read_bytes())

    try:
        Path(outro_tmp).replace(outro_path)
    except Exception:
        outro_path.write_bytes(Path(outro_tmp).read_bytes())

    return intro_path, outro_path


def _maybe_add_sora_brand_intro_outro(
    client,
    *,
    main_video_path: Union[str, Path],
    output_dir: Union[str, Path],
    final_basename: str,
    brand_spec,
    enable: bool,
    model: Optional[str] = None,
    intro_seconds: Optional[Union[str, int]] = None,
    outro_seconds: Optional[Union[str, int]] = None,
    size: Optional[str] = None,
    intro_size: Optional[str] = None,
    outro_size: Optional[str] = None,
    intro_reference_image: Optional[str] = None,
    outro_reference_image: Optional[str] = None,
    force_regen_brand: bool = False,
) -> Path:
    main_video_path = Path(main_video_path)
    output_dir = Path(output_dir)

    if not enable:
        return main_video_path

    intro_path, outro_path = _maybe_generate_or_reuse_global_brand_clips(
        client,
        output_dir=output_dir,
        brand_spec=brand_spec,
        model=model,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        size=size,
        intro_size=intro_size,
        outro_size=outro_size,
        intro_reference_image=intro_reference_image,
        outro_reference_image=outro_reference_image,
        force_regen=force_regen_brand,
    )

    out_path = output_dir / f"{final_basename}.mp4"
    _ffmpeg_concat_videos_demuxer((intro_path, main_video_path, outro_path), out_path)
    return out_path


# SECTION 17 — Public Hook: finalize_video_output()
def finalize_video_output(
    client,
    *,
    main_video_path: Union[str, Path],
    output_dir: Union[str, Path],
    final_name: str,
    brand_spec=None,
    enable_brand_clips: Optional[bool] = None,
    model: Optional[str] = None,
    intro_seconds: Optional[Union[str, int]] = None,
    outro_seconds: Optional[Union[str, int]] = None,
    size: Optional[str] = None,
    intro_size: Optional[str] = None,
    outro_size: Optional[str] = None,
    intro_reference_image: Optional[str] = None,
    outro_reference_image: Optional[str] = None,
    force_regen_brand: bool = False,
) -> Path:
    """
    Drop-in finalizer you can call after your pipeline produces the main MP4.

    - Disabled by default: returns main_video_path unchanged.
    - If enabled and brand_spec provided: returns new MP4 with GLOBAL intro/outro.
    - GLOBAL assets cached in: output_dir/brand_assets/*_GLOBAL_INTRO/OUTRO.mp4
    """
    enable = ENABLE_SORA_BRAND_CLIPS if enable_brand_clips is None else bool(enable_brand_clips)

    if (not enable) or (brand_spec is None):
        return Path(main_video_path)

    return _maybe_add_sora_brand_intro_outro(
        client,
        main_video_path=main_video_path,
        output_dir=output_dir,
        final_basename=final_name,
        brand_spec=brand_spec,
        enable=True,
        model=model,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        size=size,
        intro_size=intro_size,
        outro_size=outro_size,
        intro_reference_image=intro_reference_image,
        outro_reference_image=outro_reference_image,
        force_regen_brand=force_regen_brand,
    )
    # ✅ FIX: removed unreachable duplicate return that made troubleshooting confusing.

# SECTION 18 — Segment MP4 Renderer (Images → MP4 + narration mux)
# ---------------------------------------------------------------
# FIX GOALS:
# - Generate scene images -> clips -> MP4 video track
# - Mux narration audio WITHOUT cutoffs
# - ✅ Keep video tightly matched to audio (video ~= audio + 2s)
# - ✅ Avoid SyntaxError landmines (no backslash-heavy f-strings)
# - ✅ Streamlit Cloud stability (disk cache + ultrafast)

import os
import base64
import math
from pathlib import Path
from typing import Dict, List
from openai import OpenAI


def _read_text_file(path: str, max_chars: int = 12000) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        txt = p.read_text(encoding="utf-8", errors="ignore")
        txt = (txt or "").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "\n..."
        return txt
    except Exception:
        return ""


def _chunk_script_for_scenes(script: str, max_scenes: int) -> List[str]:
    """
    Make up to max_scenes short chunks to drive image prompts.
    Simple + robust (no strict JSON dependency).
    """
    script = (script or "").strip()
    if not script:
        return []

    script = " ".join(script.split())
    max_scenes = max(1, int(max_scenes or 1))

    target = max(220, min(750, max(220, len(script) // max_scenes)))

    chunks: List[str] = []
    i = 0
    n = len(script)

    while i < n and len(chunks) < max_scenes:
        j = min(n, i + target)

        cut = script.rfind(". ", i, j)
        if cut == -1 or cut < i + 90:
            cut = script.rfind("; ", i, j)
        if cut == -1 or cut < i + 90:
            cut = j

        chunk = script[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        i = cut

    return chunks[:max_scenes]


def _image_size_for_video(width: int, height: int) -> str:
    """
    GPT Image sizes: 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait), or auto.
    Pick closest aspect.
    """
    width = int(width)
    height = int(height)
    return "1536x1024" if width >= height else "1024x1536"


def _generate_scene_image(
    *,
    client,
    prompt: str,
    out_path: str,
    size: str,
    model: str = "gpt-image-1.5",
) -> str:
    """
    Generate PNG and write to out_path.
    Disk cache: reuse if already created.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        if p.exists() and p.stat().st_size > 10_000:
            return str(p)
    except Exception:
        pass

    rsp = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size=size,
    )

    b64 = rsp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    p.write_bytes(img_bytes)
    return str(p)


def _make_scene_clip_from_still(
    *,
    image_path: str,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    zoom_strength: float,
) -> str:
    """
    Make a short mp4 clip from a still with zoompan.
    Disk cache: reuse if already created.
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        if outp.exists() and outp.stat().st_size > 50_000:
            return str(outp)
    except Exception:
        pass

    ff = ffmpeg_exe()

    seconds = float(seconds or 0)
    if seconds <= 0:
        seconds = 3.0

    fps = int(fps)
    width = int(width)
    height = int(height)

    zs = float(zoom_strength or 1.06)
    zs = max(1.0, min(1.25, zs))

    frames = max(1, int(round(seconds * fps)))
    inc = (zs - 1.0) / max(frames, 1)

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan=z='min(zoom+{inc:.8f},{zs:.4f})':"
        f"d={frames}:s={width}x{height}:fps={fps}"
    )

    run_cmd(
        [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-t",
            f"{seconds:.3f}",
            "-vf",
            vf,
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(outp),
        ]
    )
    return str(outp)


def _concat_scene_clips(*, clip_paths: List[str], out_path: str) -> str:
    """
    Concatenate MP4 clips using ffmpeg concat demuxer (re-encode safe).
    Uses safe quoting without backslash-heavy f-strings.
    """
    if not clip_paths:
        raise ValueError("_concat_scene_clips: no clip_paths")

    ff = ffmpeg_exe()
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    list_path = outp.with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # concat demuxer expects: file '...'
            safe_p = str(p).replace("'", "'\\''")
            f.write("file '" + safe_p + "'\n")

    try:
        run_cmd(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(outp),
            ]
        )
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass

    return str(outp)


def _clamp_scene_count_for_duration(
    *,
    target_seconds: int,
    desired_scenes: int,
    min_scene_seconds: int,
) -> int:
    """
    Prevent overshoot:
    If desired_scenes * min_scene_seconds > target_seconds,
    we must reduce scene count or video will outlast audio.
    """
    target_seconds = max(1, int(target_seconds))
    desired_scenes = max(1, int(desired_scenes))
    min_scene_seconds = max(1, int(min_scene_seconds))

    max_fit = max(1, target_seconds // min_scene_seconds)
    return max(1, min(desired_scenes, max_fit))


def render_segment_mp4(
    *,
    pair: Dict,
    extract_dir: str,
    out_path: str,
    api_key: str,
    fps: int,
    width: int,
    height: int,
    zoom_strength: float,
    max_scenes: int,
    min_scene_seconds: int,
    max_scene_seconds: int,
) -> str:
    """
    Real renderer:
    - Generate images per scene
    - Build silent video track
    - Mux narration audio (no cutoffs)
    - ✅ Duration match: video ~= ceil(audio) + 2 seconds
    """

    audio_path = str(pair.get("audio_path") or "").strip()
    if not audio_path:
        raise ValueError("render_segment_mp4: pair is missing audio_path")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"render_segment_mp4: missing audio_path: {audio_path}")

    script_path = str(pair.get("script_path") or "").strip()
    script_text = _read_text_file(script_path) if script_path else ""

    # Target duration = audio + 2s (tight match like you want)
    dur = float(get_media_duration_seconds(audio_path) or 0.0)
    if dur <= 0:
        dur = 5.0
    target_seconds = int(math.ceil(dur)) + 2
    target_seconds = max(3, target_seconds)

    max_scenes = max(1, min(60, int(max_scenes or 1)))

    chunks = _chunk_script_for_scenes(script_text, max_scenes=max_scenes)
    if not chunks:
        chunks = ["Serious investigative documentary visuals. Cinematic, restrained, realistic."]

    scene_count = _clamp_scene_count_for_duration(
        target_seconds=target_seconds,
        desired_scenes=len(chunks),
        min_scene_seconds=int(min_scene_seconds),
    )
    chunks = chunks[:scene_count]

    # Allocate seconds to sum to target_seconds (best effort within bounds)
    scene_secs_int = _allocate_scene_seconds(
        int(target_seconds),
        int(scene_count),
        min_scene=int(min_scene_seconds),
        max_scene=int(max_scene_seconds),
    )

    # Hard nudge to target if allocator couldn't hit it perfectly
    cur_sum = int(sum(scene_secs_int))
    if cur_sum > target_seconds:
        overshoot = cur_sum - target_seconds
        for i in range(len(scene_secs_int) - 1, -1, -1):
            if overshoot <= 0:
                break
            can_take = max(0, scene_secs_int[i] - int(min_scene_seconds))
            take = min(can_take, overshoot)
            scene_secs_int[i] -= take
            overshoot -= take
    elif cur_sum < target_seconds:
        need = target_seconds - cur_sum
        for i in range(len(scene_secs_int) - 1, -1, -1):
            if need <= 0:
                break
            room = max(0, int(max_scene_seconds) - scene_secs_int[i])
            add = min(room, need)
            scene_secs_int[i] += add
            need -= add

    scene_durs = [float(max(1, int(s))) for s in scene_secs_int]

    base_dir = Path(str(extract_dir or "."))
    tmp_dir = base_dir / "_tmp_video"
    img_dir = base_dir / "_scene_images"
    clip_dir = base_dir / "_scene_clips"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=str(api_key))
    img_model = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1.5")
    img_size = _image_size_for_video(int(width), int(height))

    style_prefix = (
        "Create a cinematic still frame for a serious investigative documentary about UAPs. "
        "Photorealistic, restrained, no cheesy sci-fi. "
        "Muted tones, subtle film grain, moody lighting, realistic environments. "
        "No text, no logos, no subtitles, no watermarks. "
        "No aliens or monsters. "
    )

    clip_paths: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        scene_prompt = style_prefix + "Scene idea: " + str(chunk)
        img_path = str(img_dir / f"scene_{i:03d}.png")
        clip_path = str(clip_dir / f"scene_{i:03d}.mp4")

        _generate_scene_image(
            client=client,
            prompt=scene_prompt,
            out_path=img_path,
            size=img_size,
            model=img_model,
        )

        _make_scene_clip_from_still(
            image_path=img_path,
            out_path=clip_path,
            width=int(width),
            height=int(height),
            fps=int(fps),
            seconds=float(scene_durs[i - 1]),
            zoom_strength=float(zoom_strength or 1.06),
        )

        clip_paths.append(clip_path)

    tmp_video = str(tmp_dir / ("tmp_" + safe_slug(os.path.basename(out_path)) + ".mp4"))
    _concat_scene_clips(clip_paths=clip_paths, out_path=tmp_video)

    mux_audio(tmp_video, audio_path, out_path)

    try:
        os.remove(tmp_video)
    except Exception:
        pass

    return out_path





