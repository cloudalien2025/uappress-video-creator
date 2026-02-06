# ============================================================
# video_pipeline.py — UAPpress Video Creator
#
# SECTION 1 — Core helpers + core render primitives
# SECTION 2 — app.py compatibility layer
#
# GOAL: Fully compatible with the provided app.py (no import/syntax errors)
#
# Compatibility targets (as used by app.py):
#   - extract_zip_to_temp(zip_bytes_or_path) -> (workdir, extract_dir)
#   - find_files(extract_dir) -> (scripts, audios)
#   - pair_segments(scripts, audios) -> list[dict] (pair objects)
#   - segment_label(pair) -> "INTRO" / "OUTRO" / "CHAPTER X" / "SEGMENT"
#   - safe_slug(text, max_len=...) -> str
#   - ffmpeg_exe() -> str
#   - run_cmd(cmd_list) -> None
#   - render_segment_mp4(pair=..., extract_dir=..., out_path=..., api_key=..., fps=...,
#                        width=..., height=..., zoom_strength=..., max_scenes=...,
#                        min_scene_seconds=..., max_scene_seconds=...) -> None
#
# Notes:
# - Optional burned-in subtitles supported; no logos; no stitching.
# - Pipeline order: images -> scene clips -> concatenate -> mux audio (no black video).
# - Audio-driven timing + end buffer ~0.5–0.75s (default 0.65s).
# - Optional branding functions/classes exist for app.py detection:
#     BrandIntroOutroSpec + generate_sora_brand_intro_outro(...)
# - 9:16 is supported via width/height passed in from app.py.
#
# IMPORTANT:
# - This file keeps external dependencies optional/guarded.
# - Image generation uses OpenAI Images API when available; falls back to local placeholders
#   if generation fails (to avoid crashing the whole pipeline).
#
# FIXES INCLUDED (per your issues):
#   1) ffprobe missing on Streamlit Cloud: duration detection now falls back to parsing ffmpeg output
#      if ffprobe isn't available.
#   2) Repeated images across segments: cache key collisions fixed by using a stable per-file hash
#      in pair['base_name'], so different segments never share the same image cache folder.
#   3) Black video symptom: placeholders are still possible if image generation fails, but:
#      - supported image sizes are used by default (gpt-image-1 friendly)
#      - placeholders are a bit brighter (not near-black) so failures are visually obvious
# ============================================================

from __future__ import annotations


import dataclasses
import io
import json
import math
import os
import hashlib
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Optional deps (guarded)
try:
    import imageio_ffmpeg  # type: ignore
except Exception:
    imageio_ffmpeg = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


# ----------------------------
# Constants + small utilities
# ----------------------------

_END_BUFFER_MIN = 0.50
_END_BUFFER_MAX = 0.75
_DEFAULT_END_BUFFER = 0.65

_DEFAULT_FPS = 30
# ZIP parsing defaults
_SCRIPT_EXTS = {".txt", ".md"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}


# Image generation defaults (override via env if desired)
# IMPORTANT: gpt-image-1 supports sizes like 1024x1024, 1536x1024, 1024x1536.
_DEFAULT_IMAGE_MODEL = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1")
_DEFAULT_IMAGE_SIZE_169 = os.environ.get("UAPPRESS_IMAGE_SIZE_169", "1536x1024")
_DEFAULT_IMAGE_SIZE_916 = os.environ.get("UAPPRESS_IMAGE_SIZE_916", "1024x1536")

# Cache version knob (bump if you ever want to invalidate all cached images)
_IMAGE_CACHE_ROOT = os.environ.get("UAPPRESS_IMAGE_CACHE_ROOT", "_image_cache_v2")


# ----------------------------
# Visual Bible (Positive-only, reusable across documentaries)
# ----------------------------
# This is a lightweight, always-on style guide. It does NOT ban motifs.
# It provides consistent cinematic identity so scenes feel intentional.
#
# Design goals:
# - Zero extra model calls (no token burn)
# - Short prompts (no giant constraint walls)
# - Reusable for any documentary topic
#
# If you want to tune house style later, change these strings only.

_VISUAL_BIBLE_STYLE = (
    "CINEMATIC VISUAL BIBLE: photorealistic documentary reenactment still. "
    "Natural exposure, realistic materials, believable set dressing. "
    "Restrained, tasteful color grade with subtle film grain. "
    "Period-appropriate details when implied by context. "
    "No on-image text, no captions, no subtitles, no logos, no watermarks."
)

# Concrete, neutral anchors that prevent generic / repetitive outputs.
# Anchors are POSITIVE (what to show), not bans (what to avoid).
_ANCHORS_BY_TYPE: dict[str, list[str]] = {
    "ESTABLISHING": [
        "a guarded base gate with signage and a small guard booth",
        "a quiet airfield perimeter fence line at dusk",
        "a two-lane road leading to a restricted facility, overcast sky",
        "an exterior hangar area with service lights and parked vehicles",
    ],
    "DOCUMENT": [
        "a tabletop evidence layout: folders, stamped paperwork, and a map with marked coordinates",
        "a close view of archival documents on a desk beside a notepad and pen",
        "a stack of reports and photographs spread under a desk lamp, shallow depth of field",
        "a file folder and an index card box on an archival table, documentary lighting",
    ],
    "PROCESS": [
        "a radar or operations room environment with period-appropriate equipment and dim practical lighting",
        "a logbook open to handwritten entries beside a radio handset",
        "a wall map with pins and string, an investigator’s hand marking a location",
        "a filing cabinet drawer open with labeled folders, calm institutional interior",
    ],
    "WITNESS": [
        "a witness perspective from behind, looking out toward a distant horizon",
        "hands holding a notebook with sketches and notes, no readable text",
        "a silhouette at a window, observing weather and light conditions outside",
        "a person’s hands adjusting a camera or binoculars on a table, documentary realism",
    ],
    "INTERIOR": [
        "a quiet institutional corridor with practical ceiling lights",
        "an office desk with a rotary phone, paperwork, and a desk lamp",
        "a hangar interior with tool carts and floor markings, grounded realism",
        "a storage room with crates and shelving, restrained lighting",
    ],
    "ATMOSPHERE": [
        "a night sky with moving clouds and faint ambient light, no defined object",
        "an empty road at night with a single streetlight and haze",
        "low clouds over flat terrain, calm and observational",
        "a distant airfield light line through mist, documentary b-roll mood",
    ],
}

# Shot variation wheel — cheap variety without token cost.
_SHOT_WHEEL: list[str] = [
    "CAMERA: wide establishing composition, eye-level camera, stable tripod framing.",
    "CAMERA: medium documentary framing, slight push-in, natural perspective.",
    "CAMERA: close detail composition, shallow depth of field, editorial lighting.",
    "CAMERA: wide with foreground framing (fence line, doorway, window frame), calm composition.",
    "CAMERA: medium from behind (observer POV), gentle lateral slide feel, grounded realism.",
    "CAMERA: close-up of objects (maps, papers, radios, instruments), clean practical lighting.",
]

# ----------------------------
# Script Loading (REQUIRED)
# ----------------------------
# ----------------------------
def load_segment_script(script_path: Union[str, Path]) -> str:
    """
    REQUIRED: Load narration script for a segment.

    Option A contract:
      - The extracted ZIP MUST contain a .txt script file for each segment.
      - Scripts may live in subfolders inside the ZIP. We therefore load using the
        exact script_path discovered during extraction, not by reconstructing a
        root-level '<segment_id>.txt' path.

    We hard-fail if missing or empty to prevent silent fallback visuals.
    """
    sp = Path(script_path)
    if not sp.exists():
        raise RuntimeError(
            f"Missing required script file: {sp.name}. "
            "The ZIP must include a .txt script file for each segment."
        )
    text = sp.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError(f"Script file is empty: {sp.name}")
    return text
def _sanitize_scene_context(text: str, max_chars: int = 380, **_ignored: Any) -> str:
    """Sanitize script text snippets before embedding into image prompts.

    - Normalizes whitespace
    - Enforces a hard length cap to prevent prompt bloat
    - Accepts extra kwargs defensively to prevent call-site drift (Streamlit reruns / older call sites)
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()

    try:
        mc = int(max_chars)
    except Exception:
        mc = 380
    if mc < 40:
        mc = 40
    if mc > 2000:
        mc = 2000

    if len(s) > mc:
        s = s[:mc].rstrip() + "…"
    return s


def clamp_float(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sanitize_filename(name: str, keep: str = r"[^a-zA-Z0-9._-]+") -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(keep, "", name)
    return name or "unnamed"


def safe_slug(text: str, max_len: int = 60) -> str:
    """
    app.py uses this to build file slugs.
    """
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        s = "untitled"
    return s[:max_len].strip("-") or "untitled"


def _stable_hash(s: str, n: int = 12) -> str:
    """
    Stable short hash for cache keys (prevents collisions across segments).
    """
    h = sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[: max(6, int(n))]


def ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def which_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return "ffmpeg"


def which_ffprobe() -> Optional[str]:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    # Streamlit Cloud often doesn't have ffprobe. Return None so we can fall back.
    return None


def run_cmd(cmd: List[str]) -> None:
    """
    app.py expects vp.run_cmd(...) for ffmpeg utility calls in Bonus exporter.
    """
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-4000:]
        raise RuntimeError(f"Command failed (code {proc.returncode}). Tail:\n{tail}")


def ffmpeg_exe() -> str:
    """
    app.py expects vp.ffmpeg_exe()
    """
    return which_ffmpeg()


def run_ffmpeg(cmd: List[str]) -> None:
    """
    Internal: run ffmpeg and raise readable error.
    """
    if "-y" not in cmd:
        cmd = cmd[:1] + ["-y"] + cmd[1:]
    run_cmd(cmd)


def _parse_duration_from_ffmpeg_stderr(stderr_text: str) -> float:
    """
    Parse: Duration: 00:01:23.45 from ffmpeg -i output.
    Returns seconds or 0.0 if not found.
    """
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text or "")
    if not m:
        return 0.0
    try:
        hh = float(m.group(1))
        mm = float(m.group(2))
        ss = float(m.group(3))
        return hh * 3600.0 + mm * 60.0 + ss
    except Exception:
        return 0.0


def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    """
    Return media duration seconds.

    Primary: ffprobe if installed.
    Fallback: ffmpeg -i <file> parse Duration line (works on Streamlit Cloud where ffprobe is missing).
    """
    p = str(path)

    ffprobe = which_ffprobe()
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                p,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            try:
                return float((proc.stdout or "").strip())
            except Exception:
                pass

    # Fallback: ffmpeg -i parse stderr
    ffmpeg = which_ffmpeg()
    proc2 = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", p],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc2.returncode != 0:
        # ffmpeg returns non-zero on -i without output often; stderr still contains Duration.
        pass
    return _parse_duration_from_ffmpeg_stderr(proc2.stderr or "")


def clamp_scene_seconds(value: float, min_scene_s: float, max_scene_s: float) -> float:
    """
    For Streamlit slider/session_state stability: clamp into [min_scene_s, max_scene_s]
    """
    min_scene_s = float(min_scene_s)
    max_scene_s = float(max_scene_s)
    if max_scene_s < min_scene_s:
        max_scene_s = min_scene_s
    return clamp_float(float(value), min_scene_s, max_scene_s)


# ----------------------------
# Branding API (app.py compat)
# ----------------------------



def _write_bytes_to_file(data: Any, out_path: Path) -> Path:
    """
    Writes bytes / bytearray / stream-like .read() to a file.
    """
    if isinstance(data, (bytes, bytearray)):
        out_path.write_bytes(bytes(data))
        return out_path
    if hasattr(data, "read"):
        out_path.write_bytes(data.read())
        return out_path
    raise RuntimeError("Download returned unsupported type (expected bytes or stream-like).")


def _download_openai_video_content(client: Any, content_id: str, out_path: Path) -> Path:
    """
    app.py wraps client.videos.content(video_id) to call the appropriate download.
    That wrapper returns bytes/stream in most SDK shapes. We write it to disk here.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = client.videos.content(str(content_id))
        return _write_bytes_to_file(data, out_path)
    except Exception:
        pass

    try:
        if hasattr(client.videos, "download_content"):
            data = client.videos.download_content(str(content_id))
            return _write_bytes_to_file(data, out_path)
    except Exception:
        pass

    raise RuntimeError("Unable to download Sora content with current OpenAI client shape.")




def extract_zip_to_temp(zip_bytes_or_path: Union[bytes, str, Path]) -> Tuple[str, str]:
    """
    app.py expects: workdir, extract_dir
    workdir: temp folder root
    extract_dir: extracted contents folder
    """
    workdir = tempfile.mkdtemp(prefix="uappress_vc_")
    extract_dir = str(Path(workdir) / "extracted")
    ensure_dir(extract_dir)

    if isinstance(zip_bytes_or_path, (str, Path)):
        data = Path(zip_bytes_or_path).read_bytes()
    else:
        data = bytes(zip_bytes_or_path)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)

    return (workdir, extract_dir)


def find_files(extract_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """
    app.py expects: scripts, audios (lists of absolute paths).
    """
    extract_dir = Path(extract_dir)
    scripts: List[str] = []
    audios: List[str] = []
    for p in extract_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _SCRIPT_EXTS:
            scripts.append(str(p))
        elif ext in _AUDIO_EXTS:
            audios.append(str(p))

    scripts.sort(key=lambda x: Path(x).name.lower())
    audios.sort(key=lambda x: Path(x).name.lower())
    return scripts, audios


def _guess_kind_from_name(name: str) -> str:
    n = (name or "").lower()
    if "intro" in n:
        return "INTRO"
    if "outro" in n:
        return "OUTRO"
    m = re.search(r"(chapter|ch)[\s_\-]*0*(\d+)", n)
    if m:
        return f"CHAPTER {int(m.group(2))}"
    m2 = re.search(r"\b0*(\d{1,2})\b", n)
    if m2 and ("chapter" in n or "ch_" in n or n.startswith("ch")):
        return f"CHAPTER {int(m2.group(1))}"
    return "SEGMENT"


def _guess_chapter_no(kind: str) -> Optional[int]:
    m = re.search(r"chapter\s+(\d+)", kind.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _read_text_preview(path: Union[str, Path], max_chars: int = 2400) -> str:
    p = Path(path)
    try:
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict) and "text" in obj:
                s = str(obj.get("text") or "")
            else:
                s = json.dumps(obj, ensure_ascii=False)
        else:
            s = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        s = ""
    s = (s or "").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def pair_segments(scripts: List[str], audios: List[str]) -> List[Dict[str, Any]]:
    """
    app.py expects a list of dicts (pair objects). We include keys:
      - script_path
      - audio_path
      - kind_guess (INTRO/OUTRO/CHAPTER X/SEGMENT)
      - chapter_no (optional int)
      - title_guess (best-effort)
      - base_name (for stable ids / cache; MUST be unique to prevent image reuse)
    """
    audio_by_stem: Dict[str, str] = {Path(a).stem.lower(): a for a in audios}
    pairs: List[Dict[str, Any]] = []
    used_audio: set[str] = set()

    for s in scripts:
        sp = Path(s)
        stem = sp.stem.lower()

        a_match = audio_by_stem.get(stem)

        if not a_match:
            best = None
            for a in audios:
                an = Path(a).stem.lower()
                if an == stem:
                    best = a
                    break
                if stem and (stem in an or an in stem):
                    best = a
                    break
            a_match = best or ""

        # Fallback pairing (prevents 'pair has no audio_path' runtime failures):
        # If we still have no match, or the match is already used, assign the next unused audio deterministically.
        if not a_match or (a_match in used_audio):
            for a in audios:
                if a not in used_audio:
                    a_match = a
                    break

        if a_match and a_match not in used_audio:
            used_audio.add(a_match)

        kind = _guess_kind_from_name(sp.name)
        chapter_no = _guess_chapter_no(kind)

        preview = _read_text_preview(sp, max_chars=1200)
        title_guess = ""
        for line in (preview or "").splitlines():
            raw = line.strip()
            if not raw:
                continue

            # Only treat explicit headings/metadata as a "title".
            # This prevents the first narration line from becoming the UI label.
            if raw.startswith("#"):
                t = raw.lstrip("#").strip()
                if t:
                    title_guess = t[:120]
                    break
            if raw.lower().startswith("title:"):
                t = raw.split(":", 1)[1].strip()
                if t:
                    title_guess = t[:120]
                    break

        # FIX: unique base_name to prevent cache collisions across segments.
        # Use absolute path + paired audio path (if any) for a stable unique hash.
        uniq_seed = f"script:{str(sp.resolve())}|audio:{str(Path(a_match).resolve()) if a_match else ''}"
        base_name = f"{stem}_{_stable_hash(uniq_seed, 12)}"

        pairs.append(
            {
                "script_path": str(sp),
                "audio_path": str(a_match),
                "kind_guess": kind,
                "chapter_no": chapter_no,
                "title_guess": title_guess,
                "base_name": base_name,
            }
        )

    # Also include "audio-only" files if any (rare) as segments
    script_stems = {Path(s).stem.lower() for s in scripts}
    for a in audios:
        ap = Path(a)
        if ap.stem.lower() in script_stems:
            continue
        kind = _guess_kind_from_name(ap.name)
        chapter_no = _guess_chapter_no(kind)

        uniq_seed = f"audio_only:{str(ap.resolve())}"
        base_name = f"{ap.stem.lower()}_{_stable_hash(uniq_seed, 12)}"

        pairs.append(
            {
                "script_path": "",
                "audio_path": str(ap),
                "kind_guess": kind,
                "chapter_no": chapter_no,
                "title_guess": "",
                "base_name": base_name,
            }
        )

    return pairs


def segment_label(pair: Dict[str, Any]) -> str:
    """
    app.py expects one of:
      INTRO / OUTRO / CHAPTER <n> / SEGMENT
    """
    kind = str(pair.get("kind_guess") or "").strip().upper()
    if kind.startswith("CHAPTER"):
        n = pair.get("chapter_no")
        if n is None:
            m = re.search(r"CHAPTER\s+(\d+)", kind)
            if m:
                n = int(m.group(1))
        if n is not None:
            return f"CHAPTER {int(n)}"
        return "CHAPTER"
    if kind in ("INTRO", "OUTRO"):
        return kind

    sp = str(pair.get("script_path") or "")
    ap = str(pair.get("audio_path") or "")
    combo = (sp + " " + ap).lower()
    if "intro" in combo:
        return "INTRO"
    if "outro" in combo:
        return "OUTRO"
    return "SEGMENT"


# ----------------------------
# Image generation (segment-scoped)
# ----------------------------

def _image_size_for_mode(width: int, height: int) -> str:
    if height > width:
        return _DEFAULT_IMAGE_SIZE_916
    return _DEFAULT_IMAGE_SIZE_169



def _make_scene_card_image(
    path: Path,
    width: int,
    height: int,
    *,
    headline: str,
    body: str = "",
    footer: str = "UAPpress",
) -> None:
    """Local, deterministic fallback 'scene card' image.

    Prevents blank/black videos when OpenAI image generation is unavailable.
    Output is documentary-styled and readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal valid PNG fallback if Pillow isn't available.
    if Image is None:
        # Write a simple binary PPM (P6) so ffmpeg can still read a full-size, non-black image
        # without requiring Pillow. (Avoids 'black screen' symptom when image gen fails.)
        w = max(640, int(width))
        h = max(360, int(height))
        bg = (245, 245, 242)  # light paper
        border = (30, 30, 30)

        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        pixels = bytearray()

        # Simple border box + solid background
        border_thick = max(6, min(18, int(min(w, h) * 0.02)))
        for y in range(h):
            for x in range(w):
                if (
                    x < border_thick
                    or x >= w - border_thick
                    or y < border_thick
                    or y >= h - border_thick
                ):
                    pixels.extend(border)
                else:
                    pixels.extend(bg)

        # Ensure extension is .ppm for clarity
        if path.suffix.lower() != ".ppm":
            path = path.with_suffix(".ppm")

        path.write_bytes(header + bytes(pixels))
        return

    w = max(640, int(width))
    h = max(360, int(height))

    # Light paper background (not black)
    img = Image.new("RGB", (w, h), color=(245, 245, 242))
    draw = ImageDraw.Draw(img)

    pad = 18
    draw.rectangle([(pad, pad), (w - pad, h - pad)], outline=(30, 30, 30), width=4)

    try:
        font_h = ImageFont.truetype("DejaVuSans.ttf", 46)
        font_b = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_f = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font_h = ImageFont.load_default()
        font_b = ImageFont.load_default()
        font_f = ImageFont.load_default()

    x = pad + 26
    y = pad + 26
    head = (headline or "").strip()
    if len(head) > 90:
        head = head[:87] + "..."
    draw.text((x, y), head, fill=(10, 10, 10), font=font_h)

    y_div = y + 72
    draw.line([(x, y_div), (w - pad - 26, y_div)], fill=(60, 60, 60), width=2)

    y_text = y_div + 24
    body_txt = (body or "").strip()
    if body_txt:
        max_chars = 95
        out_lines = []
        for para in body_txt.splitlines():
            para = para.strip()
            if not para:
                continue
            while len(para) > max_chars:
                cut = para.rfind(" ", 0, max_chars)
                if cut == -1:
                    cut = max_chars
                out_lines.append(para[:cut].strip())
                para = para[cut:].strip()
            if para:
                out_lines.append(para)

        out_lines = out_lines[:10]
        for ln in out_lines:
            draw.text((x, y_text), ln, fill=(20, 20, 20), font=font_b)
            y_text += 34

    footer_txt = (footer or "").strip()
    draw.text((x, h - pad - 42), footer_txt, fill=(80, 80, 80), font=font_f)

    img.save(path, format="PNG")



# ----------------------------
# Image generation utilities
# ----------------------------
def _openai_generate_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    """Generate a single photorealistic image via OpenAI Images API (prompt-only).

    Streamlit Cloud reliability rules:
    - Always pass api_key explicitly to the OpenAI client.
    - Do NOT pass response_format; default b64_json output avoids SDK/version mismatches.
    - Retry a couple times to ride out transient failures.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available in this environment.")

    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key (api_key is empty).")

    client = OpenAI(api_key=str(api_key))

    # Try a small fallback ladder for robustness.
    models_to_try = [
        _DEFAULT_IMAGE_MODEL,
        (os.environ.get("UAPPRESS_IMAGE_MODEL_FALLBACK", "") or "").strip(),
    ]
    models_to_try = [m for m in models_to_try if m]

    last_err: Exception | None = None

    for model in models_to_try:
        for attempt in range(1, 3):
            try:
                r = client.images.generate(
                    model=str(model),
                    prompt=str(prompt),
                    size=str(size),
                )
                # SDK returns b64_json by default for Images API.
                b64 = r.data[0].b64_json
                if not b64:
                    raise RuntimeError("Images API returned empty b64_json.")
                import base64
                return base64.b64decode(b64)
            except Exception as e:
                last_err = e
                # brief backoff for transient issues
                time.sleep(0.35 * attempt)

    raise RuntimeError(f"Images API failed after retries. Last error: {last_err}")

def _free_roll_scene_type(beat: str, j: int) -> str:
    """Cheap deterministic 'Visual Director' router.

    No extra model calls. Uses keyword scoring + rotation to choose a scene type.
    """
    b = (beat or "").lower()

    score: Dict[str, int] = {k: 0 for k in _FREE_ROLL_SCENE_TYPES.keys()}

    def add(scene_type: str, pts: int) -> None:
        if scene_type in score:
            score[scene_type] += int(pts)

    # Evidence / paperwork
    if re.search(r"\b(memo|memorandum|report|file|files|document|documents|declassified|classification|classified|telegram|teletype|dispatch|archive|archival|newspaper|headline|foia|record|records)\b", b):
        add("DOCUMENT", 6)
    if re.search(r"\b(interview|testimony|witness|witnesses|statement|said|recalled|reported|account)\b", b):
        add("WITNESS", 5)
    if re.search(r"\b(radar|scope|blip|atc|tower|controller|logbook|logs|transcript|tape|audio|frequency|comms|communications)\b", b):
        add("PROCESS", 6)

    # Places / environments
    if re.search(r"\b(base|airfield|runway|hangar|gate|perimeter|fence|checkpoint|security|guard|patrol)\b", b):
        add("ESTABLISHING", 4)
    if re.search(r"\b(corridor|office|briefing|conference|warehouse|storage|archive room|file room|hangar interior|barracks)\b", b):
        add("INTERIOR", 4)
    if re.search(r"\b(night|dusk|dawn|storm|cloud|fog|rain|wind|haze|desert|forest|field|coast|ocean|mountain)\b", b):
        add("ATMOSPHERE", 3)

    # If beat is very abstract, bias toward ATMOSPHERE/PROCESS
    if len(b) < 180:
        add("ATMOSPHERE", 1)
        add("PROCESS", 1)

    # Pick best-scoring; if tie/low, rotate deterministically.
    best_type = None
    best_val = -10
    for k, v in score.items():
        if v > best_val:
            best_val = v
            best_type = k

    if not best_type or best_val <= 0:
        keys = list(_FREE_ROLL_SCENE_TYPES.keys())
        return keys[j % len(keys)]

    return best_type


_FREE_ROLL_SCENE_TYPES: Dict[str, str] = {
    "ESTABLISHING": "an exterior establishing shot of the location (base, road, gate, airfield, landscape)",
    "DOCUMENT": "archival evidence on a desk (papers, folders, photographs, maps), documentary tabletop composition",
    "PROCESS": "investigation process visuals (radar room, logbooks, maps with markings, radios, filing systems)",
    "WITNESS": "a human perspective without identifiable faces (silhouette, hands, notebook, looking toward distance)",
    "INTERIOR": "institutional interiors (corridors, offices, hangar interior, storage rooms), grounded realism",
    "ATMOSPHERE": "neutral b-roll atmosphere (night sky, clouds, empty road, lights, terrain), restrained and calm",
}

# ----------------------------
# Editorial Beat Routing (Intent Layer)
# ----------------------------
# Zero-token, deterministic "why this shot" layer.
# This is NOT a ban-list. It is a positive editorial purpose signal used to
# shape what is shown, so visuals feel intentional and documentary-grade.

_EDITORIAL_BEATS: Dict[str, str] = {
    "ESTABLISH": "Orient the viewer with place, time-of-day, and institutional context.",
    "EVIDENCE": "Show credible artifacts that support claims: documents, photos, files, maps, records.",
    "PROCESS": "Show how information is produced or analyzed: radar scopes, logbooks, maps, radios, investigative work.",
    "HUMAN": "Show the human stake without identities: silhouettes, hands, notebooks, observers, quiet tension.",
    "DOUBT": "Show uncertainty and limits: missing pages, redactions, empty archives, conflicting notes, unresolved spaces.",
}

# Editorial anchors (positive, reusable). Used preferentially when beat is clear.
_ANCHORS_BY_BEAT: Dict[str, List[str]] = {
    "ESTABLISH": [
        "a guarded base gate with period-appropriate signage and a small guard booth",
        "a quiet airfield perimeter at dusk with service lights and distant hangars",
        "a rural two-lane road approaching a restricted facility under overcast skies",
        "a wide establishing view of an institutional compound boundary and access road",
    ],
    "EVIDENCE": [
        "a tabletop evidence layout: folders, stamped paperwork, and a map with marked coordinates",
        "archival documents spread on a desk under a practical lamp, with a notepad and pen nearby",
        "a close view of a file folder beside photographs and a clipped report, shallow depth of field",
        "a stack of reports and an index card box on an archival table, documentary lighting",
    ],
    "PROCESS": [
        "a radar or operations room environment with period-appropriate equipment and dim practical lighting",
        "a logbook open to handwritten entries beside a radio handset and frequency notes",
        "a wall map with pins and string, an investigator’s hand marking a location",
        "a filing cabinet drawer open with labeled folders in a calm institutional interior",
    ],
    "HUMAN": [
        "a witness perspective from behind, looking out toward distant lights, face not visible",
        "hands holding a notebook with handwritten notes, interior practical lighting",
        "a lone figure silhouette in an office doorway, quiet tension, no identifiable features",
        "a seated interviewer’s viewpoint across a table with papers and a recorder, faces out of frame",
    ],
    "DOUBT": [
        "a document with heavy redactions on a desk, clipped alongside an incomplete memo",
        "an empty archive room with open drawers and missing folders, subdued institutional lighting",
        "a torn or partially missing page on a table beside a logbook, unresolved mood",
        "a corkboard of conflicting notes and timelines, some gaps left blank, investigative atmosphere",
    ],
}

def _editorial_beat(beat: str, j: int) -> str:
    """Deterministic editorial intent router (no tokens, no extra model calls)."""
    b = (beat or "").lower()
    score: Dict[str, int] = {k: 0 for k in _EDITORIAL_BEATS.keys()}

    def add(k: str, pts: int) -> None:
        if k in score:
            score[k] += int(pts)

    # Evidence signals
    if re.search(r"\b(memo|memorandum|report|file|files|document|documents|declassified|classified|telegram|teletype|dispatch|archive|archival|newspaper|headline|foia|record|records|photograph|photo|photos|map|maps)\b", b):
        add("EVIDENCE", 6)

    # Process signals
    if re.search(r"\b(radar|scope|blip|atc|tower|controller|logbook|logs|transcript|tape|audio|frequency|comms|communications|analysis|analyze|investigation|investigators|timeline|triangulate)\b", b):
        add("PROCESS", 6)

    # Human signals
    if re.search(r"\b(witness|witnesses|testimony|interview|statement|recalled|reported|saw|heard|felt|fear|panic|confused|startled)\b", b):
        add("HUMAN", 5)

    # Doubt/uncertainty signals
    if re.search(r"\b(uncertain|unclear|unknown|disputed|contradict|contradiction|incomplete|missing|redacted|withheld|classified|unverified|rumor|speculation|alleged|cannot confirm|no record)\b", b):
        add("DOUBT", 5)

    # Establishing signals
    if re.search(r"\b(base|airfield|runway|hangar|gate|perimeter|fence|checkpoint|security|corridor|office|briefing|conference|warehouse|archive room|file room|barracks)\b", b):
        add("ESTABLISH", 3)

    # If beat is very short/abstract, bias to ESTABLISH/DOUBT.
    if len(b) < 180:
        add("ESTABLISH", 1)
        add("DOUBT", 1)

    best = max(score.items(), key=lambda kv: kv[1])
    if best[1] <= 0:
        # Rotate deterministically if nothing triggers
        keys = list(_EDITORIAL_BEATS.keys())
        return keys[j % len(keys)]
    return best[0]

_FREE_ROLL_SHOT_WHEEL: List[str] = _SHOT_WHEEL


def _build_scene_prompts_from_script(script_text: str, max_scenes: int, *, incident_hint: str = "") -> List[str]:
    """Build per-scene image prompts using a **Free Roll Visual Director** (single-pass, token-cheap).

    Implements:
    - Positive-only Visual Bible (consistent cinematic identity)
    - Concrete visual anchors (one clear noun/setting per scene)
    - Storyboard-card prompt structure (WHAT / CAMERA / STYLE)

    Does NOT implement:
    - No negative/ban lists
    - No multi-generation 'best-of'
    - No extra model calls
    """
    text = (script_text or "").strip()
    if not text:
        return []

    max_scenes_i = clamp_int(int(max_scenes), 1, 120)

    # Paragraph beats; avoid over-fragmentation
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    beats = [p for p in paras if len(p) >= 120] or paras
    if not beats:
        return []

    # Choose N beats evenly (stable across runs)
    if len(beats) <= max_scenes_i:
        idxs = list(range(len(beats)))
    else:
        step = len(beats) / float(max_scenes_i)
        idxs = [min(len(beats) - 1, int(i * step)) for i in range(max_scenes_i)]

    hint = (incident_hint or "").strip()
    hint_line = f"CONTEXT: {hint}." if hint else ""

    prompts: List[str] = []
    for j, idx in enumerate(idxs):
        beat = beats[idx]
        snippet = _sanitize_scene_context(beat, max_chars=320)

        scene_type = _free_roll_scene_type(beat, j)
        beat_kind = _editorial_beat(beat, j)
        beat_desc = _EDITORIAL_BEATS.get(beat_kind, "")

        # Prefer editorial anchors when available; fall back to scene-type anchors.
        anchors = _ANCHORS_BY_BEAT.get(beat_kind) or _ANCHORS_BY_TYPE.get(scene_type) or _ANCHORS_BY_TYPE.get("ATMOSPHERE") or []
        anchor = anchors[j % len(anchors)] if anchors else "a grounded documentary environment"

        camera = _SHOT_WHEEL[j % len(_SHOT_WHEEL)]

        # Storyboard card (short, intentional)
        what = f"WHAT: {anchor}."
        if snippet:
            what += f" (Inspired by: {snippet})"

        prompt = (
            "Create a single photorealistic still image for a long-form investigative documentary.\n"
            + (hint_line + "\n" if hint_line else "")
            + (f"EDITORIAL BEAT: {beat_kind} — {beat_desc}\n" if beat_kind else "")
            + what + "\n"
            + camera + "\n"
            + "STYLE: " + _VISUAL_BIBLE_STYLE + "\n"
            + "Deliver: coherent environment details, grounded realism, believable staging."
        )
        prompts.append(prompt)

    return prompts

def _segment_image_dir(extract_dir: Union[str, Path], pair: Dict[str, Any]) -> Path:
    """
    Segment-scoped image directory to prevent cross-segment reuse.
    base_name is already collision-safe (hashed) from pair_segments().
    """
    extract_dir = Path(extract_dir)
    base = safe_slug(str(pair.get("base_name") or "segment"), max_len=80)
    label = segment_label(pair).lower().replace(" ", "_")
    return ensure_dir(extract_dir / _IMAGE_CACHE_ROOT / f"{label}_{base}")


def _generate_segment_images(
    *,
    api_key: str,
    extract_dir: Union[str, Path],
    pair: Dict[str, Any],
    width: int,
    height: int,
    max_scenes: int,
) -> List[Path]:
    """Generate (or reuse cached) images for this segment.

    IMPORTANT CACHE BEHAVIOR (bugfix):
    - If SOME cached images exist, we **do not** stop early.
    - We return cached images only when we already have the target count.
    - Otherwise we generate the remaining scenes to reach the target count.

    This prevents the 'one scene stretched across the whole segment' failure mode.
    """
    img_dir = Path(_segment_image_dir(extract_dir, pair))
    img_dir.mkdir(parents=True, exist_ok=True)

    desired = clamp_int(int(max_scenes), 1, 120)
    size = _image_size_for_mode(int(width), int(height))

    def _is_valid_img(p: Path) -> bool:
        try:
            return p.is_file() and p.stat().st_size > 1024 and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".ppm")
        except Exception:
            return False

    existing = sorted([p for p in img_dir.glob("scene_*.*") if _is_valid_img(p)])

    # If we already have enough cached images, reuse them (deterministic order).
    if len(existing) >= desired:
        return existing[:desired]

    # Script must exist in the extracted ZIP as <segment_id>.txt (hard-fail if missing).
    script_path = str(pair.get("script_path") or pair.get("script_file") or pair.get("script") or "").strip()
    if not script_path or not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Missing script for segment. Expected a .txt for this segment inside the ZIP. Got: {script_path}"
        )

    script_text = Path(script_path).read_text(encoding="utf-8", errors="ignore")

    # Best-effort hint to keep scenes coherent (e.g., 'Roswell', 'Rendlesham', etc.)
    incident_hint = (pair.get("label") or pair.get("id") or "").strip()

    prompts = _build_scene_prompts_from_script(script_text, desired, incident_hint=incident_hint)
    if not prompts:
        # Extremely defensive fallback — should not happen if script_text exists.
        prompts = ["Photorealistic documentary still, grounded military/archival context, no text, no logos."] * desired

    # Generate missing scenes only.
    for i in range(len(existing), desired):
        prompt = prompts[min(i, len(prompts) - 1)]
        img_bytes = _openai_generate_image_bytes(
            api_key=api_key,
            prompt=prompt,
            size=size,
        )
        out_path = img_dir / f"scene_{i+1:02d}.png"
        _write_bytes_to_file(img_bytes, out_path)

    final = sorted([p for p in img_dir.glob("scene_*.*") if _is_valid_img(p)])
    # Ensure we return exactly 'desired' items when possible.
    return final[:desired]

def _ensure_even(x: int) -> int:
    x = int(x)
    return x if x % 2 == 0 else x - 1 if x > 1 else 2



def build_scene_clip_from_image(
    *,
    image_path: Union[str, Path],
    out_mp4_path: Union[str, Path],
    duration_s: float,
    width: int,
    height: int,
    fps: int,
    zoom_strength: float,
) -> Path:
    """
    Create a real H.264 MP4 scene clip from a still image using FFmpeg zoompan.

    - No black video: always renders frames from the image.
    - Deterministic duration: uses -t and a fixed frame count.
    - Streamlit Cloud safe: creates parent dirs and validates output file size.
    """
    img = Path(image_path)
    out = Path(out_mp4_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not img.exists():
        raise FileNotFoundError(str(img))

    duration = max(0.05, float(duration_s))
    fps_i = clamp_int(int(fps), 6, 60)
    w = _ensure_even(clamp_int(int(width), 320, 3840))
    h = _ensure_even(clamp_int(int(height), 320, 3840))

    zs = float(zoom_strength)
    if zs < 0:
        zs = 0.0
    if zs > 3.0:
        zs = 3.0

    frames = max(2, int(round(duration * fps_i)))
    sw = _ensure_even(int(w * 1.45))
    sh = _ensure_even(int(h * 1.45))

    # zoom_strength is interpreted as the FINAL max zoom factor (e.g., 1.01–1.06).
    zmax = clamp_float(float(zs), 1.0, 1.20)
    zexpr = f"min(1+(({zmax}-1)*on/{frames}),{zmax})"
    xexpr = "iw/2-(iw/zoom/2)"
    yexpr = "ih/2-(ih/zoom/2)"

    vf = (
        f"scale={sw}:{sh},"
        f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d={frames}:s={w}x{h}:fps={fps_i},"
        f"format=yuv420p"
    )

    ff = which_ffmpeg()
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        str(img),
        "-vf",
        vf,
        "-t",
        f"{duration:.3f}",
        "-r",
        str(fps_i),
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    run_ffmpeg(cmd)

    if (not out.exists()) or out.stat().st_size < 4096:
        raise RuntimeError("Scene clip render failed (output missing or too small).")

    return out

def concat_video_clips(clip_paths: Sequence[Union[str, Path]], out_mp4_path: Union[str, Path]) -> Path:
    """
    Concatenate scene clips in order.

    We re-encode into a consistent H.264 stream to avoid concat/copy edge cases
    across ffmpeg builds (Streamlit Cloud).
    """
    out = Path(out_mp4_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = [Path(p) for p in clip_paths]
    clips = [c for c in clips if c.exists() and c.stat().st_size > 1024]
    if not clips:
        raise RuntimeError("No valid scene clips to concatenate.")

    list_file = out.parent / f"_{out.stem}_concat_list.txt"
    def _esc(p: Path) -> str:
        # ffmpeg concat demuxer single-quote escaping: ' -> '\''
        return str(p).replace("'", "'\\''")
    list_file.write_text("".join(["file '" + _esc(c) + "'\n" for c in clips]), encoding="utf-8")

    ff = which_ffmpeg()
    cmd = [
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
        str(list_file),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    run_ffmpeg(cmd)

    try:
        list_file.unlink()
    except Exception:
        pass

    if not out.exists() or out.stat().st_size < 2048:
        raise RuntimeError("Concatenation failed (output missing or too small).")

    return out


def mux_audio_to_video(
    *,
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    out_mp4_path: Union[str, Path],
    end_buffer_s: float = _DEFAULT_END_BUFFER,
    audio_seconds: Optional[float] = None,
) -> Path:
    """
    Mux audio onto the concatenated video.

    Requirements:
    - Keep the pipeline order (mux is the last step).
    - Apply a small end buffer (0.5–0.75s) deterministically.
    - Avoid black frames (tpad clones last frame for buffer).
    """
    v = Path(video_path)
    a = Path(audio_path)
    out = Path(out_mp4_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not v.exists():
        raise FileNotFoundError(str(v))
    if not a.exists():
        raise FileNotFoundError(str(a))

    buf = clamp_float(float(end_buffer_s or _DEFAULT_END_BUFFER), _END_BUFFER_MIN, _END_BUFFER_MAX)
    target_dur = None
    try:
        if audio_seconds is not None:
            ad = float(audio_seconds)
            if ad > 0:
                target_dur = ad + buf
    except Exception:
        target_dur = None

    ff = which_ffmpeg()
    # tpad extends video by cloning last frame; apad extends audio with silence for exactly buf seconds.
    filt = f"[0:v]tpad=stop_mode=clone:stop_duration={buf:.3f}[v];[1:a]apad=pad_dur={buf:.3f}[a]"
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(v),
        "-i",
        str(a),
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        "-movflags",
        "+faststart",
        str(out),
    ]
    run_ffmpeg(cmd)

    if not out.exists() or out.stat().st_size < 2048:
        raise RuntimeError("Mux failed (output missing or too small).")

    return out



def validate_mp4(path: str, *, require_audio: bool = True, min_bytes: int = 50_000) -> Tuple[bool, str]:
    """Validate an MP4 for upload gating.

    Checks:
    - exists and non-trivial size
    - duration > 0
    - contains a video stream
    - contains an audio stream if require_audio=True
    """
    try:
        p = Path(path)
        if not p.exists():
            return False, "missing"
        if p.stat().st_size < int(min_bytes):
            return False, f"too_small<{min_bytes}B"

        # Duration (ffprobe if available; otherwise ffmpeg -i parse)
        dur = 0.0
        try:
            dur = float(ffprobe_duration_seconds(p))
        except Exception:
            dur = 0.0
        if dur <= 0.05:
            return False, "zero_duration"

        ff = which_ffmpeg()
        proc = subprocess.run(
            [ff, "-hide_banner", "-i", str(p)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        info = (proc.stderr or "") + "\n" + (proc.stdout or "")
        has_video = "Video:" in info
        has_audio = "Audio:" in info

        if not has_video:
            return False, "no_video_stream"
        if require_audio and (not has_audio):
            return False, "no_audio_stream"

        return True, f"ok_dur={dur:.2f}s"
    except Exception as e:
        return False, f"error:{type(e).__name__}:{e}"



# ----------------------------
# Subtitles (Burn-in) — helpers
# ----------------------------

def _normalize_caption_text(t: str) -> str:
    """Normalize text for subtitle legibility + libass wrapping.

    Key fix for 'cutoff' in Shorts:
    - libass will NOT wrap long tokens that contain no spaces (e.g., 'stories—after').
      We normalize dash-like characters to include spaces so line breaks can occur.
    """
    s = (t or "").strip()
    if not s:
        return ""
    # Normalize dash variants to spaced em dash
    s = s.replace("\u2014", " — ").replace("\u2013", " — ").replace("\u2212", " — ")
    # Common punctuation spacing to encourage wraps
    s = s.replace("/", " / ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_text_for_captions(text: str, *, max_words: int = 6, max_chars: int = 44) -> List[str]:
    """Split narration text into short, readable caption chunks.

    Heuristic (no extra model calls). Optimized for Shorts readability.

    IMPORTANT:
    - We normalize dash characters to include spaces, otherwise libass may not wrap
      and text can get clipped off-screen (e.g., 'stories—after').
    """
    t = _normalize_caption_text(text)
    if not t:
        return []

    # Sentence-ish split
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    chunks: List[str] = []

    for sent in parts:
        s = sent.strip()
        if not s:
            continue

        # Break long sentences by commas/semicolons first
        subparts = re.split(r"(?<=[,;:])\s+", s)
        for sp in subparts:
            sp = sp.strip()
            if not sp:
                continue

            words = sp.split()
            if not words:
                continue

            cur: List[str] = []
            for w in words:
                cur.append(w)
                candidate = " ".join(cur)
                if len(cur) >= int(max_words) or len(candidate) >= int(max_chars):
                    chunks.append(candidate.strip())
                    cur = []
            if cur:
                chunks.append(" ".join(cur).strip())

    # Final cleanup: enforce max_chars by hard wrap at spaces (fallback to hard split)
    final: List[str] = []
    mc = int(max_chars)
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        if len(c) <= mc:
            final.append(c)
            continue

        # Prefer splitting at spaces
        while len(c) > mc:
            cut = c.rfind(" ", 0, mc)
            if cut <= 0:
                cut = mc
            final.append(c[:cut].strip())
            c = c[cut:].strip()
        if c:
            final.append(c)

    return final

def _format_srt_time(seconds: float) -> str:
    s = max(0.0, float(seconds))
    ms = int(round((s - int(s)) * 1000.0))
    total = int(s)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _wrap_caption_lines(c: str, *, max_line_chars: int, max_lines: int = 2) -> str:
    """Insert line breaks so captions stay inside safe margins.

    libass wrapping is space-dependent; we proactively insert '\n' to:
    - keep lines short (especially for 9:16 big subtitles)
    - avoid right-edge clipping on long tokens / hyphenated phrases
    """
    s = _normalize_caption_text(c)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    mlc = max(10, int(max_line_chars))
    ml = max(1, int(max_lines))

    # Already short enough
    if len(s) <= mlc:
        return s

    lines: List[str] = []
    remaining = s

    while remaining and len(lines) < ml:
        if len(remaining) <= mlc:
            lines.append(remaining.strip())
            remaining = ""
            break

        # Prefer break at last space within limit
        cut = remaining.rfind(" ", 0, mlc + 1)
        if cut <= 0:
            # Try break at em dash (now spaced) or punctuation
            dash = remaining.find(" — ")
            if 0 < dash < (mlc + 1):
                cut = dash + 1
            else:
                cut = mlc

        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    # If we still have remaining text, append to last line with ellipsis
    if remaining:
        if lines:
            lines[-1] = (lines[-1] + " " + remaining).strip()
        else:
            lines = [remaining.strip()]

    # Final safety: if last line is still too long, truncate a bit
    if lines and len(lines[-1]) > mlc * 2:
        lines[-1] = lines[-1][: (mlc * 2 - 1)].rstrip() + "…"

    return "\n".join(lines)


def build_srt_from_script(
    script_text: str,
    *,
    total_seconds: float,
    min_caption_s: float = 0.85,
    max_caption_s: float = 2.80,
    max_words: int = 6,
    max_chars: int = 44,
    max_line_chars: Optional[int] = None,
    max_lines: int = 2,
) -> str:
    """Create SRT text by distributing caption chunks across total_seconds."""
    chunks = _split_text_for_captions(script_text, max_words=int(max_words), max_chars=int(max_chars))
    if not chunks:
        return ""

    total_seconds = max(0.5, float(total_seconds))
    n = len(chunks)

    # Weight by word count (better pacing than equal slices)
    weights = []
    for c in chunks:
        w = max(1, len(c.split()))
        weights.append(w)
    wsum = float(sum(weights))

    # Raw durations
    durs = [(total_seconds * (w / wsum)) for w in weights]

    # Clamp + renormalize to keep within total
    min_s = float(min_caption_s)
    max_s = float(max_caption_s)
    durs = [clamp_float(d, min_s, max_s) for d in durs]

    # Renormalize to fit total_seconds
    scale = total_seconds / max(0.001, sum(durs))
    durs = [d * scale for d in durs]

    # Line wrapping settings
    mlc = int(max_line_chars) if max_line_chars is not None else int(max_chars)

    # Build SRT entries
    lines: List[str] = []
    t = 0.0
    for i, (c, d) in enumerate(zip(chunks, durs), start=1):
        start = t
        end = t + max(0.30, float(d))
        # Prevent overshoot
        if end > total_seconds:
            end = total_seconds
        if end - start < 0.25:
            end = min(total_seconds, start + 0.25)

        cap = _wrap_caption_lines(c, max_line_chars=mlc, max_lines=int(max_lines))

        lines.append(str(i))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(cap)
        lines.append("")  # blank line
        t = end

        if t >= total_seconds - 0.02:
            break

    return "\n".join(lines).strip() + "\n"

def _ffmpeg_filter_escape(path: str) -> str:
    """Escape a filesystem path for ffmpeg filter arguments."""
    p = str(path)
    p = p.replace("\\", "\\\\")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p


def burn_subtitles_to_mp4(
    *,
    src_mp4: Union[str, Path],
    srt_path: Union[str, Path],
    dst_mp4: Union[str, Path],
    style: str = "auto",
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Burn subtitles into an MP4 using ffmpeg+libass.

    style:
      - 'auto' (Shorts style for vertical, Standard for horizontal)
      - 'shorts' (big, center-ish)
      - 'standard' (bottom)
    """
    src = Path(src_mp4)
    srt = Path(srt_path)
    dst = Path(dst_mp4)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(str(src))
    if not srt.exists():
        raise FileNotFoundError(str(srt))

    h = int(height)
    w = int(width)
    is_vertical = h > w

    stl = (style or "auto").strip().lower()
    if stl == "auto":
        stl = "shorts" if is_vertical else "standard"
    if stl.startswith("short"):
        stl = "shorts"
    if stl not in ("shorts", "standard"):
        stl = "standard"
    # ASS force_style values:
    # Alignment: 2 = bottom center, 5 = middle center
    # WrapStyle=2 enables smart wrapping; MarginL/R create safe area.
    # For 9:16 big subtitles, we keep font size conservative to prevent edge clipping.
    if stl == "shorts":
        # Vertical outputs: reduce font a bit and force tighter safe margins.
        # Height-based sizing keeps 720x1280 readable without clipping.
        fs = 26 if int(height) >= 1600 else 22
        force = (
            f"FontName=DejaVu Sans,FontSize={fs},Outline=2,Shadow=1,"
            "Alignment=5,WrapStyle=2,MarginV=200,MarginL=80,MarginR=80"
        )
    else:
        fs = 30 if int(height) >= 900 else 26
        force = (
            f"FontName=DejaVu Sans,FontSize={fs},Outline=2,Shadow=1,"
            "Alignment=2,WrapStyle=2,MarginV=70,MarginL=60,MarginR=60"
        )

    vf = f"subtitles='{_ffmpeg_filter_escape(str(srt))}':force_style='{force}'"

    ff = which_ffmpeg()
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    run_ffmpeg(cmd)

    if not dst.exists() or dst.stat().st_size < 4096:
        raise RuntimeError("Subtitle burn failed (output missing or too small).")
    return dst


# ----------------------------
# The app.py-compatible segment renderer
# ----------------------------

def render_segment_mp4(
    *,
    pair: Dict[str, Any],
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
    burn_subtitles: bool = False,
    subtitle_style: str = "Auto",
    export_srt: bool = False,
) -> None:
    """Render one segment MP4 (images → scene clips → concatenate → mux audio).

    LOCKED ORDER:
      images → scene clips → concatenate → mux audio

    TIMING (STRICT):
      - Audio-driven scene timing (total video follows audio)
      - End buffer strictly 0.50–0.75s (default 0.65s) applied in mux step
      - min_scene_seconds is guidance for scene count selection only (never forces video longer than audio)
    """
    extract_dir_p = Path(extract_dir)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    audio_path = str(pair.get("audio_path") or pair.get("audio_file") or pair.get("audio") or "").strip()
    if not audio_path:
        raise RuntimeError("pair has no audio_path; cannot render segment.")
    audio_p = Path(audio_path)
    if not audio_p.exists():
        raise FileNotFoundError(str(audio_p))

    audio_seconds = float(ffprobe_duration_seconds(audio_p) or 0.0)

    # Scene pacing parameters (min is guidance only; max used to avoid overly-long scenes)
    min_s = clamp_int(int(min_scene_seconds), 1, 600)
    max_s = clamp_int(int(max_scene_seconds), min_s, 600)
    max_scenes_i = clamp_int(int(max_scenes), 1, 120)

    # Choose scene count (no token cost): keep per-scene near mid, but guarantee per-scene <= max_s when possible.
    if audio_seconds <= 0.0:
        target_scenes = clamp_int(max_scenes_i, 3, max_scenes_i)
    else:
        mid = (float(min_s) + float(max_s)) / 2.0
        approx = int(max(1, round(audio_seconds / max(1.0, mid))))

        lower = int(max(1, math.ceil(audio_seconds / float(max_s))))  # ensures per-scene <= max_s when feasible
        target_scenes = clamp_int(approx, lower, max_scenes_i)

        # If guidance-min would imply fewer scenes, allow fewer (but never below 1),
        # WITHOUT enforcing longer-than-audio total duration.
        # (We only use min_s to avoid creating tiny scenes when audio is short.)
        max_reasonable = int(max(1, math.floor(audio_seconds / float(min_s)))) if min_s > 0 else max_scenes_i
        target_scenes = min(target_scenes, max_scenes_i)
        if max_reasonable >= 1:
            target_scenes = min(target_scenes, max_scenes_i)
            # If our target creates very short scenes (<~min), reduce count (down to 1).
            if (audio_seconds / float(target_scenes)) < float(min_s) and max_reasonable < target_scenes:
                target_scenes = max(1, max_reasonable)

        target_scenes = clamp_int(target_scenes, 1, max_scenes_i)

    images = _generate_segment_images(
        api_key=str(api_key),
        extract_dir=extract_dir_p,
        pair=pair,
        width=int(width),
        height=int(height),
        max_scenes=int(target_scenes),
    )
    if not images:
        raise RuntimeError("Image generation produced zero images (unexpected).")

    # True audio-driven per-scene duration (no upward clamp).
    if audio_seconds <= 0.0:
        per = 5.0
    else:
        per = max(0.05, audio_seconds / float(len(images)))

    scene_seconds = [float(per) for _ in images]

    seg_slug = safe_slug(str(pair.get("title_guess") or segment_label(pair) or "segment"), max_len=48)
    clips_dir = ensure_dir(out_path_p.parent / f"_{seg_slug}_clips")

    clip_paths: List[Path] = []
    for idx, (img, dur) in enumerate(zip(images, scene_seconds), start=1):
        clip_out = clips_dir / f"scene_{idx:03d}.mp4"
        clip_paths.append(
            build_scene_clip_from_image(
                image_path=img,
                out_mp4_path=clip_out,
                duration_s=float(dur),
                width=int(width),
                height=int(height),
                fps=int(fps),
                zoom_strength=float(zoom_strength),
            )
        )

    concat_path = out_path_p.parent / f"_{seg_slug}_concat.mp4"
    concat_video_clips(clip_paths, concat_path)

    mux_audio_to_video(
        video_path=concat_path,
        audio_path=audio_p,
        out_mp4_path=out_path_p,
        end_buffer_s=_DEFAULT_END_BUFFER,
        audio_seconds=float(audio_seconds) if audio_seconds > 0 else None,
    )


    # Optional: burn subtitles AFTER mux (keeps locked order intact).
    if bool(burn_subtitles):
        script_path = str(pair.get("script_path") or pair.get("script_file") or pair.get("script") or "").strip()
        script_text = ""
        if script_path and Path(script_path).exists():
            script_text = Path(script_path).read_text(encoding="utf-8", errors="ignore")
        script_text = (script_text or "").strip()

        if not script_text:
            raise RuntimeError("Subtitles requested, but script text is missing/empty for this segment.")

        try:
            total_for_srt = float(audio_seconds or 0.0) + float(_DEFAULT_END_BUFFER)
            if total_for_srt <= 0.5:
                total_for_srt = float(ffprobe_duration_seconds(out_path_p) or 0.0) or 1.0
        except Exception:
            total_for_srt = 1.0

        
        # Caption density tuning:
        # 9:16 + Shorts style needs smaller chunks to prevent right-edge clipping.
        # We derive this from output aspect, not from model calls.
        is_vertical = False
        try:
            is_vertical = int(height) > int(width)
        except Exception:
            is_vertical = False

        stl = _subtitle_style_resolved(subtitle_style, width=int(width), height=int(height))

        # Caption chunking + wrapping tuned by aspect/style to prevent edge clipping.
        if is_vertical and stl == "shorts":
            cap_max_words = 5
            cap_max_chars = 28
            cap_max_line_chars = 16   # keep each line short for big center subtitles
            cap_max_lines = 2
        elif is_vertical:
            cap_max_words = 6
            cap_max_chars = 34
            cap_max_line_chars = 20
            cap_max_lines = 2
        else:
            cap_max_words = 6
            cap_max_chars = 44
            cap_max_line_chars = 44
            cap_max_lines = 1

        srt_text = build_srt_from_script(
            script_text,
            total_seconds=total_for_srt,
            max_words=cap_max_words,
            max_chars=cap_max_chars,
            max_line_chars=cap_max_line_chars,
            max_lines=cap_max_lines,
        ).strip()
        if not srt_text:
            raise RuntimeError("Subtitles requested, but SRT generation produced empty output.")

        # Write SRT next to the MP4 (preferred). Fall back to extract_dir/_subs if needed.
        srt_path = out_path_p.with_suffix(".srt")
        try:
            srt_path.write_text(srt_text, encoding="utf-8")
        except Exception:
            srt_path = ensure_dir(extract_dir_p / "_subs") / (out_path_p.stem + ".srt")
            srt_path.write_text(srt_text, encoding="utf-8")

        tmp_out = out_path_p.with_name(out_path_p.stem + "_subbed.mp4")
        style_norm = (subtitle_style or "Auto").strip().lower()
        if style_norm.startswith("short"):
            style_norm = "shorts"
        elif style_norm.startswith("standard"):
            style_norm = "standard"
        else:
            style_norm = "auto"

        burn_subtitles_to_mp4(
            src_mp4=out_path_p,
            srt_path=srt_path,
            dst_mp4=tmp_out,
            style=style_norm,
            width=int(width),
            height=int(height),
        )

        # Replace original (validate temp first).
        if not tmp_out.exists() or tmp_out.stat().st_size < 50_000:
            raise RuntimeError("Subtitle burn failed (output missing or too small).")

        try:
            out_path_p.unlink()
        except Exception:
            pass
        tmp_out.replace(out_path_p)

        # If user doesn't want sidecar SRT, remove it after successful burn.
        if not bool(export_srt):
            try:
                if srt_path.exists():
                    srt_path.unlink()
            except Exception:
                pass

    # Cleanup
    if os.environ.get("UAPPRESS_KEEP_CLIPS", "").strip() != "1":
        try:
            shutil.rmtree(str(clips_dir), ignore_errors=True)
        except Exception:
            pass
        try:
            if concat_path.exists():
                concat_path.unlink()
        except Exception:
            pass
        try:
            if concat_path.exists():
                try:
                    concat_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass

################################################################################
# Sora Shorts Prompt Helpers (added for UAPpress Shorts pipeline)
################################################################################
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

@dataclass(frozen=True)
class SoraStyle:
    mode: Literal["cinematic_realism", "archival_reenactment"]
    camera_rules: str
    lighting_rules: str
    grain_rules: str
    realism_constraints: str

def build_sora_house_style(mode: str = "cinematic_realism") -> SoraStyle:
    """Return a consistent 'house style' bundle for Sora prompts."""
    m = (mode or "cinematic_realism").strip().lower()
    if m in ("archival", "archival_reenactment", "archival reenactment", "archival-reenactment"):
        return SoraStyle(
            mode="archival_reenactment",
            camera_rules=(
                "Tripod or slow dolly only, no handheld shake. "
                "Lens: 35–50mm equivalent, modest depth of field. "
                "No fast pans, no whip-zooms, no drone shots."
            ),
            lighting_rules=(
                "Period-accurate practical lighting, sodium-vapor or tungsten feel, "
                "limited dynamic range, softer contrast, restrained highlights."
            ),
            grain_rules=(
                "Visible film grain, mild gate weave, slight vignetting, "
                "subtle halation, minor dust/scratch artifacts."
            ),
            realism_constraints=(
                "Documentary reenactment vibe, grounded staging, "
                "no sci-fi glow, no fantasy aesthetics, no modern UI overlays."
            ),
        )
    # default cinematic realism
    return SoraStyle(
        mode="cinematic_realism",
        camera_rules=(
            "Slow cinematic movement only: gentle push-in, slow lateral slide, "
            "or locked-off composition. No sudden zooms or shakes. "
            "Lens: 28–50mm equivalent, tasteful depth of field."
        ),
        lighting_rules=(
            "Moody, believable lighting. Practical sources (street lamps, headlights, "
            "hangar lights). Controlled contrast, no blown highlights."
        ),
        grain_rules=(
            "Subtle grain, light texture, clean but not glossy. "
            "No heavy stylization, no cartoon look."
        ),
        realism_constraints=(
            "Cinematic realism, restrained color palette, "
            "no text, no logos, no subtitles, no watermarks."
        ),
    )

def build_sora_prompt(
    segment=None,
    *,
    scene_text: str | None = None,
    # Documentary-first defaults (UAPpress)
    character: str = "a restrained documentary scene",
    style: str = "cinematic documentary realism",
    environment: str = "historically plausible locations, natural lighting, restrained color grade",
    camera: str = "stable tripod or slow drift, documentary framing",
    movement: str = "minimal motion, observational, no exaggerated action",
    constraints: str = (
        "No text, no logos, no subtitles, no watermarks. "
        "No aliens, no creatures, no sci-fi effects. "
        "No game aesthetics. Historically plausible."
    ),
    # Optional structured knobs (app may pass these)
    mode: str = "cinematic_realism",
    length_s: int | None = None,
    aspect: str = "9:16",
    fps: int = 30,
    **_ignored,
) -> str:
    """
    Backward-compatible Sora prompt builder.

    Supports BOTH call styles:
      1) build_sora_prompt(segment_obj)
      2) build_sora_prompt(scene_text="...", character="...", ...)

    segment may be:
      - a dict with keys like 'visual_prompt', 'scene_prompt', 'prompt', 'text'
      - an object with similarly-named attributes
    """
    # Resolve scene text
    resolved = (scene_text or "").strip()

    if not resolved and segment is not None:
        # dict-style
        if isinstance(segment, dict):
            for k in ("visual_prompt", "scene_prompt", "prompt", "text", "narration", "script"):
                v = segment.get(k)
                if isinstance(v, str) and v.strip():
                    resolved = v.strip()
                    break
        else:
            # object-style
            for k in ("visual_prompt", "scene_prompt", "prompt", "text", "narration", "script"):
                v = getattr(segment, k, None)
                if isinstance(v, str) and v.strip():
                    resolved = v.strip()
                    break

    # If still empty, provide a safe generic fallback
    if not resolved:
        resolved = "A short, engaging scene that matches the narration without any on-screen text."

    style_bundle = build_sora_house_style(mode=mode)

    # Normalize fps/aspect/length metadata (descriptive for Sora)
    try:
        fps_i = int(fps)
    except Exception:
        fps_i = 30
    if fps_i <= 0:
        fps_i = 30
    asp = (aspect or "9:16").strip()

    meta = []
    if length_s is not None:
        try:
            meta.append(f"Target length: {int(length_s)} seconds.")
        except Exception:
            pass
    meta.append(f"Aspect ratio: {asp}.")
    meta.append(f"Frame rate: {fps_i} fps.")
    meta_txt = " ".join(meta)

    prompt = (
        "Create a cinematic, documentary-style video scene.\n"
        f"{meta_txt}\n\n"
        f"Scene:\n{resolved}\n\n"
        f"Style: {style}.\n"
        f"Environment: {environment}.\n"
        f"Camera: {camera}. {style_bundle.camera_rules}\n"
        f"Motion: {movement}.\n"
        f"Lighting: {style_bundle.lighting_rules}\n"
        f"Texture: {style_bundle.grain_rules}\n"
        f"Constraints: {constraints} {style_bundle.realism_constraints}"
    )
    return prompt

def prepare_sora_short_job(
    prompt_text: str,
    *,
    preset: str = "12s",
    mode: str = "cinematic_realism",
    aspect: str = "9:16",
    fps: int = 30,
) -> Dict[str, Any]:
    """Return a structured payload the app can store/log for a Sora short."""
    preset_map = {"7s": 7, "12s": 12, "18s": 18}
    length_s = preset_map.get((preset or "12s").strip().lower(), 12)
    full_prompt = build_sora_prompt(scene_text=prompt_text, mode=mode, length_s=length_s, aspect=aspect, fps=fps)
    return {
        "preset": preset,
        "mode": mode,
        "aspect": aspect,
        "fps": fps,
        "length_s": length_s,
        "prompt": full_prompt,
    }

from dataclasses import dataclass, asdict


def generate_all_segments_sequential(
    *,
    segments: List[Dict[str, Any]],
    extract_dir: str,
    out_dir: str,
    overwrite: bool,
    api_key: str,
    fps: int,
    width: int,
    height: int,
    zoom_strength: float,
    max_scenes: int,
    min_scene_seconds: int,
    max_scene_seconds: int,
) -> List[str]:
    """Generate MP4s for all segments sequentially.

    This exists mainly for backward-compat with older app.py versions that may call it.
    The current Streamlit app typically orchestrates the loop itself.

    Returns a list of output MP4 paths.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    outputs: List[str] = []
    for i, pair in enumerate(segments or [], start=1):
        # Respect an explicit out_path if caller provided one in the pair dict
        explicit = str(pair.get("out_path") or "").strip()
        if explicit:
            out_path_p = Path(explicit)
            out_path_p.parent.mkdir(parents=True, exist_ok=True)
        else:
            base = str(pair.get("key") or pair.get("label") or segment_label(pair) or f"segment_{i:02d}")
            slug = safe_slug(base, max_len=80) or f"segment_{i:02d}"
            out_path_p = out_dir_p / f"{i:02d}_{slug}.mp4"

        if out_path_p.exists() and (not bool(overwrite)):
            outputs.append(str(out_path_p))
            continue

        render_segment_mp4(
            pair=pair,
            extract_dir=str(extract_dir),
            out_path=str(out_path_p),
            api_key=str(api_key),
            fps=int(fps),
            width=int(width),
            height=int(height),
            zoom_strength=float(zoom_strength),
            max_scenes=int(max_scenes),
            min_scene_seconds=int(min_scene_seconds),
            max_scene_seconds=int(max_scene_seconds),
        )
        outputs.append(str(out_path_p))

    return outputs

# Back-compat alias for older app.py calls
_generate_all_segments_sequential = generate_all_segments_sequential
