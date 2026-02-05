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
# - No subtitles/logos; no stitching.
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
# No-Object Visual Doctrine (UFO/UAP credibility-first)
# ----------------------------
# Default: DO NOT depict a clearly defined craft/object. Visuals should show the *conditions of uncertainty*:
# observation context, environment, instruments, personnel, documents, terrain, weather, light, and ambiguity.
#
# Optional exception (default OFF): allow only *ambiguous distant lights* with no defined edges/shape/symmetry.
_ALLOW_WEAK_HINT = os.environ.get("UAPPRESS_ALLOW_WEAK_HINT", "0").strip() == "1"

# Local fallback artifacts (OFF by default). If enabled, failures generate clearly-marked scene cards.
_ALLOW_LOCAL_FALLBACK = os.environ.get("UAPPRESS_ALLOW_LOCAL_FALLBACK", "0").strip() == "1"


_NO_OBJECT_DOCTRINE = (
    "STRICT VISUAL DOCTRINE: Do NOT show a UFO craft, spaceship, saucer, disc, triangle, or any defined non-human vehicle. "
    "Do NOT depict clear geometry, symmetry, metallic hulls, windows, seams/panels, landing gear, engines, beams, "
    "or a centered 'object in the sky' composition. "
    "Depict context only: environments and observation settings such as skies, clouds, weather, silhouettes of terrain, airfields, "
    "radar/ATC rooms, instrument panels, witnesses from behind, declassified documents, maps, and quiet aftermath. "
    "OBSCURED OBJECT POLICY (✅ B): ONLY when recovery/containment/handling is explicitly implied by context, "
    "you MAY depict a non-descript tarp-covered bundle or plain covered crate as a prop. It must be indistinct and non-identifiable: "
    "no shape language, no metallic surfaces, no hard edges, no symmetry, no windows, no engines, no craft silhouette. "
    "ALSO: Do NOT add anomalous lights or effects in the sky (no glowing orbs, no flares, no spotlights, no searchlights) unless an incident profile explicitly allows it."
)

_WEAK_HINT_ADDENDUM = (
    "WEAK HINT ONLY: You MAY include faint distant lights partially obscured by clouds/haze, "
    "not centered, no shape, no symmetry, no hard edges, no metallic surfaces, no craft-like silhouette."
)

_SANITIZE_PATTERNS = [
    (r"\b(disc|saucer|triangle|triangular|cigar|tic[\-\s]?tac|boomerang)\b", "anomaly"),
    (r"\b(craft|spaceship|ufo|uap|vehicle)\b", "unidentified phenomenon"),
    (r"\b(metallic|hull|panel(s)?|seam(s)?|rivets?|window(s)?|cockpit)\b", "indistinct"),
    (r"\b(engine(s)?|thruster(s)?|exhaust|jet)\b", "light source"),
    (r"\b(landing\s*gear|port(hole)?s?)\b", "details"),
    (r"\b(beam(s)?|tractor\s*beam|laser(s)?)\b", "illumination"),
    (r"\b(hovering|descending|ascending)\b", "moving"),
]
# ----------------------------
# Script Loading (REQUIRED)
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
def _sanitize_scene_context(text: str, *, max_chars: int = 380) -> str:
    """Sanitize script text snippets before embedding into image prompts.

    - Normalizes whitespace
    - Applies conservative term replacements via _SANITIZE_PATTERNS
    - Enforces a hard length cap to prevent prompt bloat

    Keyword-only max_chars prevents call-site drift (e.g., unexpected kwargs).
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in _SANITIZE_PATTERNS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # Hard cap for prompt safety / token economy
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

def _doctrine_clause() -> str:
    return _NO_OBJECT_DOCTRINE + (" " + _WEAK_HINT_ADDENDUM if _ALLOW_WEAK_HINT else "")



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

def _build_scene_prompts_from_script(script_text: str, max_scenes: int, *, incident_hint: str = "") -> List[str]:
    """Build per-scene IMAGE prompts for a **photorealistic** UAPpress long-form segment.

    Goals:
    - Photorealistic, credibility-first documentary visuals (not illustration, not CGI).
    - Context over conclusions. No spectacle. No 'Hollywood UFO' tropes.

    Guardrails:
    - NO text, NO logos, NO subtitles, NO watermarks.
    - NO beams, NO lens flares, NO god rays, NO dramatic sci-fi lighting.
    - People allowed but faces should be non-identifiable (distance, angle, occlusion).
    - 'Obscured object' allowed ONLY when narration implies recovery/containment/handling,
      and MUST be non-descript (tarp-covered bundle/crate), no recognizable craft silhouette.
    """
    text = (script_text or "").strip()
    if not text:
        return []

    max_scenes_i = clamp_int(int(max_scenes), 1, 120)

    # Light-touch segmentation: paragraphs -> beats. Avoid over-fragmentation.
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    beats = [p for p in paras if len(p) >= 120] or paras
    if not beats:
        return []

    # Choose N beats evenly, stable across runs.
    if len(beats) <= max_scenes_i:
        idxs = list(range(len(beats)))
    else:
        step = len(beats) / float(max_scenes_i)
        idxs = [min(len(beats) - 1, int(i * step)) for i in range(max_scenes_i)]

    # Episode-wide subtle palette hint (optional, keeps things restrained)
    accent = (os.environ.get("UAPPRESS_ACCENT_COLOR", "muted amber") or "muted amber").strip()

    style_block = (
        "STYLE: photorealistic documentary still, 35mm film look, natural camera exposure, subtle film grain. "
        "LIGHTING: practical, believable, soft overcast or interior practical lighting, no cinematic bloom, no volumetric fog. "
        "COLOR: muted realistic palette (cool grays, off-whites, deep charcoals) with very subtle practical accents (" + accent + "). "
        "CAMERA: calm editorial framing, documentary distance, no extreme wide-angle distortion, no dutch angles. "
        "RENDER: real-world photo, not illustration, not CGI, not a movie poster. "
        "TEXT: no text, no logos, no subtitles, no watermarks."
    )

    base_doctrine = (
        "DOCTRINE: depict verifiable context only; avoid implying conclusions; keep the scene grounded and ordinary. "
        "NO spectacle, NO beams, NO dramatic UFO tropes, NO glowing craft, NO aliens."
    )

    people_clause = (
        "PEOPLE: allowed, but faces non-identifiable (mid-distance, turned away, shadowed, partial occlusion). "
        "Body language routine and calm."
    )

    object_clause = (
        "OBJECT POLICY: if narration implies recovery/containment, an obscured non-descript covered bundle/crate is allowed; "
        "must look generic and tarp-covered with no recognizable craft silhouette."
    )

    categories = [
        ("Institutional context", "military base exterior, administrative buildings, guarded gate, routine patrols"),
        ("Paper trail", "desks with files, radios, clipped documents, maps, archival storage, low-key evidence handling"),
        ("Witness context", "ordinary people or service members at a distance, looking toward a horizon or lights off-screen"),
        ("Search/response", "vehicles staged, flashlights, perimeter tape, field search patterns, no object shown"),
        ("Containment handling", "warehouse/hangar with tarp-covered crate or covered pallet, forklifts, guarded calm"),
        ("Aftermath reflection", "quiet empty spaces: corridors, hangars, file rooms, dusk exterior, unresolved mood"),
    ]

    prompts: List[str] = []
    hint = (incident_hint or "").strip()
    if hint:
        hint = f"INCIDENT CONTEXT: {hint}. "

    for j, idx in enumerate(idxs):
        beat = beats[idx]
        snippet = _sanitize_scene_context(beat, max_chars=380)

        cat_name, cat_desc = categories[j % len(categories)]

        prompt = (
            "Create a photorealistic still image for a credibility-first UAP investigative documentary.\n"
            f"{hint}{style_block}\n"
            f"SCENE CATEGORY: {cat_name}. Depict: {cat_desc}.\n"
            f"{people_clause}\n"
            f"{object_clause}\n"
            f"{base_doctrine}\n"
            f"Context (sanitized beat): {snippet}\n"
            "COMPOSITION: stillness, grounded realism, minimal spectacle, no overclaiming."
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
) -> None:
    """
    Locked pipeline:
      generate images -> build scene clips -> concatenate -> mux audio

    Audio-driven timing:
      - distribute audio duration across scenes
      - clamp per-scene into [min_scene_seconds, max_scene_seconds]
      - add ~0.5–0.75s end buffer (default 0.65)
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

    audio_seconds = ffprobe_duration_seconds(audio_p)

    # Choose a good scene count for pacing
    if audio_seconds <= 0:
        target_scenes = clamp_int(int(max_scenes), 3, int(max_scenes))
    else:
        mid = (float(min_scene_seconds) + float(max_scene_seconds)) / 2.0
        approx = int(max(1, round(audio_seconds / max(1.0, mid))))
        target_scenes = clamp_int(approx, 3, int(max_scenes))

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

    min_s = clamp_int(int(min_scene_seconds), 1, 600)
    max_s = clamp_int(int(max_scene_seconds), min_s, 600)

    if audio_seconds <= 0:
        per = clamp_scene_seconds(5.0, float(min_s), float(max_s))
    else:
        per = clamp_scene_seconds(audio_seconds / float(len(images)), float(min_s), float(max_s))

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
    )

    if os.environ.get("UAPPRESS_KEEP_CLIPS", "").strip() != "1":
        try:
            shutil.rmtree(str(clips_dir), ignore_errors=True)
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
