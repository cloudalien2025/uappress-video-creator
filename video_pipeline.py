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
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


# ----------------------------
# Constants + small utilities
# ----------------------------

_END_BUFFER_MIN = 0.50
_END_BUFFER_MAX = 0.75
_DEFAULT_END_BUFFER = 0.65

_DEFAULT_FPS = 30

# Image generation defaults (override via env if desired)
# IMPORTANT: gpt-image-1 supports sizes like 1024x1024, 1536x1024, 1024x1536.
_DEFAULT_IMAGE_MODEL = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1")
_DEFAULT_IMAGE_SIZE_169 = os.environ.get("UAPPRESS_IMAGE_SIZE_169", "1536x1024")
_DEFAULT_IMAGE_SIZE_916 = os.environ.get("UAPPRESS_IMAGE_SIZE_916", "1024x1536")

# Cache version knob (bump if you ever want to invalidate all cached images)
_IMAGE_CACHE_ROOT = os.environ.get("UAPPRESS_IMAGE_CACHE_ROOT", "_image_cache_v2")


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

@dataclass
class BrandIntroOutroSpec:
    """
    app.py constructs a much richer spec. To avoid breaking on extra fields,
    we accept them via **kwargs by including a generic 'extra' dict AND providing
    the common fields app.py uses directly.
    """
    brand_name: str = "UAPpress"
    channel_or_series: str = "UAPpress Investigations"
    tagline: str = ""
    episode_title: str = "UAPpress Episode"
    visual_style: str = ""
    palette: str = ""
    logo_text: Optional[str] = None
    intro_music_cue: str = ""
    outro_music_cue: str = ""
    cta_line: str = ""
    sponsor_line: Optional[str] = None
    aspect: str = "landscape"  # "landscape" or "portrait"
    global_mode: bool = True
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


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


def generate_sora_brand_intro_outro(
    client: Any,
    spec: BrandIntroOutroSpec,
    out_dir: Union[str, Path],
    *,
    model: str = "sora-2-pro",
    intro_seconds: Optional[int] = None,
    outro_seconds: Optional[int] = None,
    seconds: Optional[int] = None,  # older fallback signature
    intro_reference_image: Optional[str] = None,
    outro_reference_image: Optional[str] = None,
) -> Tuple[str, str]:
    """
    app.py expects:
      intro_path, outro_path = generate_sora_brand_intro_outro(...)
    """
    out_dir = ensure_dir(out_dir)
    brand_slug = safe_slug(spec.brand_name or "brand", max_len=40)

    intro_s = int(intro_seconds or seconds or 8)
    outro_s = int(outro_seconds or seconds or 8)
    intro_s = clamp_int(intro_s, 2, 20)
    outro_s = clamp_int(outro_s, 2, 20)

    base_style = (spec.visual_style or "").strip() or (
        "Radar/FLIR surveillance aesthetic, monochrome/low-saturation, subtle scanlines, "
        "HUD overlays, gridlines, bearing ticks, minimal telemetry numbers, soft glow, restrained film grain. "
        "Slow camera drift. Serious investigative tone. No aliens, no monsters, no cheesy explosions."
    )

    intro_prompt = (
        f"{base_style}\n\n"
        f"GLOBAL channel intro bumper for {spec.brand_name}. "
        f"Show the title '{spec.brand_name}' and then '{spec.channel_or_series}'. "
        f"Clean premium typography, stable and readable. "
        f"Duration {intro_s} seconds."
    )

    outro_prompt = (
        f"{base_style}\n\n"
        f"GLOBAL channel outro bumper for {spec.brand_name}. "
        f"Show '{spec.brand_name}' and CTA: '{spec.cta_line or 'Subscribe for more investigations.'}'. "
        f"{'Also include sponsor line: ' + spec.sponsor_line + '. ' if spec.sponsor_line else ''}"
        f"Clean premium typography, stable and readable. "
        f"Duration {outro_s} seconds."
    )

    intro_out = out_dir / f"{brand_slug}_intro_tmp.mp4"
    outro_out = out_dir / f"{brand_slug}_outro_tmp.mp4"

    def _gen_one(prompt: str, duration_s: int, ref_img: Optional[str], dst: Path) -> Path:
        try:
            kwargs: Dict[str, Any] = {}
            if ref_img:
                kwargs["reference_image"] = ref_img

            resp = client.videos.generate(
                model=model,
                prompt=prompt,
                duration_seconds=duration_s,
                **kwargs,
            )
            content_id = None
            if isinstance(resp, dict):
                content_id = resp.get("content_id") or resp.get("id")
            else:
                content_id = getattr(resp, "content_id", None) or getattr(resp, "id", None)
            if not content_id:
                raise RuntimeError("Sora generate did not return a content id.")
            return _download_openai_video_content(client, str(content_id), dst)
        except TypeError:
            resp = client.videos.generate(
                model=model,
                prompt=prompt,
                duration_seconds=duration_s,
            )
            content_id = None
            if isinstance(resp, dict):
                content_id = resp.get("content_id") or resp.get("id")
            else:
                content_id = getattr(resp, "content_id", None) or getattr(resp, "id", None)
            if not content_id:
                raise RuntimeError("Sora generate did not return a content id.")
            return _download_openai_video_content(client, str(content_id), dst)

    intro_path = _gen_one(intro_prompt, intro_s, intro_reference_image, intro_out)
    outro_path = _gen_one(outro_prompt, outro_s, outro_reference_image, outro_out)
    return (str(intro_path), str(outro_path))


# ---------------------------------------------------------
# Core rendering: images -> scene clips -> concat -> mux audio
# ---------------------------------------------------------

def _make_zoompan_filter(
    width: int,
    height: int,
    duration_s: float,
    fps: int,
    zoom_strength: float,
) -> str:
    """
    Ken Burns style zoompan that avoids black edges:
      scale cover -> crop -> zoompan.
    """
    frames = max(1, int(math.ceil(duration_s * fps)))
    zoom_target = clamp_float(float(zoom_strength), 1.0, 1.25)
    inc = (zoom_target - 1.0) / max(1, frames)
    inc = max(0.00005, min(0.005, inc))

    zoom_expr = f"min(zoom+{inc:.6f},{zoom_target:.4f})"
    x_expr = f"(iw-{width})/2"
    y_expr = f"(ih-{height})/2"

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={width}x{height}:fps={fps}"
    )
    return vf


def build_scene_clip_from_image(
    image_path: Union[str, Path],
    out_mp4_path: Union[str, Path],
    duration_s: float,
    width: int,
    height: int,
    fps: int,
    zoom_strength: float,
) -> Path:
    """
    Turn a single image into a short MP4 scene clip (no audio).
    """
    image_path = Path(image_path)
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    vf = _make_zoompan_filter(width, height, float(duration_s), int(fps), float(zoom_strength))
    ffmpeg = which_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-t", f"{float(duration_s):.3f}",
        "-vf", vf,
        "-r", str(int(fps)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4_path),
    ]
    run_ffmpeg(cmd)
    return out_mp4_path


def concat_video_clips(clip_paths: Sequence[Union[str, Path]], out_mp4_path: Union[str, Path]) -> Path:
    """
    Concatenate MP4 clips (video only) using concat demuxer.
    """
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        list_path = td_path / "concat_list.txt"
        lines: List[str] = []
        for p in clip_paths:
            pth = Path(p)
            path_str = str(pth).replace("'", "'\\\\''")
            lines.append("file '" + path_str + "'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ffmpeg = which_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_mp4_path),
        ]
        run_ffmpeg(cmd)

    return out_mp4_path


def mux_audio_to_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    out_mp4_path: Union[str, Path],
    end_buffer_s: float = _DEFAULT_END_BUFFER,
) -> Path:
    """
    Mux narration audio to the concatenated video, audio-driven with small end buffer.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    end_buffer_s = clamp_float(float(end_buffer_s), _END_BUFFER_MIN, _END_BUFFER_MAX)
    ffmpeg = which_ffmpeg()

    audio_dur = ffprobe_duration_seconds(audio_path)
    if audio_dur > 0:
        target = audio_dur + end_buffer_s
        a_filter = f"apad,atrim=0:{target:.3f}"
    else:
        a_filter = f"apad=pad_dur={end_buffer_s:.3f}"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter:a", a_filter,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        "-shortest",
        "-movflags", "+faststart",
        str(out_mp4_path),
    ]
    run_ffmpeg(cmd)
    return out_mp4_path


# ============================================================
# SECTION 2 — app.py compatibility: ZIP, pairing, rendering shim
# ============================================================

_SCRIPT_EXTS = {".txt", ".md", ".json"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


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
            t = line.strip()
            if not t:
                continue
            t = t.lstrip("#").strip()
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



def _make_scene_card_image(path: Path, width: int, height: int, headline='SCENE IMAGE (LOCAL FALLBACK)', body=str(prompt)[:900] if 'prompt' in locals() else '', footer='UAPpress') -> None:
    """
    Local fallback if OpenAI image generation fails.

    IMPORTANT:
    - Must NOT look like a 'black screen' video.
    - Must be obviously a placeholder, so failures are diagnosable at a glance.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal valid PNG fallback if Pillow isn't available.
    if Image is None:
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDAT\x08\xd7c``\x00\x00\x00\x04\x00\x01"
            b"\r\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return

    w = max(320, int(width))
    h = max(240, int(height))

    # Bright neutral background (avoid dark/black)
    img = Image.new("RGB", (w, h), color=(235, 235, 235))

    try:
        draw = ImageDraw.Draw(img)
        draw.rectangle([(8, 8), (w - 8, h - 8)], outline=(180, 0, 0), width=6)

        msg1 = title.strip()[:80]
        msg2 = "OpenAI image generation failed — check model / key / quota."
        msg3 = "Placeholder frame (not a render bug)."

        try:
            font_big = ImageFont.truetype("DejaVuSans.ttf", 44)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 28)
        except Exception:
            font_big = ImageFont.load_default()
            font_small = ImageFont.load_default()

        y = 70
        draw.text((40, y), msg1, fill=(140, 0, 0), font=font_big)
        y += 70
        draw.text((40, y), msg2, fill=(0, 0, 0), font=font_small)
        y += 45
        draw.text((40, y), msg3, fill=(0, 0, 0), font=font_small)
    except Exception:
        pass

    img.save(path, format="PNG")

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
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDAT\x08\xd7c``\x00\x00\x00\x04\x00\x01"
            b"\r\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
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

def _openai_generate_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    """
    Generate a single image via OpenAI Images API.
    Returns raw image bytes (PNG/JPG), or raises.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available in this environment.")
    client = OpenAI(api_key=str(api_key))

    models_to_try = [
        _DEFAULT_IMAGE_MODEL,
        os.environ.get("UAPPRESS_IMAGE_MODEL_FALLBACK_1", "dall-e-3"),
        os.environ.get("UAPPRESS_IMAGE_MODEL_FALLBACK_2", "dall-e-2"),
    ]

    last_err = None
    resp = None
    for mname in models_to_try:
        try:
            resp = client.images.generate(
                model=mname,
                prompt=prompt,
                size=size,
                response_format="b64_json",
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue

    if resp is None and last_err is not None:
        raise last_err

    data0 = None
    try:
        data0 = resp.data[0]  # type: ignore[attr-defined]
    except Exception:
        data = resp.get("data") if isinstance(resp, dict) else None
        if data and isinstance(data, list) and data:
            data0 = data[0]

    if data0 is None:
        raise RuntimeError("Image response missing data[0].")

    b64 = None
    url = None
    if isinstance(data0, dict):
        b64 = data0.get("b64_json")
        url = data0.get("url")
    else:
        b64 = getattr(data0, "b64_json", None)
        url = getattr(data0, "url", None)

    if b64:
        import base64
        return base64.b64decode(b64)

    if url:
        raise RuntimeError("Image API returned URL-only output; URL fetch not supported in this pipeline.")
    raise RuntimeError("Image API returned neither b64_json nor url.")


def _build_scene_prompts_from_script(script_text: str, max_scenes: int) -> List[str]:
    """
    Heuristic: break script into chunks and generate scene prompts.
    Keep prompts restrained (investigative tone, no logos/subtitles).
    """
    text = (script_text or "").strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    paras = [p for p in paras if len(p) >= 80] or paras

    n = min(int(max_scenes), max(1, len(paras)))
    if n <= 0:
        return []

    if len(paras) <= n:
        idxs = list(range(len(paras)))
    else:
        step = len(paras) / float(n)
        idxs = [min(len(paras) - 1, int(i * step)) for i in range(n)]

    prompts: List[str] = []
    for pi in idxs:
        snippet = re.sub(r"\s+", " ", paras[pi]).strip()
        if len(snippet) > 420:
            snippet = snippet[:420].rstrip() + "…"

        prompts.append(
            "Create a documentary still image for a serious UFO/UAP investigative video. "
            "Cinematic, realistic, restrained, credibility-first. "
            "No text, no logos, no subtitles, no watermarks. "
            "Moody lighting, archival / military / night-ops / radar / airbase / forest / coast as appropriate. "
            f"Scene context: {snippet}"
        )
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
    """
    Generate (or reuse cached) images for this segment.
    Uses a segment-specific cache folder to avoid shared images across segments.
    """
    img_dir = _segment_image_dir(extract_dir, pair)
    size = _image_size_for_mode(int(width), int(height))

    existing = sorted([p for p in img_dir.glob("scene_*.png") if p.is_file()])
    if existing:
        return existing[: max(1, int(max_scenes))]

    script_path = str(pair.get("script_path") or "")
    script_text = _read_text_preview(script_path, max_chars=12000) if script_path else ""
    prompts = _build_scene_prompts_from_script(script_text, max_scenes=int(max_scenes)) if script_text else []

    if not prompts:
        prompts = [
            "Cinematic documentary still, restrained investigative tone, no text, no logos, no subtitles. "
            "UAP investigation mood: night sky, distant lights, radar room, airbase perimeter, forest trail."
        ] * max(3, min(8, int(max_scenes)))

    images: List[Path] = []
    for i, prompt in enumerate(prompts, start=1):
        out = img_dir / f"scene_{i:03d}.png"
        try:
            data = _openai_generate_image_bytes(str(api_key), prompt, size)
            out.write_bytes(data)
        except Exception:
            _make_scene_card_image(out, int(width), int(height, headline='SCENE IMAGE (LOCAL FALLBACK)', body=str(prompt)[:900] if 'prompt' in locals() else '', footer='UAPpress'))
        images.append(out)

    return images


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

    audio_path = str(pair.get("audio_path") or "").strip()
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
