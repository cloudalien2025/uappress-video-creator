# ============================================================
# video_pipeline.py — UAPpress Video Creator (CORRECTED)
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
# Locked pipeline order:
#   images -> scene clips -> concatenate -> mux audio (no black video)
#
# Audio-driven timing:
#   end buffer ~0.5–0.75s (default 0.65s)
#
# Branding helpers for app.py detection:
#   BrandIntroOutroSpec + generate_sora_brand_intro_outro(...)
#
# IMPORTANT FIXES in this corrected version:
#   1) ✅ Removes Cloud-breaking concat list f-string escaping risk (no SyntaxError)
#   2) ✅ Fixes “same images across segments” by using a provably unique, stable
#      per-segment cache key (hash of script/audio identity) instead of collapsing dirs.
#   3) ✅ Default behavior: DO NOT reuse cached images across reruns unless explicitly enabled:
#         set env UAPPRESS_REUSE_IMAGES=1 to reuse.
#   4) ✅ Writes small debug breadcrumbs per segment image run (cache dir, source script/audio)
#      to make future diagnosis obvious.
# ============================================================

from __future__ import annotations

import dataclasses
import hashlib
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
_DEFAULT_IMAGE_MODEL = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1")
_DEFAULT_IMAGE_SIZE_169 = os.environ.get("UAPPRESS_IMAGE_SIZE_169", "1536x1024")
_DEFAULT_IMAGE_SIZE_916 = os.environ.get("UAPPRESS_IMAGE_SIZE_916", "1024x1536")

# Cache reuse toggle:
#   - Default OFF to guarantee segments do not share images and to make debugging honest.
#   - Enable reuse (cheaper) only when you're sure cache keys are stable and correct.
_REUSE_IMAGES = os.environ.get("UAPPRESS_REUSE_IMAGES", "").strip() == "1"

# Debug breadcrumbs toggle (writes tiny JSON beside images)
_DEBUG_IMAGES = os.environ.get("UAPPRESS_DEBUG_IMAGES", "").strip() != "0"


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


def which_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    return "ffprobe"


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


def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    """
    Return media duration seconds using ffprobe (best-effort).
    """
    ffprobe = which_ffprobe()
    p = str(path)
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
    if proc.returncode != 0:
        return 0.0
    try:
        return float((proc.stdout or "").strip())
    except Exception:
        return 0.0


def clamp_scene_seconds(value: float, min_scene_s: float, max_scene_s: float) -> float:
    """
    For Streamlit slider/session_state stability: clamp into [min_scene_s, max_scene_s]
    """
    min_scene_s = float(min_scene_s)
    max_scene_s = float(max_scene_s)
    if max_scene_s < min_scene_s:
        max_scene_s = min_scene_s
    return clamp_float(float(value), min_scene_s, max_scene_s)


def _now_iso() -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    except Exception:
        return ""


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

    # Preferred: app.py compat wrapper provides client.videos.content(video_id)
    try:
        data = client.videos.content(str(content_id))
        return _write_bytes_to_file(data, out_path)
    except Exception:
        pass

    # Fallbacks for other SDK shapes
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

    Conservative Sora call:
      resp = client.videos.generate(...)
      then downloads returned content id.
    """
    out_dir = ensure_dir(out_dir)
    brand_slug = safe_slug(spec.brand_name or "brand", max_len=40)

    intro_s = int(intro_seconds or seconds or 8)
    outro_s = int(outro_seconds or seconds or 8)
    intro_s = clamp_int(intro_s, 2, 20)
    outro_s = clamp_int(outro_s, 2, 20)

    base_style = (spec.visual_style or "").strip()
    if not base_style:
        base_style = (
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

    zoom_strength is expected ~[1.00, 1.20] from app.py.
    """
    frames = max(1, int(math.ceil(duration_s * fps)))

    zoom_target = clamp_float(float(zoom_strength), 1.0, 1.25)
    inc = (zoom_target - 1.0) / max(1, frames)
    inc = max(0.00005, min(0.005, inc))  # stable bounds

    zoom_expr = "min(zoom+{0},{1})".format(f"{inc:.6f}", f"{zoom_target:.4f}")
    x_expr = "(iw-{0})/2".format(int(width))
    y_expr = "(ih-{0})/2".format(int(height))

    vf = (
        "scale={w}:{h}:force_original_aspect_ratio=increase,"
        "crop={w}:{h},"
        "zoompan=z='{z}':x='{x}':y='{y}':d={d}:s={w}x{h}:fps={fps}"
    ).format(
        w=int(width),
        h=int(height),
        z=zoom_expr,
        x=x_expr,
        y=y_expr,
        d=int(frames),
        fps=int(fps),
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
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        "{0:.3f}".format(float(duration_s)),
        "-vf",
        vf,
        "-r",
        str(int(fps)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4_path),
    ]
    run_ffmpeg(cmd)
    return out_mp4_path


def _ffmpeg_concat_escape(path_str: str) -> str:
    """
    Escape a path for ffmpeg concat demuxer list file.
    We use single quotes around the path and escape internal single quotes.
    """
    # ffmpeg concat list uses: file '...'
    # To include a single quote inside, close/open and escape: 'foo'\''bar'
    return path_str.replace("'", "'\\''")


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
            esc = _ffmpeg_concat_escape(str(pth))
            lines.append("file '{0}'".format(esc))
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ffmpeg = which_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
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
        a_filter = "apad,atrim=0:{0:.3f}".format(target)
    else:
        a_filter = "apad=pad_dur={0:.3f}".format(end_buffer_s)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter:a",
        a_filter,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        "-shortest",
        "-movflags",
        "+faststart",
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
        zp = Path(zip_bytes_or_path)
        data = zp.read_bytes()
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
        return "CHAPTER {0}".format(int(m.group(2)))
    m2 = re.search(r"\b0*(\d{1,2})\b", n)
    if m2 and ("chapter" in n or "ch_" in n or n.startswith("ch")):
        return "CHAPTER {0}".format(int(m2.group(1)))
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
      - base_name (legacy; kept for compatibility / display)
    """
    audio_by_stem: Dict[str, str] = {}
    for a in audios:
        audio_by_stem[Path(a).stem.lower()] = a

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
            a_match = best

        if not a_match:
            a_match = ""

        if a_match and a_match in used_audio:
            # allow reuse only if nothing else available
            pass
        if a_match:
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

        pairs.append(
            {
                "script_path": str(sp),
                "audio_path": str(a_match),
                "kind_guess": kind,
                "chapter_no": chapter_no,
                "title_guess": title_guess,
                "base_name": stem,  # legacy
            }
        )

    script_stems = {Path(s).stem.lower() for s in scripts}
    for a in audios:
        if Path(a).stem.lower() in script_stems:
            continue
        ap = Path(a)
        kind = _guess_kind_from_name(ap.name)
        chapter_no = _guess_chapter_no(kind)
        pairs.append(
            {
                "script_path": "",
                "audio_path": str(ap),
                "kind_guess": kind,
                "chapter_no": chapter_no,
                "title_guess": "",
                "base_name": ap.stem.lower(),
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
            return "CHAPTER {0}".format(int(n))
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
# Image generation (segment-scoped, FIXED)
# ----------------------------

def _image_size_for_mode(width: int, height: int) -> str:
    if height > width:
        return _DEFAULT_IMAGE_SIZE_916
    return _DEFAULT_IMAGE_SIZE_169


def _make_placeholder_image(path: Path, width: int, height: int) -> None:
    """
    Local fallback if OpenAI image generation fails.
    Creates a plain dark image (no text overlays).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        # Last resort: create a 1x1 PNG (ffmpeg will still scale/crop)
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDAT\x08\xd7c``\x00\x00\x00\x04\x00\x01"
            b"\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return

    img = Image.new("RGB", (max(64, int(width)), max(64, int(height))), color=(12, 12, 12))
    img.save(path, format="PNG")


def _openai_generate_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    """
    Attempts to generate a single image via OpenAI Images API.
    Returns raw image bytes (PNG/JPG), or raises.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available in this environment.")
    client = OpenAI(api_key=str(api_key))

    # Prefer b64_json output; fall back if SDK doesn't support response_format.
    try:
        resp = client.images.generate(
            model=_DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            size=size,
            response_format="b64_json",
        )
    except TypeError:
        resp = client.images.generate(
            model=_DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            size=size,
        )

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
        # We avoid fetching remote URLs here; fail over to placeholder upstream.
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
            "Scene context: {0}".format(snippet)
        )

    return prompts


def _rel_to_extract(extract_dir: Union[str, Path], p: Union[str, Path]) -> str:
    """
    Best-effort path identity relative to extract_dir (stable across temp root).
    """
    try:
        extract_dir = Path(extract_dir).resolve()
        pp = Path(p).resolve()
        return str(pp.relative_to(extract_dir)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _segment_cache_key(extract_dir: Union[str, Path], pair: Dict[str, Any]) -> str:
    """
    FIX: Generate a stable, provably unique cache key per segment.
    We use relative script/audio identity (plus kind/chapter) then hash.
    """
    kind = segment_label(pair)
    ch = pair.get("chapter_no")
    ch_s = "" if ch is None else str(int(ch))

    sp = pair.get("script_path") or ""
    ap = pair.get("audio_path") or ""

    sp_rel = _rel_to_extract(extract_dir, sp) if sp else ""
    ap_rel = _rel_to_extract(extract_dir, ap) if ap else ""

    # Include filenames too for extra human readability
    sp_name = Path(sp).name if sp else ""
    ap_name = Path(ap).name if ap else ""

    raw = "|".join([kind, ch_s, sp_rel, ap_rel, sp_name, ap_name]).strip()
    if not raw:
        raw = json.dumps(pair, sort_keys=True)

    h = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    # Keep dir name short and filesystem-safe
    label = kind.lower().replace(" ", "_")
    return "{0}_{1}".format(label, h)


def _segment_image_dir(extract_dir: Union[str, Path], pair: Dict[str, Any]) -> Path:
    """
    Segment-scoped image directory (FIXED).
    """
    extract_dir = Path(extract_dir)
    key = _segment_cache_key(extract_dir, pair)
    return ensure_dir(extract_dir / "_image_cache" / key)


def _write_image_debug(img_dir: Path, info: Dict[str, Any]) -> None:
    if not _DEBUG_IMAGES:
        return
    try:
        p = img_dir / "debug.json"
        info2 = dict(info)
        info2["updated_at"] = _now_iso()
        p.write_text(json.dumps(info2, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clear_dir_contents(dir_path: Path, patterns: Optional[List[str]] = None) -> None:
    """
    Delete files in a directory (used to avoid reusing cached images unless enabled).
    """
    try:
        if not dir_path.exists():
            return
        if patterns is None:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        for pat in patterns:
            for p in dir_path.glob(pat):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


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

    FIXED behavior:
      - Cache dir is truly segment-unique (hash key).
      - Default: do NOT reuse cached images (prevents “same images across segments”
        and avoids stale reuse during testing).
      - Enable reuse only with env UAPPRESS_REUSE_IMAGES=1.
    """
    img_dir = _segment_image_dir(extract_dir, pair)
    size = _image_size_for_mode(int(width), int(height))
    max_scenes = max(1, int(max_scenes))

    if not _REUSE_IMAGES:
        # Ensure we never accidentally reuse old images during testing.
        _clear_dir_contents(img_dir)

    existing = sorted([p for p in img_dir.glob("*.png") if p.is_file()])
    if existing and _REUSE_IMAGES:
        _write_image_debug(
            img_dir,
            {
                "mode": "reuse",
                "cache_dir": str(img_dir),
                "segment_label": segment_label(pair),
                "script_path": str(pair.get("script_path") or ""),
                "audio_path": str(pair.get("audio_path") or ""),
                "image_count": len(existing),
                "size": size,
                "max_scenes": max_scenes,
            },
        )
        return existing[:max_scenes]

    script_path = str(pair.get("script_path") or "")
    script_text = _read_text_preview(script_path, max_chars=12000) if script_path else ""
    prompts = _build_scene_prompts_from_script(script_text, max_scenes=max_scenes) if script_text else []

    if not prompts:
        prompts = [
            "Cinematic documentary still, restrained investigative tone, no text, no logos, no subtitles. "
            "UAP investigation mood: night sky, distant lights, radar room, airbase perimeter, forest trail."
        ] * max(3, min(8, max_scenes))

    _write_image_debug(
        img_dir,
        {
            "mode": "generate",
            "cache_dir": str(img_dir),
            "segment_label": segment_label(pair),
            "script_path": script_path,
            "audio_path": str(pair.get("audio_path") or ""),
            "size": size,
            "max_scenes": max_scenes,
            "prompt_count": len(prompts),
            "prompt_preview": prompts[:2],
            "reuse_enabled": _REUSE_IMAGES,
        },
    )

    images: List[Path] = []
    for i, prompt in enumerate(prompts[:max_scenes], start=1):
        out = img_dir / "scene_{0:03d}.png".format(i)
        try:
            data = _openai_generate_image_bytes(str(api_key), prompt, size)
            out.write_bytes(data)
        except Exception:
            _make_placeholder_image(out, int(width), int(height))
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
    Exact call signature app.py uses.

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

    # Determine target scene count based on audio length and clamp window
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
        scene_seconds = [per for _ in images]
    else:
        per = audio_seconds / float(len(images))
        per = clamp_scene_seconds(per, float(min_s), float(max_s))
        scene_seconds = [per for _ in images]

    seg_slug = safe_slug(str(pair.get("title_guess") or segment_label(pair) or "segment"), max_len=48)
    clips_dir = ensure_dir(out_path_p.parent / "_{0}_clips".format(seg_slug))

    clip_paths: List[Path] = []
    for idx, (img, dur) in enumerate(zip(images, scene_seconds), start=1):
        clip_out = clips_dir / "scene_{0:03d}.mp4".format(idx)
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

    concat_path = out_path_p.parent / "_{0}_concat.mp4".format(seg_slug)
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
