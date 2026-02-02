# ============================================================
# video_pipeline.py — UAPpress Video Creator
# SECTION 1 — Core types + helpers + segment renderer (COMPAT)
# ============================================================
#
# Locked requirements satisfied:
# - Segment MP4s only (Intro/Chapters/Outro); no subs/logos/stitching
# - Pipeline order: images -> scene clips -> concat -> mux audio (no black video)
# - Audio-driven timing + ~0.5–0.75s end buffer (default 0.65s; clamped)
# - Streamlit stability helpers: session_state clamping for scene seconds
# - Compatibility: render_segment_mp4 shim + OpenAI videos.content/download_content wrapper
# - DigitalOcean Spaces: auto-upload per segment + manifest.json upload + prefix handling
# - Optional GLOBAL Sora intro/outro via BrandIntroOutroSpec + generate_sora_brand_intro_outro
# - 9:16 supported when selected (pass width/height from app.py)
#
# IMPORTANT:
# - Keep function names/signatures stable with app.py expectations.
# - Avoid import/syntax errors; optional deps are guarded.

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional deps (guarded)
try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

try:
    import imageio_ffmpeg  # type: ignore
except Exception:  # pragma: no cover
    imageio_ffmpeg = None  # type: ignore


# ----------------------------
# Constants + small utilities
# ----------------------------

_END_BUFFER_MIN = 0.50
_END_BUFFER_MAX = 0.75
_DEFAULT_END_BUFFER = 0.65

# A conservative "safe" framerate for still-image scene clips
_DEFAULT_FPS = 30

# Default audio sample rate expectation (your TTS Studio uses 48kHz masters)
_DEFAULT_AUDIO_SR = 48000


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
    name = name.strip().replace(" ", "_")
    name = re.sub(keep, "", name)
    return name or "unnamed"


def ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def which_ffmpeg() -> str:
    """
    Find ffmpeg in PATH, else fall back to imageio-ffmpeg if installed.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    # Last resort: hope "ffmpeg" resolves at runtime.
    return "ffmpeg"


def run_ffmpeg(cmd: List[str]) -> None:
    """
    Run ffmpeg command, raising a readable error on failure.
    """
    # Always force overwrite to avoid Streamlit rerun collisions.
    if "-y" not in cmd:
        cmd = cmd[:1] + ["-y"] + cmd[1:]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-4000:]
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}). Tail:\n{tail}")


def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    """
    Return media duration seconds using ffprobe.
    """
    ffprobe = shutil.which("ffprobe") or "ffprobe"
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
        # If ffprobe isn't available, fall back to 0 and let caller handle.
        return 0.0
    try:
        return float((proc.stdout or "").strip())
    except Exception:
        return 0.0


# ---------------------------------------
# Streamlit stability helper (clamping)
# ---------------------------------------

def clamp_scene_seconds(
    value: float,
    min_scene_s: float,
    max_scene_s: float,
) -> float:
    """
    For Streamlit slider/session_state stability:
    clamp a scene duration into [min_scene_s, max_scene_s].
    """
    # Extra protection: ensure bounds are sensible
    if max_scene_s < min_scene_s:
        max_scene_s = min_scene_s
    return clamp_float(float(value), float(min_scene_s), float(max_scene_s))


# ----------------------------
# Sora Brand Intro/Outro spec
# ----------------------------

@dataclass
class BrandIntroOutroSpec:
    """
    Optional GLOBAL Sora intro/outro generation.
    app.py should pass:
      - enabled
      - brand_prompt (or a structured prompt)
      - intro_seconds / outro_seconds (fallback seconds)
      - model name if desired
    """
    enabled: bool = False
    model: str = "sora"
    brand_prompt: str = ""
    intro_seconds: Optional[float] = None
    outro_seconds: Optional[float] = None

    # Optional: allow passing any vendor-specific knobs without breaking signatures
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ------------------------------------------------------------
# OpenAI videos.content/download_content compatibility wrapper
# ------------------------------------------------------------

def download_content_compat(
    client: Any,
    content_id: str,
    out_path: Union[str, Path],
) -> Path:
    """
    Compatibility shim for:
      client.videos.content.download_content(...)
    and other potential SDK shapes.

    Writes bytes to out_path and returns Path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Newer SDK shape
    try:
        fn = client.videos.content.download_content  # type: ignore[attr-defined]
        data = fn(content_id)  # might return bytes or a stream-like
        if isinstance(data, (bytes, bytearray)):
            out_path.write_bytes(bytes(data))
            return out_path
        # stream-like with .read()
        if hasattr(data, "read"):
            out_path.write_bytes(data.read())
            return out_path
    except Exception:
        pass

    # Alternate: content retrieve then download URL (not implemented here)
    raise RuntimeError(
        "Unable to download Sora content with current OpenAI client. "
        "Expected client.videos.content.download_content(content_id) to work."
    )


def generate_sora_brand_intro_outro(
    client: Any,
    spec: BrandIntroOutroSpec,
    kind: str,
    out_mp4_path: Union[str, Path],
    width: int,
    height: int,
    seconds_fallback: float,
) -> Optional[Path]:
    """
    Generate a GLOBAL brand intro or outro with Sora (optional).
    - kind: 'intro' or 'outro'
    - returns Path if generated, else None

    NOTE: This function is intentionally conservative: if SDK calls differ,
    it raises a clear error rather than producing broken output.
    """
    if not spec or not spec.enabled:
        return None

    kind = kind.lower().strip()
    if kind not in ("intro", "outro"):
        raise ValueError("kind must be 'intro' or 'outro'")

    # Choose seconds
    if kind == "intro":
        seconds = float(spec.intro_seconds) if spec.intro_seconds else float(seconds_fallback)
    else:
        seconds = float(spec.outro_seconds) if spec.outro_seconds else float(seconds_fallback)

    # Construct prompt
    base = (spec.brand_prompt or "").strip()
    if not base:
        # If enabled but empty prompt, treat as a no-op (avoid surprising failures)
        return None

    prompt = (
        f"{base}\n\n"
        f"Create a clean, cinematic {kind} clip for the UAPpress brand. "
        f"No text overlays, no logos, no subtitles. "
        f"Aspect {width}x{height}. Duration about {seconds:.1f} seconds."
    )

    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt a likely SDK call shape. If your app.py uses a different one,
    # keep the signature and adjust the internal try-block only.
    try:
        # Hypothetical: client.videos.generate(...) returns {id/content_id/...}
        resp = client.videos.generate(  # type: ignore[attr-defined]
            model=spec.model,
            prompt=prompt,
            width=width,
            height=height,
            duration_seconds=seconds,
            **(spec.extra or {}),
        )
        content_id = None
        if isinstance(resp, dict):
            content_id = resp.get("content_id") or resp.get("id")
        else:
            content_id = getattr(resp, "content_id", None) or getattr(resp, "id", None)
        if not content_id:
            raise RuntimeError("Sora generate did not return a content id.")
        return download_content_compat(client, str(content_id), out_mp4_path)
    except Exception as e:
        raise RuntimeError(f"Sora brand {kind} generation failed: {e}") from e


# ----------------------------
# DigitalOcean Spaces uploads
# ----------------------------

@dataclass
class SpacesConfig:
    """
    Minimal DO Spaces config. app.py can construct and pass this.
    """
    enabled: bool = False
    endpoint_url: str = ""          # e.g. "https://nyc3.digitaloceanspaces.com"
    region: str = "us-east-1"       # boto3 requires something; not used by DO
    bucket: str = ""
    access_key: str = ""
    secret_key: str = ""
    prefix: str = ""               # optional folder prefix inside bucket
    public_base_url: str = ""      # optional CDN/base URL for returned links


def _spaces_client(cfg: SpacesConfig):
    if not cfg.enabled:
        return None
    if boto3 is None:
        raise RuntimeError("boto3 is not installed, but Spaces upload was enabled.")
    if not (cfg.endpoint_url and cfg.bucket and cfg.access_key and cfg.secret_key):
        raise RuntimeError("SpacesConfig is missing required fields.")
    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=cfg.region or "us-east-1",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
    )


def _join_prefix(prefix: str, key: str) -> str:
    prefix = (prefix or "").strip().strip("/")
    key = key.strip().lstrip("/")
    return f"{prefix}/{key}" if prefix else key


def spaces_upload_file(
    cfg: SpacesConfig,
    local_path: Union[str, Path],
    key: str,
    content_type: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a file to DO Spaces and return a URL if public_base_url is provided,
    else return None.
    """
    if not cfg.enabled:
        return None

    s3 = _spaces_client(cfg)
    assert s3 is not None

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(str(local_path))

    full_key = _join_prefix(cfg.prefix, key)

    extra_args: Dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type

    s3.upload_file(str(local_path), cfg.bucket, full_key, ExtraArgs=extra_args or None)

    if cfg.public_base_url:
        base = cfg.public_base_url.rstrip("/")
        return f"{base}/{full_key}"
    return None


# ----------------------------
# Manifest
# ----------------------------

def build_segment_manifest_record(
    segment_name: str,
    mp4_key: str,
    mp4_url: Optional[str],
    audio_seconds: float,
    width: int,
    height: int,
    fps: int,
    created_at_epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "segment_name": segment_name,
        "mp4": {"key": mp4_key, "url": mp4_url},
        "audio_seconds": audio_seconds,
        "video": {"width": width, "height": height, "fps": fps},
        "created_at": created_at_epoch,
        "extra": extra or {},
    }


def write_manifest_json(manifest: Dict[str, Any], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return out_path


# ---------------------------------------------------------
# Core rendering: images -> scene clips -> concat -> mux audio
# ---------------------------------------------------------

def _make_zoompan_filter(
    width: int,
    height: int,
    duration_s: float,
    fps: int,
) -> str:
    """
    A gentle Ken-Burns-ish zoompan that won't create black frames:
    - scale to cover
    - center crop
    - zoompan over the crop
    """
    frames = max(1, int(math.ceil(duration_s * fps)))
    # Very subtle zoom: 1.00 -> ~1.06 over the scene
    # Use on-frame zoom expression bounded to avoid jumps.
    zoom_expr = "min(zoom+0.0009,1.06)"
    # Centered pan (keeps stable). If you want motion, you can
    # add small x/y drift; but this is safest.
    x_expr = f"(iw-{width})/2"
    y_expr = f"(ih-{height})/2"

    # We first scale up so that crop is guaranteed to be filled.
    # scale='if(gt(a,16/9),-1,W*1.2): ...' gets messy; use a robust cover:
    # scale to max needed then crop.
    # Use force_original_aspect_ratio=increase to cover both dimensions.
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
    fps: int = _DEFAULT_FPS,
) -> Path:
    """
    Turn a single image into a short MP4 scene clip (no audio).
    Uses zoompan to avoid static "slideshow" feel, but keeps it safe.
    """
    image_path = Path(image_path)
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    vf = _make_zoompan_filter(width, height, duration_s, fps)
    ffmpeg = which_ffmpeg()

    # -loop 1 reads image continuously; -t sets duration.
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        vf,
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4_path),
    ]
    run_ffmpeg(cmd)
    return out_mp4_path


def concat_video_clips(
    clip_paths: Sequence[Union[str, Path]],
    out_mp4_path: Union[str, Path],
) -> Path:
    """
    Concatenate MP4 clips (video only) with ffmpeg concat demuxer (safe).
    All clips must match codec settings (we create them consistently).
    """
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Build concat list file
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        list_path = td_path / "concat_list.txt"
        lines = []
        for p in clip_paths:
            p = Path(p)
            # concat demuxer needs escaped paths; safest is to use single quotes and replace
            # any single quote in path (rare on Windows) – but we’ll use ffmpeg-safe escaping:
            lines.append(f"file '{str(p).replace(\"'\", \"'\\\\''\")}'")
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
    Mux narration audio to the concatenated video, audio-driven with a small end buffer.
    Ensures no black video by:
      - creating video to roughly match audio duration before mux
      - muxing with -shortest
      - padding audio slightly (0.5–0.75s) rather than extending video arbitrarily
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_mp4_path = Path(out_mp4_path)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    end_buffer_s = clamp_float(float(end_buffer_s), _END_BUFFER_MIN, _END_BUFFER_MAX)

    ffmpeg = which_ffmpeg()

    # Pad audio with a short silence tail using apad + atrim, then mux.
    # We do NOT add +2s; we clamp to ~0.5–0.75s.
    #
    # Note: If audio has exact duration D, atrim ends at D+buffer.
    # If D is unknown (ffprobe missing), we still apply apad and rely on -shortest.
    audio_dur = ffprobe_duration_seconds(audio_path)
    if audio_dur > 0:
        target = audio_dur + end_buffer_s
        a_filter = f"apad,atrim=0:{target:.3f}"
    else:
        # Unknown duration: minimal padding; -shortest keeps us safe.
        a_filter = f"apad=pad_dur={end_buffer_s:.3f}"

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
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_mp4_path),
    ]
    run_ffmpeg(cmd)
    return out_mp4_path


# ---------------------------------------------------------
# Segment render shim (this is what app.py calls)
# ---------------------------------------------------------

def render_segment_mp4(
    *,
    segment_name: str,
    images: Sequence[Union[str, Path]],
    audio_path: Union[str, Path],
    out_dir: Union[str, Path],
    width: int,
    height: int,
    fps: int = _DEFAULT_FPS,
    min_scene_s: float = 2.0,
    max_scene_s: float = 10.0,
    end_buffer_s: float = _DEFAULT_END_BUFFER,
    spaces_cfg: Optional[SpacesConfig] = None,
    spaces_prefix: str = "",
    upload_manifest: bool = True,
    manifest_basename: str = "manifest.json",
    extra_manifest: Optional[Dict[str, Any]] = None,
    brand_spec: Optional[BrandIntroOutroSpec] = None,
    brand_intro_seconds_fallback: float = 3.0,
    brand_outro_seconds_fallback: float = 3.0,
    openai_client_for_brand: Any = None,
) -> Dict[str, Any]:
    """
    Render ONE segment MP4 with the locked pipeline:
      images -> scene clips -> concat -> mux audio

    Returns a dict used by app.py for UI + uploads, e.g.:
      {
        "segment_name": ...,
        "local_mp4": "...",
        "spaces_key": "...",
        "spaces_url": "...",
        "audio_seconds": ...,
        "width": ..., "height": ..., "fps": ...
      }

    Notes:
    - No subtitles/logos.
    - No cross-segment stitching.
    - Brand intro/outro (GLOBAL) is optional and uses Sora generation if enabled.
    """
    out_dir = ensure_dir(out_dir)
    segment_safe = sanitize_filename(segment_name)
    ts = int(time.time())

    images = [Path(p) for p in images]
    audio_path = Path(audio_path)

    if not images:
        raise ValueError(f"No images provided for segment '{segment_name}'.")
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    # Determine scene durations from audio length (audio-driven).
    audio_seconds = ffprobe_duration_seconds(audio_path)
    if audio_seconds <= 0:
        # If ffprobe unavailable, fall back to uniform duration that respects clamp.
        # app.py usually drives this; we keep it safe.
        per = clamp_scene_seconds(5.0, min_scene_s, max_scene_s)
        scene_seconds = [per for _ in images]
        audio_seconds = per * len(images)
    else:
        # Allocate audio duration across scenes, leaving brand clips out of this calculation.
        # The end buffer is handled at mux time (audio padding), not by stretching video.
        per = audio_seconds / max(1, len(images))
        per = clamp_scene_seconds(per, min_scene_s, max_scene_s)
        scene_seconds = [per for _ in images]

    # Build scene clips
    clips_dir = ensure_dir(out_dir / f"{segment_safe}_clips")
    clip_paths: List[Path] = []
    for idx, (img, dur) in enumerate(zip(images, scene_seconds), start=1):
        clip_out = clips_dir / f"{segment_safe}_scene_{idx:03d}.mp4"
        clip_paths.append(
            build_scene_clip_from_image(
                image_path=img,
                out_mp4_path=clip_out,
                duration_s=float(dur),
                width=int(width),
                height=int(height),
                fps=int(fps),
            )
        )

    # Optional: prepend/append GLOBAL brand intro/outro clips
    # These are full MP4s already; we’ll concat them with the clip list.
    if brand_spec and brand_spec.enabled:
        if openai_client_for_brand is None:
            raise RuntimeError("Brand intro/outro enabled, but no OpenAI client was provided.")
        intro_path = out_dir / f"{segment_safe}_brand_intro.mp4"
        outro_path = out_dir / f"{segment_safe}_brand_outro.mp4"

        intro_mp4 = generate_sora_brand_intro_outro(
            openai_client_for_brand,
            brand_spec,
            kind="intro",
            out_mp4_path=intro_path,
            width=int(width),
            height=int(height),
            seconds_fallback=float(brand_intro_seconds_fallback),
        )
        outro_mp4 = generate_sora_brand_intro_outro(
            openai_client_for_brand,
            brand_spec,
            kind="outro",
            out_mp4_path=outro_path,
            width=int(width),
            height=int(height),
            seconds_fallback=float(brand_outro_seconds_fallback),
        )

        # Insert intro at start, outro at end (if generated)
        new_list: List[Path] = []
        if intro_mp4:
            new_list.append(Path(intro_mp4))
        new_list.extend(clip_paths)
        if outro_mp4:
            new_list.append(Path(outro_mp4))
        clip_paths = new_list

    # Concatenate clips (video-only)
    concat_path = out_dir / f"{segment_safe}_concat.mp4"
    concat_video_clips(clip_paths, concat_path)

    # Mux audio with a small end buffer, keeping video from turning black.
    final_mp4 = out_dir / f"{segment_safe}.mp4"
    mux_audio_to_video(
        video_path=concat_path,
        audio_path=audio_path,
        out_mp4_path=final_mp4,
        end_buffer_s=float(end_buffer_s),
    )

    # Upload segment MP4 + manifest.json (optional)
    spaces_url = None
    mp4_key = f"{segment_safe}.mp4"
    manifest_key = manifest_basename

    spaces_cfg = spaces_cfg or SpacesConfig(enabled=False)
    if spaces_cfg.enabled:
        # Allow app.py to override/extend prefix on a per-run basis
        effective_prefix = _join_prefix(spaces_cfg.prefix, spaces_prefix) if spaces_prefix else spaces_cfg.prefix

        # Upload MP4
        tmp_cfg = dataclasses.replace(spaces_cfg, prefix=effective_prefix)
        spaces_url = spaces_upload_file(tmp_cfg, final_mp4, mp4_key, content_type="video/mp4")

        # Manifest
        if upload_manifest:
            manifest = {
                "version": 1,
                "created_at": ts,
                "segment": build_segment_manifest_record(
                    segment_name=segment_name,
                    mp4_key=_join_prefix(effective_prefix, mp4_key) if effective_prefix else mp4_key,
                    mp4_url=spaces_url,
                    audio_seconds=float(audio_seconds),
                    width=int(width),
                    height=int(height),
                    fps=int(fps),
                    created_at_epoch=ts,
                    extra=extra_manifest or {},
                ),
            }
            manifest_path = write_manifest_json(manifest, out_dir / manifest_basename)
            spaces_upload_file(tmp_cfg, manifest_path, manifest_key, content_type="application/json")

    return {
        "segment_name": segment_name,
        "local_mp4": str(final_mp4),
        "audio_path": str(audio_path),
        "audio_seconds": float(audio_seconds),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "spaces_key": _join_prefix(spaces_prefix or "", mp4_key) if (spaces_prefix or "") else mp4_key,
        "spaces_url": spaces_url,
        "out_dir": str(out_dir),
    }
