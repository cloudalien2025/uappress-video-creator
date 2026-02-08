# video_pipeline.py
# GODMODE video engine for UAPpress
#
# Guarantees:
# - Import safe
# - Deterministic ZIP pairing (script + mp3)
# - Audio duration is AUTHORITY (never cut narration)
# - Scenes auto-computed from audio (covers full duration + small buffer)
# - Shorts: fill the frame, crop if necessary, NEVER pad
# - Subtitles: deterministic SRT from script (no AI), burn-in optional, sane styling
# - Cache that actually caches (persistent under /tmp by default)
# - Output validated (size + streams + duration)
# - Optional DigitalOcean Spaces upload (boto3 S3 compatible)
#
# Cost discipline:
# - NO GPT calls for scene prompts or subtitles (deterministic only)
# - OpenAI used ONLY for images
# - Aggressive reuse of images + clips across reruns

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class Segment:
    label: str
    script_path: str
    audio_path: str


@dataclass(frozen=True)
class SpacesConfig:
    bucket: str
    region: str = "nyc3"
    endpoint: Optional[str] = None  # e.g. "https://nyc3.digitaloceanspaces.com"
    access_key: str = ""
    secret_key: str = ""
    public_read: bool = True
    prefix: str = ""  # optional key prefix


# -----------------------------
# Persistent cache root
# -----------------------------
def cache_root() -> Path:
    # Streamlit Cloud allows /tmp; use it for real caching across reruns.
    # You can override for local/dev: export UAPPRESS_CACHE_DIR=/path
    root = Path(os.environ.get("UAPPRESS_CACHE_DIR", "/tmp/uappress_cache")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


# -----------------------------
# Utilities
# -----------------------------
def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "segment"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ffmpeg_exe() -> str:
    return os.environ.get("FFMPEG", "ffmpeg")


def ffprobe_exe() -> str:
    return os.environ.get("FFPROBE", "ffprobe")


def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def _quote_path(p: Union[str, Path]) -> str:
    # for ffmpeg filters
    return str(p).replace("\\", "/").replace(":", "\\:")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# -----------------------------
# ZIP discovery + pairing
# -----------------------------
def find_files(root: Union[str, Path]) -> Tuple[List[Path], List[Path]]:
    root = Path(root)
    scripts: List[Path] = []
    audios: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".txt":
            scripts.append(p)
        elif ext in (".mp3", ".wav", ".m4a", ".aac"):
            audios.append(p)
    scripts.sort()
    audios.sort()
    return scripts, audios


def _stem_key(p: Path) -> str:
    return re.sub(r"\s+", " ", p.stem).strip().lower()


def pair_segments(scripts: List[Path], audios: List[Path]) -> List[Segment]:
    """
    Deterministic pairing:
    - Prefer exact stem match: intro.txt ↔ intro.mp3
    - Otherwise if only one audio+script, pair them
    - Otherwise pair by sorted order (stable)
    """
    s_map = {_stem_key(p): p for p in scripts}
    a_map = {_stem_key(p): p for p in audios}

    pairs: List[Segment] = []

    common = sorted(set(s_map.keys()) & set(a_map.keys()))
    for k in common:
        pairs.append(Segment(label=k, script_path=str(s_map[k]), audio_path=str(a_map[k])))

    used_s = {Path(seg.script_path) for seg in pairs}
    used_a = {Path(seg.audio_path) for seg in pairs}
    rem_s = [p for p in scripts if p not in used_s]
    rem_a = [p for p in audios if p not in used_a]

    if not pairs and len(scripts) == 1 and len(audios) == 1:
        sp, ap = scripts[0], audios[0]
        return [Segment(label=_stem_key(sp), script_path=str(sp), audio_path=str(ap))]

    for sp, ap in zip(sorted(rem_s), sorted(rem_a)):
        pairs.append(Segment(label=_stem_key(sp), script_path=str(sp), audio_path=str(ap)))

    return pairs


# -----------------------------
# Duration (Audio is authority)
# -----------------------------
def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    cmd = [
        ffprobe_exe(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = _run(cmd, check=True).stdout
        data = json.loads(out) if out else {}
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        return max(0.0, dur)
    except Exception:
        return 0.0


def _decode_duration_seconds(path: Union[str, Path]) -> float:
    ff = ffmpeg_exe()
    proc = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "info", "-i", str(path), "-vn", "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    text = proc.stderr or ""
    matches = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not matches:
        return 0.0
    hh, mm, ss = matches[-1]
    try:
        return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    except Exception:
        return 0.0


def audio_duration_seconds(path: Union[str, Path]) -> float:
    p = Path(path)
    d = float(ffprobe_duration_seconds(p))
    try:
        sz = p.stat().st_size
    except Exception:
        sz = 0

    if d <= 0.0 or d < 15.0:
        d2 = float(_decode_duration_seconds(p))
        if d2 > 0.0:
            return d2

    if sz > 250_000 and d < 20.0:
        d2 = float(_decode_duration_seconds(p))
        if d2 > d * 1.2:
            return d2

    return d


# -----------------------------
# Scene plan (never truncate audio)
# -----------------------------
def _compute_scene_plan(
    *,
    audio_duration: float,
    max_scenes: int,
    target_scene_seconds: float,
) -> Tuple[int, float, bool]:
    """
    Returns (n_scenes, seconds_per_scene, cap_overridden_upward)
    Guarantees: n_scenes * sec_per_scene >= audio_duration
    If user cap would truncate, override upward.
    """
    dur = max(0.5, float(audio_duration))
    cap = max(1, int(max_scenes))
    target = max(0.5, float(target_scene_seconds))

    # target-based initial scene count
    n = max(1, int(round(dur / target)))

    overridden = False
    if n > cap:
        cap = n
        overridden = True

    n = max(1, min(cap, n))
    sec = dur / n

    # prevent unreadably-fast flicker
    min_sec = 1.0
    if sec < min_sec:
        n = max(1, int(math.floor(dur / min_sec)))
        sec = dur / n

    # final coverage guard
    if n * sec + 1e-3 < dur:
        n = int(math.ceil(dur / sec))
        sec = dur / n
        if n > cap:
            overridden = True

    return int(n), float(sec), bool(overridden)


# -----------------------------
# OpenAI image generation (robust + correct endpoints)
# -----------------------------
def _openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")



def _openai_safe_size(w: int, h: int) -> str:
    # OpenAI Images supports limited sizes; choose closest and upscale via ffmpeg
    # Cost discipline: never fail due to size
    if w >= 1024 and h >= 1024:
        return "1024x1024"
    if w >= 512 and h >= 512:
        return "512x512"
    return "256x256"

def _openai_image_generate_bytes(

    *,
    prompt: str,
    api_key: str,
    size: str,
    timeout_s: int = 120,
    max_retries: int = 4,
) -> bytes:
    """
    OpenAI Images: uses /v1/images/generations (primary).
    Fixes the 404 shown in the UI when code incorrectly hits /v1/images.
    - Only falls back to /v1/images if /generations returns 404.
    - Retries on 429/5xx with backoff.
    """
    model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
    base = _openai_base_url()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "prompt": prompt, "size": size, "n": 1}

    endpoints = [f"{base}/v1/images/generations", f"{base}/v1/images"]  # fallback only if first is 404

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        for idx, url in enumerate(endpoints):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

                # If primary endpoint not found, allow fallback; otherwise raise.
                if r.status_code == 404 and idx == 0:
                    continue

                if r.status_code in (429, 500, 502, 503, 504):
                    # transient; retry
                    raise RuntimeError(f"OpenAI transient {r.status_code}: {r.text[:200]}")

                r.raise_for_status()
                j = r.json()

                if "data" not in j or not j["data"]:
                    raise RuntimeError("OpenAI image response missing data")

                d0 = j["data"][0]
                if d0.get("b64_json"):
                    return base64.b64decode(d0["b64_json"])
                if d0.get("url"):
                    rr = requests.get(d0["url"], timeout=timeout_s)
                    rr.raise_for_status()
                    return rr.content

                raise RuntimeError("OpenAI image response missing url/b64")
            except Exception as e:
                last_err = e
                # If primary was 404 we already continued; otherwise break to retry/backoff
                break

        # backoff before next retry
        time.sleep(0.6 * (2 ** attempt))

    raise RuntimeError(f"OpenAI image generation failed: {last_err}")


# -----------------------------
# Cache paths (real caching)
# -----------------------------
def _image_cache_path(size: str, key: str) -> Path:
    return cache_root() / "images" / size / f"{key}.png"


def _clip_cache_path(w: int, h: int, fps: int, key: str) -> Path:
    return cache_root() / "clips" / f"{w}x{h}_{fps}" / f"{key}.mp4"


def _ensure_image_for_scene(*, api_key: str, prompt: str, size: str) -> Path:
    key = _sha1(prompt + "|" + size)
    out = _image_cache_path(size, key)
    if out.exists() and out.stat().st_size > 10_000:
        return out

    img_bytes = _openai_image_generate_bytes(prompt=prompt, api_key=api_key, size=size)
    if not img_bytes or len(img_bytes) < 10_000:
        raise RuntimeError("OpenAI returned invalid image bytes")
    _atomic_write_bytes(out, img_bytes)
    return out


# -----------------------------
# Prompts (script-driven, retention-minded, deterministic)
# -----------------------------
def _scene_prompts_from_script(script: str, n_scenes: int, vertical: bool) -> List[str]:
    """
    Deterministic prompt derivation:
    - Split script into n chunks
    - House style ensures consistency + retention framing
    - NO extra token burn
    """
    text = re.sub(r"\s+", " ", (script or "").strip())
    if not text:
        text = "A cinematic documentary scene."

    L = len(text)
    step = max(1, L // max(1, n_scenes))
    chunks: List[str] = []
    for i in range(n_scenes):
        a = i * step
        b = (i + 1) * step if i < n_scenes - 1 else L
        chunks.append(text[a:b].strip())

    framing = (
        "vertical 9:16, fill-frame composition, subject dominates frame, strong foreground, shallow depth of field"
        if vertical
        else "wide 16:9, cinematic composition, strong foreground, shallow depth of field"
    )

    base = (
        "Cinematic photoreal documentary still, natural film grain, dramatic but realistic lighting, high detail, "
        "documentary realism, no text, no captions, no logos, no watermarks. "
        f"{framing}. "
    )

    return [base + f"Scene {i+1}: {c}" for i, c in enumerate(chunks)]


# -----------------------------
# FFmpeg video building
# -----------------------------
def _vf_fill_frame(w: int, h: int) -> str:
    # Fill frame, crop if necessary, never pad.
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"


def _clamp_sub_style(style: Dict[str, Any], frame_h: int) -> Dict[str, Any]:
    s = dict(style or {})

    # Font clamp: 2.8%–5.0% of frame height
    min_fs = max(18, int(frame_h * 0.028))
    max_fs = max(min_fs, int(frame_h * 0.050))
    fs = int(s.get("font_size", min_fs))
    s["font_size"] = int(max(min_fs, min(max_fs, fs)))

    # Margin clamp: 1%–12% of frame height
    mv = int(s.get("margin_v", int(frame_h * 0.05)))
    mv = max(int(frame_h * 0.01), min(int(frame_h * 0.12), mv))
    s["margin_v"] = int(mv)

    s["alignment"] = 2        # bottom-center
    s["border_style"] = int(s.get("border_style", 3))  # boxed
    return s


def _vf_subtitles(srt_path: Path, style: Dict[str, Any]) -> str:
    srt_esc = _quote_path(srt_path)

    font = style.get("font_name", "DejaVu Sans")
    font_size = int(style.get("font_size", 28))
    outline = int(style.get("outline", 2))
    shadow = int(style.get("shadow", 1))
    border_style = int(style.get("border_style", 1))
    alignment = int(style.get("alignment", 2))
    margin_v = int(style.get("margin_v", 48))

    force = (
        f"FontName={font},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&HAA000000,"
        f"Outline={outline},"
        f"Shadow={shadow},"
        f"BorderStyle={border_style},"
        f"Alignment={alignment},"
        f"MarginV={margin_v}"
    )
    return f"subtitles='{srt_esc}':force_style='{force}'"


def _make_image_clip_cached(
    *,
    image_path: Path,
    w: int,
    h: int,
    fps: int,
    duration: float,
) -> Path:
    """
    Creates (or reuses) a still-image clip for duration seconds.
    Caches by (image sha1 + w/h/fps + duration_ms bucket).
    """
    dur_ms = int(round(max(0.1, float(duration)) * 1000.0))
    # quantize to 10ms buckets for stable keys
    dur_ms = int(round(dur_ms / 10.0) * 10)
    key = _sha1(f"{image_path.name}|{w}x{h}|{fps}|{dur_ms}")
    out = _clip_cache_path(w, h, fps, key)
    if out.exists() and out.stat().st_size > 50_000:
        return out

    vf = _vf_fill_frame(w, h)
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-loglevel", "error",
        "-loop", "1",
        "-i", str(image_path),
        "-t", f"{dur_ms/1000.0:.3f}",
        "-vf", vf,
        "-r", str(int(fps)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out),
    ]
    _run(cmd, check=True)

    if out.exists() and out.stat().st_size > 50_000:
        return out

    raise RuntimeError("Clip render produced invalid output")


def _concat_clips(clips: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = out_path.parent / f"concat_{_sha1(str(out_path))}.txt"
    lst.write_text("\n".join([f"file {shlex.quote(str(c))}" for c in clips]) + "\n", encoding="utf-8")

    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(lst),
        "-c", "copy",
        str(out_path),
    ]
    _run(cmd, check=True)


def _mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd, check=True)


def _burn_subtitles(video_in: Path, srt_path: Path, out_path: Path, *, w: int, h: int, style: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    style = _clamp_sub_style(style or {}, h)

    vf = _vf_fill_frame(w, h)
    vf2 = vf + "," + _vf_subtitles(srt_path, style)

    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-y",
        "-loglevel", "error",
        "-i", str(video_in),
        "-vf", vf2,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd, check=True)


# -----------------------------
# Subtitles (deterministic SRT)
# -----------------------------
def _clean_script_for_subs(text: str) -> str:
    # Remove backslashes and normalize whitespace.
    t = (text or "").replace("\\", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t).strip()
    return t


def _format_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}".replace(".", ",")


def make_srt_from_script(script_text: str, duration: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = _clean_script_for_subs(script_text)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        txt = " "

    parts = re.split(r"(?<=[\.\!\?\:;])\s+", txt)
    lines: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > 64:
            cut = p.rfind(" ", 0, 64)
            if cut <= 0:
                cut = 64
            lines.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            lines.append(p)

    if not lines:
        lines = [txt[:64]]

    words_per_line = [max(1, len(l.split())) for l in lines]
    total_words = sum(words_per_line)
    dur = max(0.5, float(duration))

    t = 0.0
    blocks: List[str] = []
    for i, (line, wc) in enumerate(zip(lines, words_per_line), 1):
        frac = wc / total_words if total_words > 0 else 1.0 / len(lines)
        seg = max(0.9, dur * frac)
        t2 = min(dur, t + seg)
        if t2 - t < 0.5:
            t2 = min(dur, t + 0.5)

        blocks.append(str(i))
        blocks.append(f"{_format_ts(t)} --> {_format_ts(t2)}")
        blocks.append(line)
        blocks.append("")
        t = t2
        if t >= dur:
            break

    out_path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    return out_path


# -----------------------------
# Validation (no silent failures)
# -----------------------------
def _ffprobe_streams(path: Path) -> Dict[str, Any]:
    cmd = [
        ffprobe_exe(),
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(path),
    ]
    out = _run(cmd, check=True).stdout
    return json.loads(out) if out else {}


def validate_mp4(path: Union[str, Path], *, require_audio: bool = True, min_bytes: int = 50_000) -> Tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "missing"
    if p.stat().st_size < min_bytes:
        return False, f"too small ({p.stat().st_size} bytes)"
    try:
        info = _ffprobe_streams(p)
        streams = info.get("streams", [])
        has_v = any(s.get("codec_type") == "video" for s in streams)
        has_a = any(s.get("codec_type") == "audio" for s in streams)
        dur = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        if not has_v:
            return False, "no video stream"
        if require_audio and not has_a:
            return False, "no audio stream"
        if dur <= 0.1:
            return False, "zero duration"
        return True, "ok"
    except Exception as e:
        return False, f"ffprobe failed: {e}"


# -----------------------------
# Spaces upload
# -----------------------------
def validate_spaces_config(cfg: SpacesConfig) -> Tuple[bool, str]:
    if not cfg.bucket:
        return False, "bucket is empty"
    if not cfg.access_key or not cfg.secret_key:
        return False, "missing access/secret key"
    if not cfg.region:
        return False, "region is empty"
    return True, "ok"


def _spaces_client(cfg: SpacesConfig):
    import boto3
    from botocore.client import Config as BotoConfig

    endpoint = cfg.endpoint or f"https://{cfg.region}.digitaloceanspaces.com"
    s3 = boto3.client(
        "s3",
        region_name=cfg.region,
        endpoint_url=endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )
    return s3, endpoint


def upload_to_spaces(file_path: Path, cfg: SpacesConfig) -> str:
    ok, msg = validate_spaces_config(cfg)
    if not ok:
        raise RuntimeError(msg)

    ok2, msg2 = validate_mp4(file_path, require_audio=True)
    if not ok2:
        raise RuntimeError(f"Refusing upload; invalid MP4: {msg2}")

    s3, _endpoint = _spaces_client(cfg)

    prefix = (cfg.prefix or "").strip().strip("/")
    key_name = file_path.name
    key = f"{prefix}/{key_name}" if prefix else key_name

    extra = {"ACL": "public-read"} if cfg.public_read else {}
    s3.upload_file(str(file_path), cfg.bucket, key, ExtraArgs=(extra or None))

    return f"https://{cfg.bucket}.{cfg.region}.digitaloceanspaces.com/{key}"


# -----------------------------
# Main render
# -----------------------------
def render_segment_mp4(
    *,
    segment: Segment,
    out_dir: Union[str, Path],
    openai_api_key: str,
    width: int,
    height: int,
    fps: int,
    vertical: bool,
    target_scene_seconds: float,
    max_scenes: int,
    burn_subtitles: bool,
    subtitle_style: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Returns (mp4_path, metadata).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = Path(segment.script_path)
    ap = Path(segment.audio_path)

    if not sp.exists():
        raise FileNotFoundError(f"Script missing: {sp}")
    if not ap.exists():
        raise FileNotFoundError(f"Audio missing: {ap}")

    script_text = sp.read_text(encoding="utf-8", errors="ignore")

    # 1) AUDIO duration (truth)
    dur = float(audio_duration_seconds(ap))
    if dur <= 0.5:
        raise RuntimeError("Audio duration could not be determined")

    # Strict buffer: 0.50–0.75 seconds
    end_buffer = float(os.environ.get("UAPPRESS_END_BUFFER", "0.60"))
    end_buffer = min(0.75, max(0.50, end_buffer))
    dur_plus = dur + end_buffer

    # 2) scene plan to COVER dur_plus (never truncate)
    n_scenes, sec_per, cap_overridden = _compute_scene_plan(
        audio_duration=dur_plus,
        max_scenes=int(max_scenes),
        target_scene_seconds=float(target_scene_seconds),
    )

    # 3) output path
    label = _safe_slug(segment.label)
    base = f"{label}_{width}x{height}_{fps}fps"
    final_mp4 = out_dir / f"{base}.mp4"

    if final_mp4.exists() and not overwrite:
        ok, msg = validate_mp4(final_mp4, require_audio=True)
        if ok:
            return final_mp4, {
                "audio_duration": dur,
                "audio_duration_plus_buffer": dur_plus,
                "end_buffer": end_buffer,
                "scenes": n_scenes,
                "sec_per_scene": sec_per,
                "burn_subtitles": bool(burn_subtitles),
                "cached_final": True,
                "cap_overridden": cap_overridden,
            }

    def _p(x: float) -> None:
        if progress_cb:
            try:
                progress_cb(float(x))
            except Exception:
                pass

    _p(0.02)

    # 4) prompts (deterministic)
    prompts_all = _scene_prompts_from_script(script_text, n_scenes, vertical)
    image_budget = int(os.environ.get('UAPPRESS_IMAGE_BUDGET', '6'))
    prompts = prompts_all[:max(1, min(len(prompts_all), image_budget))]
