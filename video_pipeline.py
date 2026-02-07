# video_pipeline.py
"""
UAPpress — Video Pipeline (GODMODE)

Immutable pipeline:
1) Script -> scene prompts (deterministic, low-token)
2) Images (OpenAI images API) cached per segment UID
3) Concat images into a real H.264 MP4 (no black video)
4) Mux narration audio (AAC) -> final MP4
5) Optional burn-in subtitles (from script) via libass
6) Validate output (size, duration, streams)

Design:
- CPU-first
- Streamlit Cloud-safe (no ffprobe required; fallback parser)
- Cache the expensive steps
- Never upload invalid artifacts (validation helper exposed)
"""

from __future__ import annotations

import base64
import hashlib
import math
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional OpenAI SDK
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

_SCRIPT_EXTS = {".txt", ".md"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}

_DEFAULT_IMAGE_MODEL = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1")
_DEFAULT_IMAGE_SIZE_169 = os.environ.get("UAPPRESS_IMAGE_SIZE_169", "1536x1024")
_DEFAULT_IMAGE_SIZE_916 = os.environ.get("UAPPRESS_IMAGE_SIZE_916", "1024x1536")
_IMAGE_CACHE_DIRNAME = os.environ.get("UAPPRESS_IMAGE_CACHE_DIRNAME", "_images_cache_v2")

# House style: realistic + cinematic documentary stills
_VISUAL_STYLE = (
    "Photorealistic investigative documentary still. Natural light, realistic materials, "
    "cinematic composition, subtle film grain, credible staging. "
    "No text, no captions, no logos, no watermarks."
)

_SHOTS = [
    "Wide establishing frame, eye-level, stable tripod feel.",
    "Medium documentary framing, natural perspective.",
    "Close detail shot, shallow depth of field.",
    "Wide with foreground framing (fence/window/doorway).",
    "Medium from behind (observer POV), faces not visible.",
    "Close-up of objects (papers, radios, maps), practical lighting.",
]


# ----------------------------
# Small utilities
# ----------------------------

def safe_rmtree(path: Union[str, Path]) -> None:
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


def safe_slug(text: str, max_len: int = 60) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "untitled"
    return (s[:max_len].strip("-") or "untitled")


def _stable_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[: max(6, int(n))]


def ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def which_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe") or None


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-4000:]
        raise RuntimeError(f"Command failed (code {proc.returncode}). Tail:\n{tail}")


def run_ffmpeg(cmd: List[str]) -> None:
    if "-y" not in cmd:
        cmd = cmd[:1] + ["-y"] + cmd[1:]
    run_cmd(cmd)


def _parse_duration_from_ffmpeg_stderr(stderr_text: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text or "")
    if not m:
        return 0.0
    hh = float(m.group(1))
    mm = float(m.group(2))
    ss = float(m.group(3))
    return hh * 3600.0 + mm * 60.0 + ss



def _decode_duration_seconds(path: Union[str, Path]) -> float:
    """Accurate duration by decoding to null and parsing final time=. Slower but robust."""
    ff = ffmpeg_exe()
    p = str(path)
    proc = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "info", "-i", p, "-vn", "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    text = proc.stderr or ""
    # find last time=XX:YY:ZZ.xx
    matches = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not matches:
        return 0.0
    hh, mm, ss = matches[-1]
    try:
        return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    except Exception:
        return 0.0


def audio_duration_seconds(path: Union[str, Path]) -> float:
    """Best-effort duration with safety: if metadata duration looks suspiciously short, decode to confirm."""
    p = Path(path)
    d = float(ffprobe_duration_seconds(p))
    try:
        sz = p.stat().st_size
    except Exception:
        sz = 0
    # Heuristic: large file but tiny reported duration => wrong header; decode for truth.
    if d > 0 and sz > 250_000 and d < 12.0:
        d2 = float(_decode_duration_seconds(p))
        if d2 > d * 1.5:
            return d2
    if d <= 0.0 and sz > 0:
        d2 = float(_decode_duration_seconds(p))
        if d2 > 0:
            return d2
    return d

def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    p = str(path)
    fp = which_ffprobe()
    if fp:
        proc = subprocess.run(
            [fp, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", p],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode == 0:
            try:
                return float((proc.stdout or "").strip())
            except Exception:
                pass

    ff = ffmpeg_exe()
    proc2 = subprocess.run([ff, "-hide_banner", "-i", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return _parse_duration_from_ffmpeg_stderr(proc2.stderr or "")


def has_audio_stream(path: Union[str, Path]) -> bool:
    p = str(path)
    fp = which_ffprobe()
    if fp:
        proc = subprocess.run([fp, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                               "-of", "default=nw=1:nk=1", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    # fallback: naive grep
    ff = ffmpeg_exe()
    proc2 = subprocess.run([ff, "-hide_banner", "-i", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return "Audio:" in (proc2.stderr or "")


def has_video_stream(path: Union[str, Path]) -> bool:
    p = str(path)
    fp = which_ffprobe()
    if fp:
        proc = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                               "-of", "default=nw=1:nk=1", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    ff = ffmpeg_exe()
    proc2 = subprocess.run([ff, "-hide_banner", "-i", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return "Video:" in (proc2.stderr or "")


def validate_mp4(path: Union[str, Path], min_bytes: int = 50_000) -> Tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "missing"
    if p.stat().st_size < int(min_bytes):
        return False, "too_small"
    dur = float(ffprobe_duration_seconds(p))
    if dur <= 0.05:
        return False, "zero_duration"
    if not has_video_stream(p):
        return False, "no_video_stream"
    # audio is expected in this app
    if not has_audio_stream(p):
        return False, "no_audio_stream"
    return True, "ok"


def default_max_scenes(is_vertical: bool) -> int:
    return 10 if is_vertical else 16


# ----------------------------
# ZIP handling + pairing
# ----------------------------

def extract_zip_to_temp(zip_bytes_or_path: Union[bytes, str, Path]) -> Tuple[str, str]:
    workdir = tempfile.mkdtemp(prefix="uappress_vc_")
    extract_dir = str(Path(workdir) / "extracted")
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(zip_bytes_or_path, (str, Path)):
        data = Path(zip_bytes_or_path).read_bytes()
    else:
        data = bytes(zip_bytes_or_path)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)

    return workdir, extract_dir


def find_files(extract_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
    ed = Path(extract_dir)
    scripts: List[str] = []
    audios: List[str] = []
    for p in ed.rglob("*"):
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
    return "SEGMENT"


def segment_label(pair: Dict[str, Any]) -> str:
    kind = str(pair.get("kind_guess") or "").strip().upper()
    if kind.startswith("CHAPTER"):
        return kind
    if kind in ("INTRO", "OUTRO"):
        return kind
    combo = (str(pair.get("script_path") or "") + " " + str(pair.get("audio_path") or "")).lower()
    if "intro" in combo:
        return "INTRO"
    if "outro" in combo:
        return "OUTRO"
    return "SEGMENT"


def pair_segments(scripts: List[str], audios: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministic pairing:
    - Prefer exact stem match
    - Else assign next unused audio (stable order)
    Emits uid for cache safety.
    """
    audio_by_stem: Dict[str, str] = {Path(a).stem.lower(): a for a in audios}
    used_audio: set[str] = set()
    pairs: List[Dict[str, Any]] = []

    for s in scripts:
        sp = Path(s)
        stem = sp.stem.lower()
        a_match = audio_by_stem.get(stem, "")

        if (not a_match) or (a_match in used_audio):
            for a in audios:
                if a not in used_audio:
                    a_match = a
                    break

        if a_match:
            used_audio.add(a_match)

        kind = _guess_kind_from_name(sp.name)

        seed = f"script:{str(sp.resolve())}|audio:{str(Path(a_match).resolve()) if a_match else ''}"
        uid = _stable_hash(seed, 14)

        pairs.append(
            {
                "script_path": str(sp),
                "audio_path": str(a_match),
                "kind_guess": kind,
                "uid": uid,
            }
        )

    return pairs


# ----------------------------
# Script -> prompts (deterministic)
# ----------------------------

def _sanitize(text: str, max_chars: int = 320) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def _split_into_beats(script_text: str) -> List[str]:
    # Prefer paragraph beats; fallback to sentences.
    paras = [p.strip() for p in re.split(r"\n\s*\n+", script_text or "") if p.strip()]
    if paras:
        return paras
    s = (script_text or "").strip()
    if not s:
        return []
    sents = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", s) if p.strip()]
    return sents if sents else [s]


def _choose_beats(script_text: str, n: int) -> List[str]:
    beats = _split_into_beats(script_text)
    if not beats:
        return []
    n = max(1, int(n))
    if len(beats) <= n:
        return beats
    step = len(beats) / float(n)
    idxs = [min(len(beats) - 1, int(i * step)) for i in range(n)]
    out: List[str] = []
    last = -1
    for i in idxs:
        if i == last:
            i = min(len(beats) - 1, i + 1)
        out.append(beats[i])
        last = i
    return out[:n]


def _image_size_for_wh(width: int, height: int) -> str:
    return _DEFAULT_IMAGE_SIZE_916 if height > width else _DEFAULT_IMAGE_SIZE_169


def _build_prompts(script_text: str, n_scenes: int) -> List[str]:
    beats = _choose_beats(script_text, n_scenes)
    prompts: List[str] = []
    for i, b in enumerate(beats):
        snippet = _sanitize(b, 280)
        cam = _SHOTS[i % len(_SHOTS)]
        prompts.append(
            "Create ONE image.\n"
            f"STYLE: {_VISUAL_STYLE}\n"
            f"CAMERA: {cam}\n"
            f"SCENE: {snippet}\n"
        )
    return prompts


# ----------------------------
# OpenAI images (cached)
# ----------------------------

def _openai_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Add 'openai' to requirements.txt.")
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("OpenAI API key is empty.")
    client = OpenAI(api_key=api_key)

    last_err: Optional[Exception] = None
    for attempt in range(1, 3):
        try:
            r = client.images.generate(model=_DEFAULT_IMAGE_MODEL, prompt=prompt, size=size)
            b64 = r.data[0].b64_json
            if not b64:
                raise RuntimeError("Images API returned empty b64_json.")
            return base64.b64decode(b64)
        except Exception as e:
            last_err = e
            time.sleep(0.25 * attempt)
    raise RuntimeError(f"Images API failed: {last_err}")


def _segment_cache_dir(extract_dir: str, uid: str) -> Path:
    root = Path(extract_dir) / _IMAGE_CACHE_DIRNAME / uid
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_good_image(path: Path, min_bytes: int = 30_000) -> bool:
    try:
        return path.exists() and path.stat().st_size >= int(min_bytes)
    except Exception:
        return False


def _ensure_images(*, extract_dir: str, uid: str, api_key: str, prompts: List[str], size: str) -> List[Path]:
    cache = _segment_cache_dir(extract_dir, uid)
    out: List[Path] = []
    for i, prompt in enumerate(prompts, start=1):
        # prompt-hash to avoid "polluted cache" if prompts change
        ph = _stable_hash(prompt, 10)
        img_path = cache / f"img_{i:02d}_{ph}.png"
        if _is_good_image(img_path):
            out.append(img_path)
            continue
        data = _openai_image_bytes(api_key, prompt, size)
        img_path.write_bytes(data)
        out.append(img_path)
    return out


# ----------------------------
# Subtitles (from script) -> SRT
# ----------------------------

def _clean_sub_text(s: str) -> str:
    s = (s or "").replace("\\", "")  # kill backslashes (recurring bug)
    s = s.replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _to_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000.0))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_srt_from_script(script_text: str, audio_duration: float, out_srt: Union[str, Path]) -> str:
    """
    Deterministic subtitles: split script into short readable chunks and spread across duration.
    """
    text = _clean_sub_text(script_text)
    if not text:
        Path(out_srt).write_text("", encoding="utf-8")
        return str(out_srt)

    # Chunk by sentences, then pack into ~48 chars lines
    sents = [t.strip() for t in re.split(r"(?<=[\.\?\!])\s+", text) if t.strip()]
    if not sents:
        sents = [text]

    chunks: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= 92:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)

    # prevent tiny durations per caption
    n = max(1, len(chunks))
    dur = max(0.5, float(audio_duration))
    per = max(1.2, min(5.5, dur / n))  # readable
    # if duration is short, reduce chunk count
    if dur / n < 1.0:
        n2 = max(1, int(dur / 1.0))
        chunks = chunks[:n2]
        n = len(chunks)
        per = dur / n

    t = 0.0
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        start = t
        end = min(dur, t + per)
        # clamp final
        if i == n:
            end = dur
        lines.append(str(i))
        lines.append(f"{_to_srt_timestamp(start)} --> {_to_srt_timestamp(end)}")
        lines.append(c)
        lines.append("")
        t = end

    Path(out_srt).write_text("\n".join(lines), encoding="utf-8")
    return str(out_srt)


# ----------------------------
# FFmpeg assembly
# ----------------------------

def _concat_list_file(images: List[Path], seconds_per: float) -> Path:
    seconds_per = max(0.25, float(seconds_per))
    p = Path(tempfile.mkstemp(prefix="uappress_concat_", suffix=".txt")[1])
    lines: List[str] = []
    for img in images:
        lines.append(f"file '{img.as_posix()}'")
        lines.append(f"duration {seconds_per:.3f}")
    # repeat last file (concat demuxer requirement)
    lines.append(f"file '{images[-1].as_posix()}'")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _vf_fill_frame(width: int, height: int) -> str:
    # FILL FRAME ALWAYS: scale up + crop, never pad.
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p"


def _vf_subtitles(srt_path: Union[str, Path], style: Dict[str, Any]) -> str:
    # libass style string
    font = style.get("font_name", "DejaVu Sans")
    size = int(style.get("font_size", 40))
    outline = int(style.get("outline", 2))
    shadow = int(style.get("shadow", 1))
    border_style = int(style.get("border_style", 3))
    alignment = int(style.get("alignment", 2))
    margin_v = int(style.get("margin_v", 40))

    # ASS color format: &HAABBGGRR (we use white + semi-black box)
    force = (
        f"FontName={font},"
        f"FontSize={size},"
        f"BorderStyle={border_style},"
        f"Outline={outline},"
        f"Shadow={shadow},"
        f"PrimaryColour=&H00FFFFFF,"
        f"BackColour=&H80000000,"
        f"Alignment={alignment},"
        f"MarginV={margin_v}"
    )
    # IMPORTANT: path must be escaped for ffmpeg filtergraph
    p = str(Path(srt_path).as_posix()).replace(":", r"\:")
    return f"subtitles='{p}':force_style='{force}'"


def _compute_scene_plan(
    *,
    audio_duration: float,
    max_scenes: int,
    target_scene_seconds: float,
    min_scene_seconds: float,
    max_scene_seconds: float,
) -> Tuple[int, float]:
    """
    Decide scene count and seconds per scene to cover full audio duration,
    respecting min/max per scene, capped by max_scenes.
    """
    dur = max(0.5, float(audio_duration))
    max_scenes = max(1, int(max_scenes))
    target = max(0.5, float(target_scene_seconds))
    min_s = max(0.25, float(min_scene_seconds))
    max_s = max(min_s, float(max_scene_seconds))

    # initial guess
    n = int(round(dur / target))
    n = max(1, min(max_scenes, n))
    sec = dur / n

    # adjust to satisfy sec within [min_s, max_s]
    if sec < min_s:
        n = int(max(1, min(max_scenes, dur / min_s)))
        sec = dur / n
    elif sec > max_s:
        n = int(max(1, min(max_scenes, dur / max_s)))
        sec = dur / n

    # final clamp
    sec = max(min_s, min(max_s, sec))
    return int(n), float(sec)


def render_segment_mp4(
    *,
    pair: Dict[str, Any],
    extract_dir: str,
    out_path: str,
    api_key: str,
    fps: int = 24,
    width: int = 1080,
    height: int = 1920,
    zoom_strength: float = 0.0,  # reserved (0.0 means off)
    max_scenes: int = 14,
    target_scene_seconds: float = 4.0,
    min_scene_seconds: float = 1.0,
    max_scene_seconds: float = 12.0,
    burn_subtitles: bool = True,
    subtitle_style: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render one segment MP4 with real frames + narration audio.
    Optional: burn-in subtitles generated from script text.
    """
    sp = Path(pair.get("script_path") or "")
    ap = Path(pair.get("audio_path") or "")
    uid = str(pair.get("uid") or _stable_hash(str(sp) + "|" + str(ap), 14))

    if not sp.exists():
        raise RuntimeError(f"Missing script file: {sp}")
    if not ap.exists():
        raise RuntimeError(f"Missing audio file: {ap}")

    script_text = sp.read_text(encoding="utf-8", errors="ignore").strip()
    if not script_text:
        raise RuntimeError(f"Empty script file: {sp.name}")

    dur = float(audio_duration_seconds(ap))
    if dur <= 0.1:
        raise RuntimeError(f"Audio duration invalid: {ap.name}")

    is_vertical = height > width
    fps = int(max(15, min(60, int(fps))))

    # scene plan
    n_scenes, sec_per = _compute_scene_plan(
        audio_duration=dur,
        max_scenes=int(max_scenes or default_max_scenes(is_vertical)),
        target_scene_seconds=float(target_scene_seconds),
        min_scene_seconds=float(min_scene_seconds),
        max_scene_seconds=float(max_scene_seconds),
    )

    prompts = _build_prompts(script_text, n_scenes)
    if not prompts:
        raise RuntimeError("Unable to derive prompts from script.")

    size = _image_size_for_wh(width, height)
    imgs = _ensure_images(extract_dir=extract_dir, uid=uid, api_key=api_key, prompts=prompts, size=size)

    # concat list
    lst = _concat_list_file(imgs, sec_per)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # intermediate video without subtitles (but with audio)
    tmp_video = outp.with_suffix(".tmp.mp4")
    tmp_video2 = outp.with_suffix(".tmp2.mp4")

    vf = _vf_fill_frame(width, height)

    # Step A: images -> video stream (no audio) with explicit timebase
    cmd_video = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(lst),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "stillimage",
        "-crf", "19" if not is_vertical else "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_video),
    ]
    run_ffmpeg(cmd_video)

    # Step B: mux narration audio (guarantee audio exists)
    # -shortest ensures video trims to audio if image-track runs longer.
    cmd_mux = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(tmp_video),
        "-i", str(ap),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(tmp_video2),
    ]
    run_ffmpeg(cmd_mux)

    # Optional Step C: burn subtitles (keeps audio)
    final_in = tmp_video2
    if burn_subtitles:
        srt_path = outp.with_suffix(".srt")
        make_srt_from_script(script_text, dur, srt_path)
        style = subtitle_style or {"font_name": "DejaVu Sans", "font_size": int(max(30, height * 0.047)), "margin_v": int(max(24, height * 0.06))}
        vf2 = _vf_subtitles(srt_path, style)
        cmd_sub = [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(tmp_video2),
            "-vf", vf2,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19" if not is_vertical else "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(outp),
        ]
        run_ffmpeg(cmd_sub)
    else:
        shutil.move(str(tmp_video2), str(outp))

    # Cleanup temps
    for p in [tmp_video, tmp_video2]:
        try:
            if Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass
    try:
        if Path(lst).exists():
            Path(lst).unlink()
    except Exception:
        pass

    ok, why = validate_mp4(outp)
    if not ok:
        raise RuntimeError(f"Render produced invalid MP4 ({why}).")


