# video_pipeline.py
# UAPpress Video Pipeline — GODMODE
# Deterministic, Streamlit Cloud-safe, cost-disciplined.
#
# Pipeline:
#   images -> concat video -> mux audio -> (optional) burn subtitles -> validate
#
from __future__ import annotations

import base64
import hashlib
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

# Optional OpenAI SDK (install via requirements.txt)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

_SCRIPT_EXTS = {".txt", ".md"}
_SUB_EXTS = {".srt"}  # keep it simple & deterministic
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}

_DEFAULT_IMAGE_MODEL = os.environ.get("UAPPRESS_IMAGE_MODEL", "gpt-image-1")
_DEFAULT_IMAGE_SIZE_169 = os.environ.get("UAPPRESS_IMAGE_SIZE_169", "1536x1024")
_DEFAULT_IMAGE_SIZE_916 = os.environ.get("UAPPRESS_IMAGE_SIZE_916", "1024x1536")
_IMAGE_CACHE_DIRNAME = os.environ.get("UAPPRESS_IMAGE_CACHE_DIRNAME", "_images_cache_v1")

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


# --------------------- tiny utilities ---------------------
def safe_slug(text: str, max_len: int = 60) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "untitled"
    return (s[:max_len].strip("-") or "untitled")


def _stable_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[: max(6, int(n))]


def _rmtree_safe(p: Union[str, Path]) -> None:
    try:
        shutil.rmtree(str(p), ignore_errors=True)
    except Exception:
        pass


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
    # Always overwrite to avoid stale partials
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

    # Fallback (works even if ffprobe missing)
    ff = ffmpeg_exe()
    proc2 = subprocess.run([ff, "-hide_banner", "-i", p], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return _parse_duration_from_ffmpeg_stderr(proc2.stderr or "")


def _ffprobe_has_audio(path: Union[str, Path]) -> bool:
    fp = which_ffprobe()
    if not fp:
        # Best effort: assume yes after encode unless proven otherwise
        return True
    p = str(path)
    proc = subprocess.run(
        [fp, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "default=nw=1", p],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc.returncode == 0 and "codec_type=audio" in (proc.stdout or "")


def default_max_scenes(is_vertical: bool) -> int:
    return 6 if is_vertical else 12


# --------------------- ZIP discovery & pairing ---------------------
def find_files(extract_dir: Union[str, Path]) -> Tuple[List[str], List[str], List[str]]:
    ed = Path(extract_dir)
    scripts: List[str] = []
    audios: List[str] = []
    subs: List[str] = []
    for p in ed.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _SCRIPT_EXTS:
            scripts.append(str(p))
        elif ext in _AUDIO_EXTS:
            audios.append(str(p))
        elif ext in _SUB_EXTS:
            subs.append(str(p))

    scripts.sort(key=lambda x: Path(x).name.lower())
    audios.sort(key=lambda x: Path(x).name.lower())
    subs.sort(key=lambda x: Path(x).name.lower())
    return scripts, audios, subs


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


def pair_segments(scripts: List[str], audios: List[str], subs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Deterministic pairing (stable, cache-safe):
      1) Prefer exact stem match (script <-> audio <-> srt)
      2) Else assign next unused audio/sub in stable order.
    """
    subs = subs or []
    audio_by_stem: Dict[str, str] = {Path(a).stem.lower(): a for a in audios}
    sub_by_stem: Dict[str, str] = {Path(s).stem.lower(): s for s in subs}

    used_audio: set[str] = set()
    used_sub: set[str] = set()
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

        sub_match = sub_by_stem.get(stem, "")
        if (not sub_match) or (sub_match in used_sub):
            for sub in subs:
                if sub not in used_sub:
                    sub_match = sub
                    break
        if sub_match:
            used_sub.add(sub_match)

        kind = _guess_kind_from_name(sp.name)

        title_guess = ""
        try:
            preview = sp.read_text(encoding="utf-8", errors="ignore").strip()
            for line in preview.splitlines():
                raw = line.strip()
                if not raw:
                    continue
                if raw.startswith("#"):
                    title_guess = raw.lstrip("#").strip()[:120]
                    break
                if raw.lower().startswith("title:"):
                    title_guess = raw.split(":", 1)[1].strip()[:120]
                    break
        except Exception:
            pass

        seed = f"script:{str(sp.resolve())}|audio:{str(Path(a_match).resolve()) if a_match else ''}"
        uid = _stable_hash(seed, 14)

        pairs.append(
            {
                "script_path": str(sp),
                "audio_path": str(a_match),
                "sub_path": str(sub_match) if sub_match else "",
                "kind_guess": kind,
                "title_guess": title_guess,
                "uid": uid,
            }
        )

    return pairs


# --------------------- image generation (cached) ---------------------
def _sanitize(text: str, max_chars: int = 320) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def _choose_beats(script_text: str, n: int) -> List[str]:
    """
    Prefer paragraph beats (double-newline). If not available, fall back to sentence-ish splits.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n+", script_text or "") if p.strip()]
    if not paras:
        paras = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", (script_text or "").strip()) if p.strip()]
    if not paras:
        return []
    n = max(1, int(n))
    if len(paras) <= n:
        return paras
    step = len(paras) / float(n)
    idxs = [min(len(paras) - 1, int(i * step)) for i in range(n)]
    out: List[str] = []
    last = -1
    for i in idxs:
        if i == last:
            i = min(len(paras) - 1, i + 1)
        out.append(paras[i])
        last = i
    return out[:n]


def _image_size_for_wh(width: int, height: int) -> str:
    return _DEFAULT_IMAGE_SIZE_916 if height > width else _DEFAULT_IMAGE_SIZE_169


def _build_prompts(script_text: str, max_scenes: int) -> List[str]:
    beats = _choose_beats(script_text, max_scenes)
    prompts: List[str] = []
    for i, b in enumerate(beats):
        snippet = _sanitize(b, 300)
        cam = _SHOTS[i % len(_SHOTS)]
        prompts.append(
            "Create one image.\n"
            f"STYLE: {_VISUAL_STYLE}\n"
            f"CAMERA: {cam}\n"
            f"SCENE: {snippet}\n"
        )
    return prompts


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


def _is_good_image(path: Path, min_bytes: int = 50_000) -> bool:
    try:
        return path.exists() and path.stat().st_size >= int(min_bytes)
    except Exception:
        return False


def _ensure_images(*, extract_dir: str, uid: str, api_key: str, prompts: List[str], size: str) -> List[Path]:
    out: List[Path] = []
    cache = _segment_cache_dir(extract_dir, uid)
    for i, prompt in enumerate(prompts, start=1):
        img_path = cache / f"img_{i:02d}.png"
        if _is_good_image(img_path):
            out.append(img_path)
            continue
        data = _openai_image_bytes(api_key, prompt, size)
        img_path.write_bytes(data)
        out.append(img_path)
    return out


# --------------------- subtitles (deterministic) ---------------------
def _strip_speaker_tags(line: str) -> str:
    # Remove "NARRATOR:" style prefixes
    m = re.match(r"^\s*[A-Z][A-Z0-9 _-]{1,20}:\s*(.+)$", line.strip())
    return (m.group(1).strip() if m else line.strip())


def _tokenize_words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", re.sub(r"[^\w'\- ]+", " ", text or "").strip()) if w]


def _srt_timestamp(t: float) -> str:
    t = max(0.0, float(t))
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = int(t % 60)
    ms = int(round((t - int(t)) * 1000.0))
    if ms >= 1000:
        ms = 999
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def build_srt_from_script(script_text: str, duration: float, target_words_per_line: int = 6) -> str:
    """
    Cheap & deterministic captioning:
    - Split into short lines
    - Allocate times proportional to word counts across the full duration
    """
    duration = max(0.5, float(duration))
    lines_raw = []
    for ln in (script_text or "").splitlines():
        ln = _strip_speaker_tags(ln)
        if not ln:
            continue
        lines_raw.append(ln)

    joined = " ".join(lines_raw).strip()
    if not joined:
        joined = (script_text or "").strip()

    words = _tokenize_words(joined)
    if not words:
        return ""

    # Build caption lines
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + target_words_per_line]).strip()
        if chunk:
            chunks.append(chunk)
        i += target_words_per_line

    # Timing proportional to words (avoid 0-length)
    total_words = max(1, len(words))
    srt_lines: List[str] = []
    t = 0.0
    idx = 1
    for c in chunks:
        cw = max(1, len(_tokenize_words(c)))
        seg = duration * (cw / total_words)
        seg = max(0.75, min(3.5, seg))  # shorts-friendly
        t2 = min(duration, t + seg)
        srt_lines.append(str(idx))
        srt_lines.append(f"{_srt_timestamp(t)} --> {_srt_timestamp(t2)}")
        srt_lines.append(c)
        srt_lines.append("")
        idx += 1
        t = t2
        if t >= duration - 0.05:
            break

    return "\n".join(srt_lines).strip() + "\n"


def _subtitle_style_force(height: int, mode: str) -> str:
    """
    FFmpeg subtitles filter uses ASS style keys in force_style.
    Keep it clean: bottom-centered, boxed, readable, never clown letters.
    """
    h = int(height)
    # Font size scaled by resolution
    fs = 36 if h <= 720 else 48 if h <= 1080 else 64
    margin_v = 40 if h <= 720 else 60 if h <= 1080 else 80

    if mode == "bold":
        # Bold/high-contrast for noisy backgrounds
        return (
            f"FontName=Arial,FontSize={fs},PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,BackColour=&H80000000,"
            f"BorderStyle=3,Outline=2,Shadow=0,Bold=1,Alignment=2,MarginV={margin_v}"
        )

    # Clean default
    return (
        f"FontName=Arial,FontSize={fs},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BackColour=&H60000000,"
        f"BorderStyle=3,Outline=1,Shadow=0,Bold=0,Alignment=2,MarginV={margin_v}"
    )


def _ffmpeg_filter_path(p: Union[str, Path]) -> str:
    # subtitles filter wants a single string; escape ":" and "\" safely.
    s = str(p)
    s = s.replace("\\", "\\\\").replace(":", "\\:")
    return s


# --------------------- render ---------------------
def _concat_list_file(images: List[Path], seconds_per: float) -> Path:
    seconds_per = max(0.5, float(seconds_per))
    p = Path(tempfile.mkstemp(prefix="uappress_concat_", suffix=".txt")[1])
    lines: List[str] = []
    for img in images:
        lines.append(f"file '{img.as_posix()}'")
        lines.append(f"duration {seconds_per:.3f}")
    lines.append(f"file '{images[-1].as_posix()}'")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _vf_fill_frame(width: int, height: int) -> str:
    # FILL FRAME ALWAYS (crop, never pad)
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p"


def render_segment_mp4(
    *,
    pair: Dict[str, Any],
    extract_dir: str,
    out_path: str,
    api_key: str,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
    max_scenes: int = 12,
    min_scene_seconds: int = 6,
    max_scene_seconds: int = 120,
    burn_subtitles: bool = True,
    subtitle_style: str = "clean",  # "clean" | "bold"
) -> None:
    sp = Path(pair.get("script_path") or "")
    ap = Path(pair.get("audio_path") or "")
    subp = Path(pair.get("sub_path") or "") if (pair.get("sub_path") or "") else None
    uid = str(pair.get("uid") or _stable_hash(str(sp) + "|" + str(ap), 14))

    if not sp.exists():
        raise RuntimeError(f"Missing script file: {sp}")
    if not ap.exists():
        raise RuntimeError(f"Missing audio file: {ap}")

    script_text = sp.read_text(encoding="utf-8", errors="ignore").strip()
    if not script_text:
        raise RuntimeError(f"Empty script file: {sp.name}")

    dur = float(ffprobe_duration_seconds(ap))
    if dur <= 0.1:
        raise RuntimeError(f"Audio duration invalid: {ap.name}")

    is_vertical = height > width
    max_scenes = max(1, int(max_scenes or default_max_scenes(is_vertical)))
    max_scenes = min(60, max_scenes)

    prompts = _build_prompts(script_text, max_scenes)
    if not prompts:
        raise RuntimeError("Unable to derive prompts from script.")

    size = _image_size_for_wh(width, height)
    imgs = _ensure_images(extract_dir=extract_dir, uid=uid, api_key=api_key, prompts=prompts, size=size)

    # Seconds per scene: audio-driven, but bounded
    sec_per = dur / float(max(1, len(imgs)))
    sec_per = max(float(min_scene_seconds), min(float(max_scene_seconds), sec_per))

    lst = _concat_list_file(imgs, sec_per)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    vf = _vf_fill_frame(width, height)

    # Subtitles input (prefer .srt from ZIP; else generate deterministic SRT)
    srt_path: Optional[Path] = None
    if burn_subtitles:
        if subp and subp.exists() and subp.suffix.lower() == ".srt":
            srt_path = subp
        else:
            gen = build_srt_from_script(script_text, dur)
            if gen.strip():
                srt_path = Path(tempfile.mkstemp(prefix="uappress_", suffix=".srt")[1])
                srt_path.write_text(gen, encoding="utf-8")

    if burn_subtitles and srt_path:
        style = _subtitle_style_force(height, "bold" if subtitle_style == "bold" else "clean")
        vf = vf + f",subtitles='{_ffmpeg_filter_path(srt_path)}':force_style='{style}'"

    preset = "veryfast"
    crf = "20" if is_vertical else "19"

    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-r", str(int(fps)),
        "-f", "concat",
        "-safe", "0",
        "-i", str(lst),
        "-i", str(ap),
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", "stillimage",
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(outp),
    ]
    run_ffmpeg(cmd)

    # Validate output
    if not outp.exists() or outp.stat().st_size < 120_000:
        raise RuntimeError("Render produced an invalid/too-small MP4.")

    out_dur = float(ffprobe_duration_seconds(outp))
    if out_dur <= 0.05:
        raise RuntimeError("Render produced zero-duration MP4.")

    # Enforce audio presence (prevents silent uploads)
    if not _ffprobe_has_audio(outp):
        raise RuntimeError("Render produced MP4 without an audio stream.")
