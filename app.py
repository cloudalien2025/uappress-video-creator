# app.py
# 🛸 UAPpress — Documentary MP4 Studio (FAST + FINAL)
# Features:
# 1) Fast Mode / Final Mode toggle (draft vs cinematic settings)
# 2) Automatic image caching (generate once, reuse forever)
# 3) Parallel chapter rendering + final concat

import os
import re
import json
import time
import shutil
import zipfile
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
from openai import OpenAI


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary MP4 Studio (Fast + Final)")
st.caption("Fast drafts in minutes. Final cinematic export when ready.")


# ----------------------------
# Helpers
# ----------------------------
def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-\.\s]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def run(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run command and raise on error with readable output."""
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stdout}")


def ffprobe_duration_sec(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def which_or_hint(bin_name: str) -> None:
    if shutil.which(bin_name) is None:
        raise RuntimeError(
            f"Missing dependency: {bin_name}\n"
            "Install ffmpeg and ensure it's on PATH."
        )


def ffmpeg_concat_escape(path: str) -> str:
    """
    Escape a file path for FFmpeg concat demuxer file list.
    FFmpeg expects: file '<path>'
    If path contains single quotes, escape them in the standard shell-safe way.
    """
    return path.replace("'", "'\\''")


# ----------------------------
# Render settings
# ----------------------------
@dataclass
class RenderProfile:
    name: str
    images_per_chapter: int
    enable_ken_burns: bool
    image_min_seconds: int
    image_max_seconds: int
    crf: int
    preset: str
    fps: int
    audio_bitrate: str
    video_codec: str = "libx264"
    pix_fmt: str = "yuv420p"


FAST_PROFILE = RenderProfile(
    name="Fast (Draft)",
    images_per_chapter=3,
    enable_ken_burns=False,     # BIG speed win
    image_min_seconds=60,
    image_max_seconds=120,
    crf=28,
    preset="ultrafast",         # BIG speed win
    fps=30,
    audio_bitrate="160k",
)

FINAL_PROFILE = RenderProfile(
    name="Final (Cinematic)",
    images_per_chapter=6,
    enable_ken_burns=True,
    image_min_seconds=30,
    image_max_seconds=75,
    crf=18,
    preset="slow",
    fps=30,
    audio_bitrate="192k",
)


# ----------------------------
# Image caching + generation
# ----------------------------
def cached_image_path(cache_dir: str, prompt: str, size: str) -> str:
    key = sha1(json.dumps({"prompt": prompt, "size": size}, ensure_ascii=False))
    return os.path.join(cache_dir, f"{key}_{size}.png")


def generate_image_openai(client: OpenAI, prompt: str, size: str) -> bytes:
    """
    Uses OpenAI Images API.
    If your account requires a different model, change model= below.
    """
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size
    )
    import base64
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)


def get_or_make_image(client: OpenAI, cache_dir: str, prompt: str, size: str) -> str:
    ensure_dir(cache_dir)
    out_path = cached_image_path(cache_dir, prompt, size)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    img_bytes = generate_image_openai(client, prompt, size)
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path


# ----------------------------
# FFmpeg clip building
# ----------------------------
def build_image_clip(
    image_path: str,
    out_path: str,
    duration: float,
    profile: RenderProfile,
    target_w: int,
    target_h: int
) -> None:
    """
    Creates a video segment from a still image.
    Fast mode uses simple loop (very fast).
    Final mode uses subtle zoompan (Ken Burns-like).
    """
    which_or_hint("ffmpeg")

    # Scale to target while preserving aspect, pad to fill
    base_vf = (
        f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    )

    if not profile.enable_ken_burns:
        # Very fast: loop image for duration
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-t", f"{duration:.3f}",
            "-vf", base_vf,
            "-r", str(profile.fps),
            "-c:v", profile.video_codec,
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-pix_fmt", profile.pix_fmt,
            "-an",
            out_path
        ]
        run(cmd)
        return

    # Ken Burns via zoompan — subtle and smooth
    frames = max(1, int(duration * profile.fps))
    zoom_expr = "if(lte(on,1),1.0, min(1.08, zoom+0.0008))"
    x_expr = "(iw-(iw/zoom))/2"
    y_expr = "(ih-(ih/zoom))/2"

    vf = (
        f"{base_vf},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={target_w}x{target_h}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-r", str(profile.fps),
        "-c:v", profile.video_codec,
        "-preset", profile.preset,
        "-crf", str(profile.crf),
        "-pix_fmt", profile.pix_fmt,
        "-an",
        out_path
    ]
    run(cmd)


def concat_video_segments(segments: List[str], out_path: str) -> None:
    which_or_hint("ffmpeg")
    with tempfile.TemporaryDirectory() as td:
        list_file = os.path.join(td, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for s in segments:
                safe_path = ffmpeg_concat_escape(s)
                f.write(f"file '{safe_path}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            out_path
        ]
        run(cmd)


def mux_audio(video_path: str, audio_path: str, out_path: str, profile: RenderProfile) -> None:
    which_or_hint("ffmpeg")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", profile.audio_bitrate,
        "-shortest",
        out_path
    ]
    run(cmd)


# ----------------------------
# Chapter rendering (single)
# ----------------------------
def pick_durations(total_sec: float, n: int, min_s: int, max_s: int) -> List[float]:
    """Spread total duration across n images, clamped to [min_s, max_s]."""
    if n <= 0:
        return []

    base = total_sec / n
    base_clamped = min(max(base, min_s), max_s)
    durations = [base_clamped] * n
    current = sum(durations)

    if current < total_sec:
        remaining = total_sec - current
        i = 0
        while remaining > 0.001:
            add = min(max_s - durations[i], remaining)
            if add > 0:
                durations[i] += add
                remaining -= add
            i = (i + 1) % n
            if all(abs(max_s - d) < 1e-6 for d in durations):
                break
    elif current > total_sec:
        excess = current - total_sec
        i = 0
        while excess > 0.001:
            sub = min(durations[i] - min_s, excess)
            if sub > 0:
                durations[i] -= sub
                excess -= sub
            i = (i + 1) % n
            if all(abs(d - min_s) < 1e-6 for d in durations):
                break

    scale = total_sec / max(0.001, sum(durations))
    return [d * scale for d in durations]


def render_chapter(
    chapter_idx: int,
    chapter_title: str,
    image_prompts: List[str],
    audio_path: str,
    cache_dir: str,
    profile: RenderProfile,
    out_dir: str,
    image_size: str,
    target_w: int,
    target_h: int,
    api_key: str,
) -> str:
    """
    Renders one chapter MP4 and returns output path.
    Safe to call in parallel (no Streamlit calls).
    """
    client = OpenAI(api_key=api_key)
    ensure_dir(out_dir)

    dur = ffprobe_duration_sec(audio_path)
    if dur <= 0:
        raise RuntimeError(f"Audio duration not found or zero: {audio_path}")

    prompts = (image_prompts or [])[: profile.images_per_chapter]
    if len(prompts) < 1:
        prompts = [f"Documentary still image, atmospheric, related to: {chapter_title}"] * profile.images_per_chapter
    if len(prompts) < profile.images_per_chapter:
        prompts = prompts + [prompts[-1]] * (profile.images_per_chapter - len(prompts))

    images = [get_or_make_image(client, cache_dir, p, image_size) for p in prompts]

    durs = pick_durations(
        total_sec=dur,
        n=len(images),
        min_s=profile.image_min_seconds,
        max_s=profile.image_max_seconds
    )

    seg_dir = os.path.join(out_dir, f"chapter_{chapter_idx:02d}_segs")
    ensure_dir(seg_dir)
    segments = []
    for i, (img, seg_dur) in enumerate(zip(images, durs), start=1):
        seg_path = os.path.join(seg_dir, f"seg_{i:02d}.mp4")
        build_image_clip(img, seg_path, seg_dur, profile, target_w, target_h)
        segments.append(seg_path)

    silent_video = os.path.join(out_dir, f"chapter_{chapter_idx:02d}_silent.mp4")
    concat_video_segments(segments, silent_video)

    out_name = safe_filename(f"{chapter_idx:02d}_{chapter_title}") + ".mp4"
    out_path = os.path.join(out_dir, out_name)
    mux_audio(silent_video, audio_path, out_path, profile)
    return out_path


# ----------------------------
# Parallel render + concat
# ----------------------------
def render_all_chapters_parallel(
    chapters: List[Dict],
    cache_dir: str,
    profile: RenderProfile,
    out_dir: str,
    image_size: str,
    target_w: int,
    target_h: int,
    api_key: str,
    max_workers: int
) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ensure_dir(out_dir)
    results = [None] * len(chapters)

    def _job(i: int, ch: Dict) -> Tuple[int, str]:
        return i, render_chapter(
            chapter_idx=i + 1,
            chapter_title=ch.get("title", f"Chapter {i+1}"),
            image_prompts=ch.get("image_prompts", []),
            audio_path=ch["audio_path"],
            cache_dir=cache_dir,
            profile=profile,
            out_dir=out_dir,
            image_size=image_size,
            target_w=target_w,
            target_h=target_h,
            api_key=api_key,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, i, ch) for i, ch in enumerate(chapters)]
        for fut in as_completed(futures):
            i, out_path = fut.result()
            results[i] = out_path

    return results


def concat_final_movie(chapter_mp4s: List[str], out_path: str) -> None:
    which_or_hint("ffmpeg")
    with tempfile.TemporaryDirectory() as td:
        list_file = os.path.join(td, "final_concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in chapter_mp4s:
                safe_path = ffmpeg_concat_escape(p)
                f.write(f"file '{safe_path}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path]
        run(cmd)


# ----------------------------
# UI
# ----------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

profile_choice = st.sidebar.radio("Render Profile", [FAST_PROFILE.name, FINAL_PROFILE.name], index=0)
profile = FAST_PROFILE if profile_choice == FAST_PROFILE.name else FINAL_PROFILE

st.sidebar.markdown("**Parallel Rendering**")
max_workers = st.sidebar.slider("Chapters rendered at the same time", min_value=1, max_value=8, value=4)

st.sidebar.markdown("**Output**")
target_resolution = st.sidebar.selectbox("Resolution", ["1920x1080 (1080p)", "1280x720 (720p)"], index=0)
target_w, target_h = (1920, 1080) if "1920x1080" in target_resolution else (1280, 720)

image_size = st.sidebar.selectbox(
    "AI image size (generation)",
    ["1024x1024", "1536x1024", "1024x1536"],
    index=1
)

cache_dir = st.sidebar.text_input("Image cache folder", value=os.path.join(os.getcwd(), "image_cache"))
out_dir = st.sidebar.text_input("Output folder", value=os.path.join(os.getcwd(), "outputs"))

st.sidebar.info(
    f"**{profile.name}**\n\n"
    f"- Images/chapter: {profile.images_per_chapter}\n"
    f"- Ken Burns: {'ON' if profile.enable_ken_burns else 'OFF'}\n"
    f"- Preset: {profile.preset}\n"
    f"- CRF: {profile.crf}\n"
)

st.markdown("### Chapters")
st.write("Upload your chapter audio files (MP3/WAV/M4A). Optionally provide image prompts per chapter.")

uploaded = st.file_uploader("Upload chapter audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

default_titles = [os.path.splitext(f.name)[0] for f in uploaded] if uploaded else []

chapters: List[Dict] = []
if uploaded:
    st.markdown("#### Chapter Setup")
    for idx, f in enumerate(uploaded, start=1):
        with st.expander(f"Chapter {idx}: {f.name}", expanded=(idx == 1)):
            title = st.text_input(f"Title (Chapter {idx})", value=default_titles[idx - 1], key=f"title_{idx}")
            prompts_raw = st.text_area(
                "Image prompts (one per line) — leave blank to auto-generate from title",
                value="",
                height=120,
                key=f"prompts_{idx}"
            )
            prompts = [p.strip() for p in prompts_raw.splitlines() if p.strip()]
            chapters.append({"title": title, "image_prompts": prompts, "file_obj": f})

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    render_btn = st.button("🎬 Render MP4s (Parallel)", type="primary", disabled=(not uploaded or not api_key))

with col2:
    concat_btn = st.button("🧩 Concat into 1 Full Movie", disabled=(not uploaded))

if "rendered_chapters" not in st.session_state:
    st.session_state.rendered_chapters = []
if "chapter_tempdir" not in st.session_state:
    st.session_state.chapter_tempdir = None


def save_uploads_to_temp(chapters_ui: List[Dict]) -> List[Dict]:
    td = tempfile.mkdtemp(prefix="uappress_")
    st.session_state.chapter_tempdir = td
    prepared = []
    for ch in chapters_ui:
        f = ch["file_obj"]
        audio_path = os.path.join(td, safe_filename(f.name))
        with open(audio_path, "wb") as out:
            out.write(f.read())
        prepared.append({"title": ch["title"], "image_prompts": ch["image_prompts"], "audio_path": audio_path})
    return prepared


if render_btn:
    try:
        which_or_hint("ffmpeg")
        which_or_hint("ffprobe")
        ensure_dir(cache_dir)
        ensure_dir(out_dir)

        st.info("Rendering chapters in parallel…")
        prepared_chapters = save_uploads_to_temp(chapters)

        t0 = time.time()
        with st.spinner("Rendering…"):
            chapter_mp4s = render_all_chapters_parallel(
                chapters=prepared_chapters,
                cache_dir=cache_dir,
                profile=profile,
                out_dir=out_dir,
                image_size=image_size,
                target_w=target_w,
                target_h=target_h,
                api_key=api_key,
                max_workers=max_workers
            )
        elapsed = time.time() - t0
        st.session_state.rendered_chapters = chapter_mp4s

        st.success(f"Done! Rendered {len(chapter_mp4s)} chapter(s) in {elapsed:.1f}s.")
        for p in chapter_mp4s:
            st.write(f"✅ {p}")

        with tempfile.TemporaryDirectory() as tdzip:
            zip_path = os.path.join(tdzip, "chapter_mp4s.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in chapter_mp4s:
                    z.write(p, arcname=os.path.basename(p))
            with open(zip_path, "rb") as f:
                st.download_button("⬇️ Download ZIP of chapter MP4s", data=f, file_name="chapter_mp4s.zip")

    except Exception as e:
        st.error(str(e))


if concat_btn:
    try:
        if not st.session_state.rendered_chapters:
            st.warning("Render chapters first (so there are MP4s to concat).")
        else:
            final_name = safe_filename("UAPpress_Full_Movie") + ".mp4"
            final_path = os.path.join(out_dir, final_name)

            st.info("Concatenating chapters into one movie…")
            with st.spinner("Concatenating…"):
                concat_final_movie(st.session_state.rendered_chapters, final_path)

            st.success(f"Full movie created: {final_path}")
            with open(final_path, "rb") as f:
                st.download_button("⬇️ Download Full Movie MP4", data=f, file_name=final_name)

    except Exception as e:
        st.error(str(e))


st.markdown("---")
st.markdown("### Speed Tips (built into this app)")
st.markdown(
    "- **Fast Mode** disables Ken Burns and uses ultrafast encoding.\n"
    "- **Image caching** means you generate images once per prompt and reuse forever.\n"
    "- **Parallel rendering** renders multiple chapters at the same time, then concatenates.\n"
)
