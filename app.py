# app.py — UAPpress ZIP → Segment MP4 Studio (ZIP upload, generate/download per segment)
# - Upload the ZIP from Documentary TTS Studio (scripts + audio)
# - Each segment has its own Generate + Download buttons
# - Builds ONE segment per click (Streamlit-safe)
# - Subtitles are ALWAYS burned into each segment MP4
# - Hard-caps per-scene seconds to avoid huge ffmpeg zoompan spikes
# - Outputs persist in UAPPRESS_CACHE_DIR so reruns keep downloads

from __future__ import annotations

import os
import json
import math
import time
import shutil
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg

from video_pipeline import (
    extract_zip_to_temp,
    find_files,
    read_script_file,
    plan_scenes,
    generate_video_clip,
    transcribe_audio_to_srt,
    write_text,
    safe_slug,
    mux_audio,
    embed_srt_softsubs,
    get_media_duration_seconds,
    reencode_mp4,
)

# ----------------------------
# Safe concat helpers
# ----------------------------

def _ffmpeg_escape_path(p: str) -> str:
    # concat demuxer wants: file '...'
    return (p or "").replace("'", "'\\''")

def safe_concat_mp4s(paths: List[str], out_path: str) -> None:
    if not paths:
        raise ValueError("safe_concat_mp4s: no input paths provided")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"safe_concat_mp4s: missing input file: {p}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in paths:
            f.write(f"file '{_ffmpeg_escape_path(os.path.abspath(p))}'\n")

    try:
        cmd = [
            ffmpeg, "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


# ----------------------------
# Segment pairing (robust intro/outro + chapter matching)
# ----------------------------

def _norm_tokens(s: str) -> set[str]:
    import re
    s = (s or "").lower()
    toks = set(re.split(r"[^a-z0-9]+", s))
    toks.discard("")
    return toks

def _chapter_no_from_name(name: str) -> Optional[int]:
    import re
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

def _is_intro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("intro") or b == "intro" or " intro" in b or "_intro" in b or "-intro" in b

def _is_outro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("outro") or b == "outro" or " outro" in b or "_outro" in b or "-outro" in b

def segment_label(p: dict) -> str:
    if _is_intro_name(p.get("title_guess", "")) or _is_intro_name(p.get("script_path", "")) or _is_intro_name(p.get("audio_path", "")):
        return "INTRO"
    if _is_outro_name(p.get("title_guess", "")) or _is_outro_name(p.get("script_path", "")) or _is_outro_name(p.get("audio_path", "")):
        return "OUTRO"
    n = p.get("chapter_no")
    if n is not None:
        return f"CHAPTER {n}"
    return "SEGMENT"

def pair_segments(scripts: List[str], audios: List[str]) -> List[dict]:
    intro_script = next((s for s in scripts if _is_intro_name(s)), None)
    outro_script = next((s for s in scripts if _is_outro_name(s)), None)
    intro_audio  = next((a for a in audios  if _is_intro_name(a)), None)
    outro_audio  = next((a for a in audios  if _is_outro_name(a)), None)

    scripts_left = [s for s in scripts if s not in {intro_script, outro_script}]
    audios_left  = [a for a in audios  if a not in {intro_audio,  outro_audio}]

    audio_by_no: Dict[int, List[str]] = {}
    for a in audios_left:
        n = _chapter_no_from_name(a)
        if n is not None:
            audio_by_no.setdefault(n, []).append(a)

    used_audio = set()
    pairs: List[dict] = []

    if intro_script and intro_audio:
        pairs.append({"chapter_no": 0, "title_guess": os.path.splitext(os.path.basename(intro_script))[0], "script_path": intro_script, "audio_path": intro_audio})
        used_audio.add(intro_audio)

    if outro_script and outro_audio:
        pairs.append({"chapter_no": 9998, "title_guess": os.path.splitext(os.path.basename(outro_script))[0], "script_path": outro_script, "audio_path": outro_audio})
        used_audio.add(outro_audio)

    def token_score(sp: str, ap: str) -> int:
        return len(_norm_tokens(sp).intersection(_norm_tokens(ap)))

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
            pairs.append({"chapter_no": sn, "title_guess": os.path.splitext(os.path.basename(s))[0], "script_path": s, "audio_path": chosen})

    def sort_key(p: dict):
        n = p.get("chapter_no")
        if n == 0:
            return (-1, p["title_guess"].lower())
        if n == 9998:
            return (999999, p["title_guess"].lower())
        if n is None:
            return (500000, p["title_guess"].lower())
        return (n, p["title_guess"].lower())

    pairs.sort(key=sort_key)
    return pairs


# ----------------------------
# Persistence helpers (cache-backed)
# ----------------------------

def sha1_file(path: str, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:12]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def out_base_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "uappress_segments_out")
    ensure_dir(d)
    return d

def segment_output_paths(cache_dir: str, seg_slug: str, audio_path: str) -> Dict[str, str]:
    """
    Deterministic output paths so reruns can rediscover completed work.
    Includes a short audio hash so if audio changes, it won't collide.
    """
    akey = sha1_file(audio_path)
    base = os.path.join(out_base_dir(cache_dir), f"{seg_slug}_{akey}")
    ensure_dir(base)
    return {
        "dir": base,
        "final_mp4": os.path.join(base, f"{seg_slug}_{akey}_final_subs.mp4"),
        "srt": os.path.join(base, f"{seg_slug}_{akey}.srt"),
        "scene_plan": os.path.join(base, f"{seg_slug}_{akey}_scene_plan.json"),
    }


# ----------------------------
# App UI
# ----------------------------

st.set_page_config(page_title="UAPpress — Segment MP4 Studio (ZIP)", layout="wide")
st.title("🛸 UAPpress — Segment MP4 Studio (ZIP upload)")
st.caption("Upload the ZIP from Documentary TTS Studio. Generate and download ONE segment at a time (subtitles burned in).")

# Session state
st.session_state.setdefault("pairs", [])
st.session_state.setdefault("workdir", None)
st.session_state.setdefault("out_dir", None)
st.session_state.setdefault("zip_bytes", None)
st.session_state.setdefault("status_by_slug", {})   # slug -> {state, paths, err}

with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎞 Output")
    resolution = st.selectbox("Resolution", ["1280x720", "1920x1080", "720x1280"], index=0)

    cache_dir = st.text_input("Cache folder (persistent)", value=os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache"))
    os.environ["UAPPRESS_CACHE_DIR"] = cache_dir
    ensure_dir(cache_dir)

    if st.button("🧹 Clear cache + outputs"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        ensure_dir(cache_dir)
        st.session_state["status_by_slug"] = {}
        st.success("Cache cleared.")

    st.divider()
    st.header("🎬 Scene pacing (stability)")
    max_scene_seconds = st.slider("Max seconds per scene (hard cap)", 8, 25, 18, 1)
    min_scene_seconds = st.slider("Min seconds per scene", 6, 15, 10, 1)

    st.divider()
    st.header("🧠 Models")
    text_model = st.selectbox("Scene planning model", ["gpt-5-mini", "gpt-5"], index=0)
    stt_model = st.selectbox("Transcription model", ["whisper-1"], index=0)
    language = st.text_input("Subtitle language (ISO-639-1)", value="en")

    st.divider()
    st.header("⚙ Safety")
    force_reencode = st.checkbox("Force re-encode final MP4 (safer, slower)", value=False)
    retry_images = st.checkbox("Retry image generation failures", value=True)


st.subheader("1) Upload Documentary ZIP")
zip_file = st.file_uploader("Upload ZIP from Documentary TTS Studio", type=["zip"])

if zip_file:
    st.session_state.zip_bytes = zip_file.getvalue()

# Extract once per session
if st.session_state.zip_bytes and not st.session_state.pairs:
    workdir, extract_dir = extract_zip_to_temp(st.session_state.zip_bytes)
    st.session_state.workdir = workdir
    scripts, audios = find_files(extract_dir)
    pairs = pair_segments(scripts, audios)
    st.session_state.pairs = pairs
    st.success(f"ZIP extracted. Found {len(scripts)} script file(s) and {len(audios)} audio file(s).")

pairs = st.session_state.pairs
if not pairs:
    st.info("Upload a ZIP to detect segments.")
    st.stop()

# Instantiate client only when needed
def get_client() -> OpenAI:
    if not (os.environ.get("OPENAI_API_KEY") or api_key):
        raise RuntimeError("Missing OpenAI API key.")
    return OpenAI()

def compute_desired_scenes(seg_seconds: float) -> int:
    total_needed = int(math.ceil(seg_seconds)) + 2  # small pad
    desired = int(math.ceil(total_needed / max_scene_seconds))
    return max(3, min(desired, 18))  # clamp to sane range

def generate_segment(p: dict) -> Dict[str, str]:
    client = get_client()
    title = p["title_guess"]
    audio_path = p["audio_path"]
    script_path = p["script_path"]

    seg_slug = f"{safe_slug(segment_label(p))}_{safe_slug(title)}"
    paths = segment_output_paths(cache_dir, seg_slug, audio_path)

    # If already built, return immediately
    if os.path.exists(paths["final_mp4"]) and os.path.exists(paths["srt"]):
        return paths

    script_text = read_script_file(script_path)
    seg_duration = float(get_media_duration_seconds(audio_path))
    if seg_duration <= 0:
        raise RuntimeError(f"Could not read duration for audio: {audio_path}")

    desired_scenes = compute_desired_scenes(seg_duration)

    st.info(f"{segment_label(p)} audio ≈ {seg_duration:.2f}s → requesting up to {desired_scenes} scenes (cap {max_scene_seconds}s/scene)")

    # 1) plan scenes (ask for enough scenes to avoid long clips)
    scenes = plan_scenes(
        client,
        chapter_title=title,
        chapter_text=script_text,
        max_scenes=desired_scenes,
        seconds_per_scene=min_scene_seconds,  # guidance only; we allocate below
        model=text_model,
    )
    scene_count = max(1, len(scenes))

    # 2) allocate per-scene seconds with hard cap
    total_needed = int(math.ceil(seg_duration)) + 2
    base = max(min_scene_seconds, int(math.floor(total_needed / scene_count)))
    base = min(base, max_scene_seconds)
    # Recompute counts if allocation would underfill total_needed
    # If base*scene_count < total_needed, we distribute remaining seconds but never exceed cap.
    alloc = [base] * scene_count
    remaining = total_needed - sum(alloc)
    i = 0
    while remaining > 0 and i < 5000:
        idx = i % scene_count
        if alloc[idx] < max_scene_seconds:
            alloc[idx] += 1
            remaining -= 1
        i += 1

    # If still remaining, we need more scenes; simplest: append extra scenes by reusing last prompt style
    # (Keeps the run stable rather than creating giant clips)
    if remaining > 0:
        extra = int(math.ceil(remaining / max_scene_seconds))
        extra = min(extra, 10)
        # Create lightweight extra scenes from the last prompt
        last_prompt = (scenes[-1].get("prompt") or "").strip() if scenes else "cinematic documentary b-roll"
        for k in range(extra):
            scenes.append({"scene": len(scenes) + 1, "seconds": min_scene_seconds, "prompt": last_prompt, "on_screen_text": None})
            alloc.append(min(min_scene_seconds, max_scene_seconds))
        scene_count = len(scenes)
        # try finishing remaining distribution
        remaining = total_needed - sum(alloc)
        i = 0
        while remaining > 0 and i < 5000:
            idx = i % scene_count
            if alloc[idx] < max_scene_seconds:
                alloc[idx] += 1
                remaining -= 1
            i += 1

    write_text(paths["scene_plan"], json.dumps(scenes, ensure_ascii=False, indent=2))

    # 3) generate clips
    progress = st.progress(0.0)
    status = st.empty()

    scene_paths: List[str] = []
    for j, sc in enumerate(scenes, start=1):
        sc_no = int(sc.get("scene", j))
        sc_seconds = int(alloc[j - 1]) if j - 1 < len(alloc) else base
        prompt = str(sc.get("prompt", "")).strip() or "cinematic documentary b-roll, realistic lighting"

        clip_path = os.path.join(paths["dir"], f"scene_{sc_no:02d}.mp4")

        status.write(f"Generating scene {sc_no}/{len(scenes)} ({sc_seconds}s)")
        if not os.path.exists(clip_path):
            last_err = None
            attempts = 3 if retry_images else 1
            for a in range(1, attempts + 1):
                try:
                    generate_video_clip(
                        client,
                        prompt=prompt,
                        seconds=sc_seconds,
                        size=resolution,
                        model="unused",
                        out_path=clip_path,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.25 * a)
            if last_err is not None:
                raise RuntimeError(f"Scene {sc_no} generation failed: {last_err}")

        scene_paths.append(clip_path)
        progress.progress(min(1.0, j / max(1, len(scenes))))

    # 4) stitch scenes
    status.write("Stitching scenes…")
    stitched_path = os.path.join(paths["dir"], "stitched.mp4")
    try:
        safe_concat_mp4s(scene_paths, stitched_path)
    except Exception:
        status.write("Concat-copy failed; re-encoding scenes for compatibility…")
        reencoded = []
        for sp in scene_paths:
            rp = sp.replace(".mp4", "_reencoded.mp4")
            if not os.path.exists(rp):
                reencode_mp4(sp, rp)
            reencoded.append(rp)
        safe_concat_mp4s(reencoded, stitched_path)

    # 5) mux audio
    status.write("Adding narration audio…")
    with_audio = os.path.join(paths["dir"], "with_audio.mp4")
    mux_audio(stitched_path, audio_path, with_audio)

    # 6) transcribe + burn subtitles
    status.write("Transcribing subtitles…")
    srt_text = transcribe_audio_to_srt(client, audio_path, model=stt_model, language=language)
    write_text(paths["srt"], srt_text)

    status.write("Burning subtitles into MP4…")
    final_mp4 = paths["final_mp4"]
    embed_srt_softsubs(with_audio, paths["srt"], final_mp4)

    if force_reencode:
        final_re = final_mp4.replace(".mp4", "_reencoded.mp4")
        final_mp4 = reencode_mp4(final_mp4, final_re)
        # keep canonical pointer
        paths["final_mp4"] = final_mp4

    status.write("✅ Segment ready.")
    progress.empty()
    return paths


st.subheader("2) Segments (Generate any order)")
st.caption("Tip: Generate one segment, download it, then move to the next. Each MP4 includes burned subtitles.")

for idx, p in enumerate(pairs, start=1):
    title = p["title_guess"]
    label = segment_label(p)
    seg_slug = f"{safe_slug(label)}_{safe_slug(title)}"

    with st.container(border=True):
        st.markdown(f"### {idx}. [{label}] {title}")

        st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

        # Discover existing outputs (if built previously)
        try:
            paths_guess = segment_output_paths(cache_dir, seg_slug, p["audio_path"])
            built_already = os.path.exists(paths_guess["final_mp4"]) and os.path.exists(paths_guess["srt"])
        except Exception:
            paths_guess = None
            built_already = False

        status_box = st.empty()
        cols = st.columns([1, 1, 2])

        gen_key = f"gen_{idx}_{seg_slug}"
        dl_key  = f"dl_{idx}_{seg_slug}"
        srt_key = f"srt_{idx}_{seg_slug}"

        with cols[0]:
            do_gen = st.button("⚙️ Generate", key=gen_key, type="primary")

        with cols[1]:
            if built_already and paths_guess:
                with open(paths_guess["final_mp4"], "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=f,
                        file_name=os.path.basename(paths_guess["final_mp4"]),
                        mime="video/mp4",
                        key=dl_key,
                    )
            else:
                st.button("⬇️ Download MP4", disabled=True, key=f"dl_disabled_{gen_key}")

        with cols[2]:
            if built_already and paths_guess:
                status_box.success("Ready (already generated).")
            else:
                status_box.info("Not generated yet.")

        # Optional SRT download (useful even though subs are burned-in)
        if built_already and paths_guess:
            with open(paths_guess["srt"], "rb") as f:
                st.download_button(
                    "⬇️ Download SRT (optional)",
                    data=f,
                    file_name=os.path.basename(paths_guess["srt"]),
                    mime="text/plain",
                    key=srt_key,
                )

        if do_gen:
            if not (os.environ.get("OPENAI_API_KEY") or api_key):
                st.error("Enter your OpenAI API key in the sidebar.")
                st.stop()

            try:
                status_box.warning("Generating… (this segment only)")
                paths = generate_segment(p)
                status_box.success("✅ Segment generated. Use Download buttons above.")
                st.rerun()
            except Exception as e:
                status_box.error("❌ Segment failed. You can retry Generate.")
                st.exception(e)

st.divider()
st.subheader("3) Notes")
st.write(
    "- This app is intentionally *one-segment-per-click* to avoid Streamlit Cloud long-run instability.\n"
    "- Each segment MP4 includes burned subtitles.\n"
    "- Outputs persist under your cache folder so reruns keep the downloads."
)
