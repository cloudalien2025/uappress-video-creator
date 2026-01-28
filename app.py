# app.py — UAPpress Segment MP4 Studio (ZIP upload) — OPTION A
# ✅ Generate + Download ONE segment at a time (subtitles burned in)
# ✅ NO logo on segments (faster + avoids extra re-encode passes)
# ✅ Build FINAL MP4 from a ZIP of segment MP4s (stitching) + apply logo ONCE
# ✅ Automatic YouTube description generator from ZIP scripts (no Whisper needed)

from __future__ import annotations

import os
import io
import re
import json
import math
import time
import zipfile
import shutil
import hashlib
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

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
    renumber_srt_blocks,
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------
# ffmpeg helpers
# ----------------------------

def _ffmpeg_escape_concat_path(p: str) -> str:
    # concat demuxer wants: file '...'
    return (p or "").replace("'", "'\\''")

def safe_concat_mp4s(paths: List[str], out_path: str) -> None:
    if not paths:
        raise ValueError("safe_concat_mp4s: no input paths provided")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"safe_concat_mp4s: missing input file: {p}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in paths:
            f.write(f"file '{_ffmpeg_escape_concat_path(os.path.abspath(p))}'\n")

    try:
        subprocess.run(
            [
                FFMPEG, "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                out_path,
            ],
            check=True,
        )
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass

def overlay_logo_mp4(
    in_mp4: str,
    logo_path: str,
    out_mp4: str,
    *,
    size_pct: int = 10,
    margin_px: int = 30,
    opacity: float = 0.9,
) -> None:
    """
    Overlays logo top-right with scaling and opacity.
    NOTE: Any overlay requires video re-encode.
    """
    opacity = max(0.0, min(1.0, float(opacity)))
    size_pct = max(3, min(30, int(size_pct)))
    margin_px = max(0, int(margin_px))

    filter_complex = (
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[lg];"
        f"[lg][0:v]scale2ref=w=iw*{size_pct}/100:h=-1[lg2][base];"
        f"[base][lg2]overlay=x=W-w-{margin_px}:y={margin_px}:format=auto[v]"
    )

    subprocess.run(
        [
            FFMPEG, "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", in_mp4,
            "-i", logo_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_mp4,
        ],
        check=True,
    )

# ----------------------------
# Pairing helpers (ZIP from TTS Studio)
# ----------------------------

def _is_intro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("intro") or b == "intro" or " intro" in b or "_intro" in b or "-intro" in b

def _is_outro_name(name: str) -> bool:
    b = os.path.splitext(os.path.basename(name))[0].lower()
    return b.startswith("outro") or b == "outro" or " outro" in b or "_outro" in b or "-outro" in b

def _chapter_no_from_name(name: str) -> Optional[int]:
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
    """
    Creates pairs:
      - intro script ↔ intro audio (by name)
      - outro script ↔ outro audio (by name)
      - chapter_N script ↔ chapter_N audio (by extracted chapter number)
    Falls back to token overlap if needed.
    """
    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        toks = set(re.split(r"[^a-z0-9]+", s))
        toks.discard("")
        return toks

    def token_score(sp: str, ap: str) -> int:
        return len(norm_tokens(sp).intersection(norm_tokens(ap)))

    intro_script = next((s for s in scripts if _is_intro_name(s)), None)
    outro_script = next((s for s in scripts if _is_outro_name(s)), None)
    intro_audio = next((a for a in audios if _is_intro_name(a)), None)
    outro_audio = next((a for a in audios if _is_outro_name(a)), None)

    scripts_left = [s for s in scripts if s not in {intro_script, outro_script}]
    audios_left = [a for a in audios if a not in {intro_audio, outro_audio}]

    audio_by_no: Dict[int, List[str]] = {}
    for a in audios_left:
        n = _chapter_no_from_name(a)
        if n is not None:
            audio_by_no.setdefault(n, []).append(a)

    used_audio = set()
    pairs: List[dict] = []

    if intro_script and intro_audio:
        pairs.append({"chapter_no": 0, "title_guess": os.path.splitext(os.path.basename(intro_script))[0],
                      "script_path": intro_script, "audio_path": intro_audio})
        used_audio.add(intro_audio)

    if outro_script and outro_audio:
        pairs.append({"chapter_no": 9998, "title_guess": os.path.splitext(os.path.basename(outro_script))[0],
                      "script_path": outro_script, "audio_path": outro_audio})
        used_audio.add(outro_audio)

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
            pairs.append({"chapter_no": sn, "title_guess": os.path.splitext(os.path.basename(s))[0],
                          "script_path": s, "audio_path": chosen})

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
# Persistence helpers
# ----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _sha1_file(path: str, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:12]

def _out_base_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "uappress_segments_out")
    _ensure_dir(d)
    return d

def _segment_output_paths(cache_dir: str, seg_slug: str, audio_path: str) -> Dict[str, str]:
    akey = _sha1_file(audio_path)
    base = os.path.join(_out_base_dir(cache_dir), f"{seg_slug}_{akey}")
    _ensure_dir(base)
    return {
        "dir": base,
        "final_mp4": os.path.join(base, f"{seg_slug}_{akey}_final_subs.mp4"),
        "srt": os.path.join(base, f"{seg_slug}_{akey}.srt"),
        "scene_plan": os.path.join(base, f"{seg_slug}_{akey}_scene_plan.json"),
        "with_audio": os.path.join(base, f"{seg_slug}_{akey}_with_audio.mp4"),
    }

# ----------------------------
# YouTube description helpers
# ----------------------------

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def _seconds_to_ts(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"

def build_estimated_timestamps(pairs: List[dict], scripts_by_path: Dict[str, str], wpm: int = 145) -> str:
    """
    Estimate timestamps from script word counts (no audio needed).
    """
    t = 0
    lines = []
    for p in pairs:
        label = segment_label(p)
        title = p["title_guess"]
        txt = scripts_by_path.get(p["script_path"], "")
        words = _word_count(txt)
        est_seconds = int(round((words / max(1, wpm)) * 60))

        if label == "INTRO":
            lines.append(f"{_seconds_to_ts(t)} Intro")
        elif label == "OUTRO":
            lines.append(f"{_seconds_to_ts(t)} Outro")
        else:
            lines.append(f"{_seconds_to_ts(t)} {label.title()} — {title}")

        t += max(20, est_seconds)  # forward progress even if tiny
    return "\n".join(lines)

def youtube_prompt(topic_guess: str, timestamps: str, script_bundle: str) -> str:
    sponsor_block = (
        "This episode is sponsored by OPA Nutrition, makers of premium wellness supplements designed to support "
        "focus, clarity, energy, resilience, and long-term health. Learn more at opanutrition.com."
    )
    return f"""
You are writing a YouTube description for a serious, audio-first investigative documentary channel (UAPpress).
Tone: restrained, precise, credible. No hype. No sensationalism. No medical claims.

TOPIC (best guess):
{topic_guess}

CHAPTER TIMESTAMPS (estimated):
{timestamps}

FULL SCRIPT (intro + chapters + outro):
{script_bundle}

DELIVER EXACTLY:
1) YouTube Description (ready to paste)
   - Hook (2–3 lines)
   - What this documentary covers (5–8 bullets)
   - Include the timestamps exactly as provided
   - Include sponsor block once, naturally
   - Engagement CTA: subscribe, comment where you're listening from, what you think happened, what case to cover next
2) Pinned Comment (2–3 lines, question-forward)
3) 6–10 Hashtags (safe, relevant)

Rules:
- Do not invent facts beyond the script.
- Avoid demonetization bait words/phrases (no “shocking”, “proof”, “exposed”, “cover-up” in the hook).
- Keep it clean and skimmable.

Sponsor block to include (verbatim):
{sponsor_block}
""".strip()

# ----------------------------
# App UI
# ----------------------------

st.set_page_config(page_title="UAPpress — Segment MP4 Studio", layout="wide")
st.title("🛸 UAPpress — Segment MP4 Studio (ZIP upload) — Option A")
st.caption(
    "Upload the ZIP from Documentary TTS Studio. Generate/download segments one at a time (subs burned in). "
    "Then stitch a final MP4 from a ZIP of segment MP4s and apply logo ONCE."
)

st.session_state.setdefault("pairs", [])
st.session_state.setdefault("zip_bytes", None)
st.session_state.setdefault("scripts_by_path", {})
st.session_state.setdefault("topic_guess", "")
st.session_state.setdefault("yt_desc_text", "")

with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎨 Branding (FINAL MP4 only)")
    logo_file = st.file_uploader("Upload logo (PNG preferred)", type=["png", "jpg", "jpeg"])
    apply_logo_on_final = st.checkbox("Apply logo to FINAL movie (top-right)", value=True)
    logo_size = st.slider("Logo size (% of width)", 5, 20, 10, 1)
    logo_margin = st.slider("Logo margin (px)", 10, 80, 30, 1)
    logo_opacity = st.slider("Logo opacity", 0.40, 1.00, 0.90, 0.05)

    st.divider()
    st.header("🎞 Output")
    resolution = st.selectbox("Resolution", ["1280x720", "1920x1080", "720x1280"], index=0)

    cache_dir = st.text_input("Cache folder (persistent)", value=os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache"))
    os.environ["UAPPRESS_CACHE_DIR"] = cache_dir
    _ensure_dir(cache_dir)

    if st.button("🧹 Clear cache + outputs"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        _ensure_dir(cache_dir)
        st.success("Cache cleared.")

    st.divider()
    st.header("🎬 Scene pacing (stability)")
    max_scene_seconds = st.slider("Max seconds per scene (hard cap)", min_value=10, max_value=40, value=30, step=1,)
    min_scene_seconds = st.slider("Min seconds per scene", min_value=10, max_value=30, value=20, step=1,)

    st.divider()
    st.header("🧠 Models")
    text_model = st.selectbox("Scene planning model", ["gpt-5-mini", "gpt-5"], index=0)
    stt_model = st.selectbox("Transcription model", ["whisper-1"], index=0)
    language = st.text_input("Subtitle language (ISO-639-1)", value="en")

    st.divider()
    st.header("⚙ Safety")
    force_reencode = st.checkbox("Force re-encode final MP4 (safer, slower)", value=False)
    retry_images = st.checkbox("Retry image generation failures", value=True)

def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or api_key
    if not key:
        raise RuntimeError("Missing OpenAI API key.")
    return OpenAI(api_key=key)

# ----------------------------
# 1) Upload ZIP (TTS Studio)
# ----------------------------

st.subheader("1) Upload ZIP from Documentary TTS Studio")
zip_file = st.file_uploader("ZIP (scripts + audio)", type=["zip"], key="tts_zip")

if zip_file:
    st.session_state.zip_bytes = zip_file.getvalue()
    # reset detection on new upload
    st.session_state.pairs = []
    st.session_state.scripts_by_path = {}
    st.session_state.topic_guess = ""
    st.session_state.yt_desc_text = ""

if st.session_state.zip_bytes and not st.session_state.pairs:
    workdir, extract_dir = extract_zip_to_temp(st.session_state.zip_bytes)
    scripts, audios = find_files(extract_dir)
    pairs = pair_segments(scripts, audios)

    st.session_state.pairs = pairs
    st.session_state.scripts_by_path = {p: read_script_file(p) for p in scripts}
    st.session_state.topic_guess = pairs[0]["title_guess"] if pairs else "UAPpress Documentary"

    st.success(f"ZIP extracted. Found {len(scripts)} script file(s) and {len(audios)} audio file(s).")

pairs = st.session_state.pairs
scripts_by_path = st.session_state.scripts_by_path

if not pairs:
    st.info("Upload the TTS Studio ZIP to detect segments.")
    st.stop()

# ----------------------------
# 2) Auto YouTube description (from scripts)
# ----------------------------

st.divider()
st.subheader("2) Automatic YouTube Description (from ZIP scripts)")

# Bundle scripts in order
ordered_text_parts = []
for p in pairs:
    label = segment_label(p)
    title = p["title_guess"]
    txt = scripts_by_path.get(p["script_path"], "")
    if not txt.strip():
        continue
    ordered_text_parts.append(f"[{label}] {title}\n{txt.strip()}")

script_bundle = "\n\n".join(ordered_text_parts).strip()
timestamps_text = build_estimated_timestamps(pairs, scripts_by_path, wpm=145)

colA, colB = st.columns([1, 1])
with colA:
    st.text_area("Estimated timestamps (word-count based)", value=timestamps_text, height=220)
with colB:
    st.text_area(
        "Script bundle (preview)",
        value=script_bundle[:12000] + ("\n\n[...truncated...]" if len(script_bundle) > 12000 else ""),
        height=220,
    )

if st.button("📝 Generate YouTube Description", type="primary"):
    try:
        client = get_client()
        prompt = youtube_prompt(st.session_state.topic_guess or "UAPpress Documentary", timestamps_text, script_bundle[:12000])
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": "You write high-performing YouTube metadata for serious investigative documentaries."},
                {"role": "user", "content": prompt},
            ],
        )
        st.session_state.yt_desc_text = (resp.output_text or "").strip()
        st.success("YouTube description generated.")
    except Exception as e:
        st.error(f"Description generation failed: {e}")

if st.session_state.yt_desc_text:
    st.text_area("YouTube Description Output (copy/paste)", value=st.session_state.yt_desc_text, height=360)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "⬇️ Download youtube_description.txt",
            data=st.session_state.yt_desc_text.encode("utf-8"),
            file_name="youtube_description.txt",
            mime="text/plain",
        )
    with c2:
        st.download_button(
            "⬇️ Download timestamps.txt",
            data=timestamps_text.encode("utf-8"),
            file_name="timestamps.txt",
            mime="text/plain",
        )

# ----------------------------
# 3) Segment generation (one at a time) — NO LOGO (Option A)
# ----------------------------

st.divider()
st.subheader("3) Segments (Generate any order, one at a time)")
st.caption("Each segment produces a final MP4 with burned subtitles. (Logo is applied only to the FINAL stitched movie.)")

def compute_desired_scenes(seg_seconds: float) -> int:
    total_needed = int(math.ceil(seg_seconds)) + 2
    desired = int(math.ceil(total_needed / max_scene_seconds))
    return max(3, min(desired, 18))

def generate_segment(p: dict) -> Dict[str, str]:
    client = get_client()
    title = p["title_guess"]
    label = segment_label(p)
    audio_path = p["audio_path"]
    script_path = p["script_path"]

    seg_slug = f"{safe_slug(label)}_{safe_slug(title)}"
    paths = _segment_output_paths(cache_dir, seg_slug, audio_path)

    # If already built, return quickly
    if os.path.exists(paths["final_mp4"]) and os.path.exists(paths["srt"]):
        return paths

    script_text = read_script_file(script_path)
    seg_duration = float(get_media_duration_seconds(audio_path))
    if seg_duration <= 0:
        raise RuntimeError(f"Could not read duration for audio: {audio_path}")

    desired_scenes = compute_desired_scenes(seg_duration)
    st.info(f"{label} audio ≈ {seg_duration:.2f}s → requesting up to {desired_scenes} scenes (cap {max_scene_seconds}s/scene)")

    scenes = plan_scenes(
        client,
        chapter_title=title,
        chapter_text=script_text,
        max_scenes=desired_scenes,
        seconds_per_scene=min_scene_seconds,
        model=text_model,
    )
    scene_count = max(1, len(scenes))

    # Allocate durations with hard cap
    total_needed = int(math.ceil(seg_duration)) + 2
    base = max(min_scene_seconds, int(math.floor(total_needed / scene_count)))
    base = min(base, max_scene_seconds)

    alloc = [base] * scene_count
    remaining = total_needed - sum(alloc)

    i = 0
    while remaining > 0 and i < 5000:
        idx = i % scene_count
        if alloc[idx] < max_scene_seconds:
            alloc[idx] += 1
            remaining -= 1
        i += 1

    # If still remaining, append extra scenes using last prompt
    if remaining > 0:
        extra = int(math.ceil(remaining / max_scene_seconds))
        extra = min(extra, 10)
        last_prompt = (scenes[-1].get("prompt") or "").strip() if scenes else "cinematic documentary b-roll"
        for _ in range(extra):
            scenes.append({"scene": len(scenes) + 1, "seconds": min_scene_seconds, "prompt": last_prompt, "on_screen_text": None})
            alloc.append(min_scene_seconds)
        scene_count = len(scenes)
        remaining = total_needed - sum(alloc)
        i = 0
        while remaining > 0 and i < 5000:
            idx = i % scene_count
            if alloc[idx] < max_scene_seconds:
                alloc[idx] += 1
                remaining -= 1
            i += 1

    write_text(paths["scene_plan"], json.dumps(scenes, ensure_ascii=False, indent=2))

    prog = st.progress(0.0)
    status = st.empty()

    scene_paths: List[str] = []
    for j, sc in enumerate(scenes, start=1):
        sc_no = int(sc.get("scene", j))
        sc_seconds = int(alloc[j - 1]) if j - 1 < len(alloc) else base
        prompt = str(sc.get("prompt", "")).strip() or "cinematic documentary b-roll, realistic lighting"

        clip_path = os.path.join(paths["dir"], f"scene_{sc_no:02d}.mp4")

        status.write(f"Generating visuals — Scene {sc_no}/{len(scenes)} ({sc_seconds}s)")
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
        prog.progress(min(1.0, j / max(1, len(scenes))))

    # Stitch scenes
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

    # Mux audio
    status.write("Adding narration audio…")
    mux_audio(stitched_path, audio_path, paths["with_audio"])

    # Transcribe subtitles + burn
    status.write("Transcribing subtitles…")
    srt_text = transcribe_audio_to_srt(client, audio_path, model=stt_model, language=language)
    write_text(paths["srt"], srt_text)

    status.write("Burning subtitles into MP4…")
    embed_srt_softsubs(paths["with_audio"], paths["srt"], paths["final_mp4"])

    final_mp4 = paths["final_mp4"]

    if force_reencode:
        status.write("Force re-encoding segment MP4…")
        final_re = final_mp4.replace(".mp4", "_reencoded.mp4")
        final_mp4 = reencode_mp4(final_mp4, final_re)
        paths["final_mp4"] = final_mp4

    status.write("✅ Segment ready.")
    prog.empty()
    return paths

for idx, p in enumerate(pairs, start=1):
    title = p["title_guess"]
    label = segment_label(p)
    seg_slug = f"{safe_slug(label)}_{safe_slug(title)}"

    with st.container(border=True):
        st.markdown(f"### {idx}. [{label}] {title}")
        st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

        # discover existing output
        paths_guess = None
        built = False
        try:
            paths_guess = _segment_output_paths(cache_dir, seg_slug, p["audio_path"])
            built = os.path.exists(paths_guess["final_mp4"]) and os.path.exists(paths_guess["srt"])
        except Exception:
            pass

        cols = st.columns([1, 1, 2])
        with cols[0]:
            do_gen = st.button("⚙️ Generate", key=f"gen_{idx}_{seg_slug}", type="primary")
        with cols[1]:
            if built and paths_guess:
                with open(paths_guess["final_mp4"], "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=f,
                        file_name=os.path.basename(paths_guess["final_mp4"]),
                        mime="video/mp4",
                        key=f"dl_{idx}_{seg_slug}",
                    )
            else:
                st.button("⬇️ Download MP4", disabled=True, key=f"dl_disabled_{idx}_{seg_slug}")
        with cols[2]:
            if built:
                st.success("Ready (already generated).")
            else:
                st.info("Not generated yet.")

        if built and paths_guess:
            with open(paths_guess["srt"], "rb") as f:
                st.download_button(
                    "⬇️ Download SRT (optional)",
                    data=f,
                    file_name=os.path.basename(paths_guess["srt"]),
                    mime="text/plain",
                    key=f"srt_{idx}_{seg_slug}",
                )

        if do_gen:
            try:
                st.warning("Generating… (this segment only)")
                _ = generate_segment(p)
                st.success("✅ Segment generated. Download it now.")
                st.rerun()
            except Exception as e:
                st.error("❌ Segment failed. You can retry Generate.")
                st.exception(e)

# ----------------------------
# 4) Build final MP4 from ZIP of segment MP4s (logo applied once)
#    ✅ PATCH: Use DigitalOcean (or any) ZIP URL + STREAM download to disk (no RAM blowup)
#    NOTE: Add this near the top of your file if not already present:
#          import requests
# ----------------------------

import requests  # <-- if you already have this import at the top, remove this line here.

st.divider()
st.subheader("4) Build FINAL MP4 from ZIP (stitching + optional logo ONCE)")
st.caption(
    "Best practice on Streamlit Cloud: provide a ZIP URL (DigitalOcean Spaces) so the app can stream-download to disk. "
    "Upload is kept for small test ZIPs only."
)

mode = st.radio(
    "ZIP source",
    ["URL (recommended)", "Upload (small test ZIP only)"],
    index=0,
    horizontal=True,
)

zip_url = ""
uploaded_zip = None

if mode == "URL (recommended)":
    zip_url = st.text_input(
        "ZIP URL (e.g., https://cloud-alien.nyc3.digitaloceanspaces.com/episodes/roswell/segments.zip)",
        value="",
        placeholder="Paste your public DigitalOcean Spaces URL here…",
    )
else:
    uploaded_zip = st.file_uploader("Upload ZIP of segment MP4s", type=["zip"], key="zip_segments_small")

def _sort_key_mp4(name: str) -> Tuple[int, str]:
    base = os.path.basename(name).lower()
    # prefer seg_01..., seg_02...
    m = re.search(r"seg[\s_\-]*0*(\d{1,4})", base)
    if m:
        return (int(m.group(1)), base)
    # fallback: chapter number
    m2 = re.search(r"(?:chapter|ch)[\s_\-]*0*(\d{1,4})", base)
    if m2:
        return (1000 + int(m2.group(1)), base)
    # intro/outro
    if "intro" in base:
        return (-1, base)
    if "outro" in base:
        return (999999, base)
    return (500000, base)

def _download_zip_streaming(url: str, dest_path: str) -> None:
    """
    Streams a remote ZIP to disk to avoid holding the full file in RAM.
    Shows a progress bar when Content-Length is available.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Please paste a ZIP URL.")

    headers = {"User-Agent": "uappress-streamlit/1.0"}
    prog = st.progress(0.0)
    status = st.empty()

    with requests.get(url, stream=True, timeout=(15, 300), headers=headers) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        status.write("Downloading ZIP to server disk…")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    prog.progress(min(1.0, downloaded / max(1, total_bytes)))

    prog.empty()
    status.empty()

build_final_disabled = True
if mode == "URL (recommended)":
    build_final_disabled = not bool((zip_url or "").strip())
else:
    build_final_disabled = uploaded_zip is None

build_final = st.button("🎞 Build FINAL MP4", type="primary", disabled=build_final_disabled)

if build_final:
    try:
        with tempfile.TemporaryDirectory() as td:
            # 1) Get ZIP onto disk (NOT in RAM)
            zip_path = os.path.join(td, "segments.zip")

            if mode == "URL (recommended)":
                _download_zip_streaming(zip_url, zip_path)
            else:
                # Small test ZIP only; still arrives in memory, but we write it to disk immediately.
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getvalue())

            # 2) Extract ZIP
            extract_dir = os.path.join(td, "unzipped")
            os.makedirs(extract_dir, exist_ok=True)

            st.info("Extracting ZIP…")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            # 3) Find MP4s
            mp4s: List[str] = []
            for root, _, files in os.walk(extract_dir):
                for fn in files:
                    if fn.lower().endswith(".mp4"):
                        mp4s.append(os.path.join(root, fn))

            if not mp4s:
                st.error("No MP4 files found in the ZIP.")
                st.stop()

            mp4s.sort(key=_sort_key_mp4)

            st.write("MP4 order:")
            for m in mp4s:
                st.write(f"- {os.path.basename(m)}")

            # 4) Concat
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir, exist_ok=True)

            full_mp4 = os.path.join(out_dir, "uappress_full_documentary.mp4")

            st.info("Concatenating segments…")
            try:
                safe_concat_mp4s(mp4s, full_mp4)
            except Exception:
                st.info("Concat-copy failed; re-encoding segments for compatibility…")
                reencoded = []
                for mp in mp4s:
                    rp = mp.replace(".mp4", "_reencoded.mp4")
                    if not os.path.exists(rp):
                        reencode_mp4(mp, rp)
                    reencoded.append(rp)
                safe_concat_mp4s(reencoded, full_mp4)

            final_out = full_mp4

            # 5) Apply logo ONCE (Option A)
            if apply_logo_on_final and logo_file is not None:
                st.info("Applying logo watermark to FINAL movie…")
                logo_path = os.path.join(out_dir, "logo_upload.png")
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getvalue())

                branded = os.path.join(out_dir, "uappress_full_documentary_logo.mp4")
                overlay_logo_mp4(
                    full_mp4,
                    logo_path,
                    branded,
                    size_pct=logo_size,
                    margin_px=logo_margin,
                    opacity=logo_opacity,
                )
                final_out = branded

            # 6) Optional force re-encode
            if force_reencode:
                st.info("Force re-encoding FINAL movie…")
                final_re = final_out.replace(".mp4", "_reencoded.mp4")
                final_out = reencode_mp4(final_out, final_re)

            st.success("✅ Final movie ready!")
            st.video(final_out)

            with open(final_out, "rb") as f:
                st.download_button(
                    "⬇️ Download FINAL MP4 (YouTube Ready)",
                    data=f,
                    file_name=os.path.basename(final_out),
                    mime="video/mp4",
                )

    except Exception as e:
        st.error(f"Final build failed: {e}")
        st.exception(e)
