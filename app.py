# ============================
# PART 1/5 — Core Setup, Sidebar Keys, Config, Imports, Session State
# ============================
# app.py — UAPpress Video Creator (TTS ZIP → Generate Videos → Upload each MP4 to Spaces)
#
# GOALS (per your new plan):
# ✅ Upload ONE TTS Studio ZIP (scripts + audio)
# ✅ ONE button: "Generate Videos"
# ✅ Sequential generation order: Intro → Chapters 1..N → Outro
# ✅ After each MP4: upload to DigitalOcean Spaces → clear per-segment cache → continue
# ✅ NO subtitle burn-in (you’ll do subtitles in CapCut)
# ✅ OpenAI API key entered manually in sidebar per run (public GitHub safe)
# ✅ app.py broken into 5+ labeled parts for easier troubleshooting

from __future__ import annotations

import os
import io
import re
import json
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
    # ZIP ingestion + pairing
    extract_zip_to_temp,
    find_files,
    read_script_file,
    safe_slug,
    pair_segments,      # expects your existing pairing helper (intro/outro/chapter matching)
    segment_label,      # expects your existing label helper
    # Single-segment builder (NO subtitle burn-in)
    build_segment_mp4_no_subs,  # to be defined/updated in video_creator.py in later parts
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress — Video Creator", layout="wide")
st.title("🛸 UAPpress — Video Creator (TTS ZIP → MP4s → Spaces)")
st.caption(
    "Upload the ZIP from TTS Studio. Click **Generate Videos** to build Intro → Chapters → Outro sequentially. "
    "Each MP4 uploads to DigitalOcean Spaces immediately. Subtitles are NOT burned in (CapCut later)."
)


# ----------------------------
# Sidebar — API key + Spaces config + output options
# ----------------------------
def _sec_get(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


with st.sidebar:
    st.header("🔐 Keys (not saved)")
    st.caption("OpenAI key is kept only in session memory. Spaces creds come from Secrets/env.")

    st.session_state.setdefault("OPENAI_API_KEY_INPUT", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        key="OPENAI_API_KEY_INPUT",
        placeholder="sk-...",
        help="Stored only in st.session_state for this run (not written to disk).",
    )

    st.divider()
    st.header("☁️ DigitalOcean Spaces")
    st.caption("Set these in Streamlit Secrets for safety (recommended).")

    # Prefer secrets/env; do NOT ask user to type Spaces secrets in a public app.
    DO_SPACES_REGION = _sec_get("DO_SPACES_REGION", os.getenv("DO_SPACES_REGION", "nyc3"))
    DO_SPACES_BUCKET = _sec_get("DO_SPACES_BUCKET", os.getenv("DO_SPACES_BUCKET", ""))
    DO_SPACES_PUBLIC_BASE = _sec_get("DO_SPACES_PUBLIC_BASE", os.getenv("DO_SPACES_PUBLIC_BASE", ""))

    st.text_input("Region", value=DO_SPACES_REGION, disabled=True)
    st.text_input("Bucket", value=DO_SPACES_BUCKET, disabled=True)
    st.text_input("Public base URL (optional)", value=DO_SPACES_PUBLIC_BASE, disabled=True)

    st.divider()
    st.header("🎞 Output")
    resolution = st.selectbox("Resolution", ["1280x720", "1920x1080", "720x1280"], index=0)

    # Persistent cache directory
    cache_dir = st.text_input(
        "Cache folder (persistent)",
        value=os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache"),
        help="Used for temporary assets and checkpoints. Safe to clear if needed.",
    )
    os.environ["UAPPRESS_CACHE_DIR"] = cache_dir

    # Where to upload each segment MP4 in Spaces
    upload_prefix = st.text_input(
        "Spaces upload folder (prefix)",
        value="uappress/segments",
        help="Example: uappress/segments",
    ).strip().strip("/")

    st.divider()
    st.header("🧹 Utilities")
    if st.button("Clear cache folder"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        st.success("Cache cleared.")


# ----------------------------
# OpenAI client (requires sidebar key)
# ----------------------------
api_key = (api_key_input or "").strip()
if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

client = OpenAI(api_key=api_key)


# ----------------------------
# Minimal session state (job + results)
# ----------------------------
st.session_state.setdefault("tts_zip_bytes", None)
st.session_state.setdefault("pairs", [])
st.session_state.setdefault("scripts_by_path", {})
st.session_state.setdefault("job_manifest", {})  # segment_id -> {status, url, error}
st.session_state.setdefault("last_public_urls", [])  # list of uploaded URLs (ordered)


# NOTE:
# Part 2 will cover:
# - ZIP uploader + extraction
# - segment pairing Intro/Chapters/Outro
# - building a deterministic job list + checkpoint manifest

# ============================
# PART 2/5 — ZIP Upload + Extraction + Pairing + Job Manifest (Resume-Safe)
# ============================

# ----------------------------
# Helpers: stable IDs + manifest persistence
# ----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1_bytes(b: bytes, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    h.update(b[:max_bytes])
    return h.hexdigest()[:12]


def _sha1_file(path: str, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:12]


def _manifest_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "uappress_video_creator")
    _ensure_dir(d)
    return d


def _manifest_path(cache_dir: str, zip_hash: str) -> str:
    return os.path.join(_manifest_dir(cache_dir), f"manifest_{zip_hash}.json")


def _load_manifest(cache_dir: str, zip_hash: str) -> Dict[str, dict]:
    p = _manifest_path(cache_dir, zip_hash)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_manifest(cache_dir: str, zip_hash: str, manifest: Dict[str, dict]) -> None:
    p = _manifest_path(cache_dir, zip_hash)
    _ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _segment_id(p: dict) -> str:
    """
    Deterministic segment ID: order+slug+audio hash.
    We hash audio file so re-uploads with different audio produce new IDs.
    """
    label = segment_label(p)
    title = p.get("title_guess") or "segment"
    audio_path = p.get("audio_path") or ""
    ah = _sha1_file(audio_path) if (audio_path and os.path.exists(audio_path)) else "noaudio"
    return f"{safe_slug(label)}__{safe_slug(title)}__{ah}"


def _build_ordered_job_list(pairs: List[dict]) -> List[dict]:
    """
    Enforces strict generation order:
      Intro → Chapters (ascending) → Outro
    `pair_segments` already sorts well, but we lock the order explicitly.
    """
    intro = [p for p in pairs if segment_label(p) == "INTRO"]
    outro = [p for p in pairs if segment_label(p) == "OUTRO"]
    chapters = [p for p in pairs if segment_label(p).startswith("CHAPTER")]
    others = [p for p in pairs if p not in intro + chapters + outro]

    # Chapters sorted by extracted chapter_no if present; fallback title
    def ch_key(p: dict):
        n = p.get("chapter_no")
        if n is None:
            return (999999, (p.get("title_guess") or "").lower())
        return (int(n), (p.get("title_guess") or "").lower())

    chapters = sorted(chapters, key=ch_key)
    # Keep others last (rare) — still deterministic
    others = sorted(others, key=lambda p: (p.get("title_guess") or "").lower())

    return intro + chapters + outro + others


# ----------------------------
# 1) Upload ZIP (TTS Studio)
# ----------------------------
st.subheader("1) Upload ZIP from TTS Studio (scripts + audio)")
zip_file = st.file_uploader("TTS Studio ZIP", type=["zip"], key="tts_zip_video_creator")

if zip_file:
    # Store raw bytes in session (fast)
    st.session_state.tts_zip_bytes = zip_file.getvalue()

    # Reset runtime state for new ZIP
    st.session_state.pairs = []
    st.session_state.scripts_by_path = {}
    st.session_state.last_public_urls = []
    st.session_state.job_manifest = {}

# If no ZIP, stop early
if not st.session_state.tts_zip_bytes:
    st.info("Upload the TTS Studio ZIP to detect segments.")
    st.stop()

zip_hash = _sha1_bytes(st.session_state.tts_zip_bytes)

# ----------------------------
# 2) Extract + pair segments
# ----------------------------
if not st.session_state.pairs:
    with st.spinner("Extracting ZIP and detecting segments…"):
        workdir, extract_dir = extract_zip_to_temp(st.session_state.tts_zip_bytes)
        scripts, audios = find_files(extract_dir)

        if not scripts:
            st.error("No scripts found in ZIP. Expected scripts/*.txt")
            st.stop()
        if not audios:
            st.error("No audio found in ZIP. Expected audio/*.mp3 (or wav)")
            st.stop()

        pairs = pair_segments(scripts, audios)

        # Cache script text for preview/metadata
        scripts_by_path = {p: read_script_file(p) for p in scripts}

        # Build manifest by resuming if it exists
        manifest = _load_manifest(cache_dir, zip_hash)

        # Ensure every pair has a manifest entry
        ordered = _build_ordered_job_list(pairs)
        for p in ordered:
            sid = _segment_id(p)
            manifest.setdefault(
                sid,
                {
                    "label": segment_label(p),
                    "title": p.get("title_guess") or "",
                    "status": "pending",  # pending | done | failed
                    "public_url": "",
                    "error": "",
                    "updated_at": "",
                },
            )

        _save_manifest(cache_dir, zip_hash, manifest)

        st.session_state.pairs = ordered
        st.session_state.scripts_by_path = scripts_by_path
        st.session_state.job_manifest = manifest

        st.success(f"Detected {len(ordered)} segment(s). Manifest loaded for resume support.")

pairs = st.session_state.pairs
manifest = st.session_state.job_manifest
scripts_by_path = st.session_state.scripts_by_path

# ----------------------------
# 3) Preview detected order + manifest status
# ----------------------------
st.markdown("### Detected generation order")
for i, p in enumerate(pairs, start=1):
    sid = _segment_id(p)
    m = manifest.get(sid, {})
    status = (m.get("status") or "pending").upper()
    label = segment_label(p)
    title = p.get("title_guess") or "Untitled"
    st.write(f"{i}. [{label}] {title} — **{status}**")

st.caption("Next: Part 3 will add the ONE button **Generate Videos** + sequential loop + Spaces upload per segment.")

# ----------------------------
# 3) Segment generation (segments ONLY — no subs, no logo)
# ----------------------------

st.divider()
st.subheader("3) Generate Segment MP4s (clean output for CapCut)")
st.caption(
    "Each segment generates a clean MP4 with narration audio only. "
    "No subtitles, no logos, no final stitching. Finish everything in CapCut."
)

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

    # If already generated, skip work
    if os.path.exists(paths["with_audio"]):
        return paths

    script_text = read_script_file(script_path)
    seg_duration = float(get_media_duration_seconds(audio_path))
    if seg_duration <= 0:
        raise RuntimeError(f"Could not read duration for audio: {audio_path}")

    desired_scenes = compute_desired_scenes(seg_duration)
    st.info(f"{label} audio ≈ {seg_duration:.1f}s → up to {desired_scenes} scenes")

    # ---- Scene planning (LLM)
    scenes = plan_scenes(
        client,
        chapter_title=title,
        chapter_text=script_text,
        max_scenes=desired_scenes,
        seconds_per_scene=min_scene_seconds,
        model=text_model,
    )

    scene_count = max(1, len(scenes))
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

    # ---- Generate scene clips
    prog = st.progress(0.0)
    status = st.empty()
    scene_paths: List[str] = []

    for j, sc in enumerate(scenes, start=1):
        sc_no = int(sc.get("scene", j))
        sc_seconds = int(alloc[j - 1]) if j - 1 < len(alloc) else base
        prompt = str(sc.get("prompt", "")).strip() or "cinematic documentary b-roll"

        clip_path = os.path.join(paths["dir"], f"scene_{sc_no:02d}.mp4")

        status.write(f"Generating scene {sc_no}/{len(scenes)} ({sc_seconds}s)")
        if not os.path.exists(clip_path):
            generate_video_clip(
                client,
                prompt=prompt,
                seconds=sc_seconds,
                size=resolution,
                model="unused",   # kept for compatibility
                out_path=clip_path,
            )

        scene_paths.append(clip_path)
        prog.progress(j / max(1, len(scenes)))

    prog.empty()

    # ---- Stitch scenes
    status.write("Stitching scenes…")
    stitched_path = os.path.join(paths["dir"], "stitched.mp4")

    try:
        safe_concat_mp4s(scene_paths, stitched_path)
    except Exception:
        # Fallback: re-encode then concat
        reencoded = []
        for sp in scene_paths:
            rp = sp.replace(".mp4", "_re.mp4")
            if not os.path.exists(rp):
                reencode_mp4(sp, rp)
            reencoded.append(rp)
        safe_concat_mp4s(reencoded, stitched_path)

    # ---- Add narration audio (FINAL output)
    status.write("Adding narration audio…")
    mux_audio(stitched_path, audio_path, paths["with_audio"])

    status.write("✅ Segment ready (clean MP4, no subs, no logo).")
    status.empty()

    return paths


# ---- UI per segment
for idx, p in enumerate(pairs, start=1):
    title = p["title_guess"]
    label = segment_label(p)
    seg_slug = f"{safe_slug(label)}_{safe_slug(title)}"

    with st.container(border=True):
        st.markdown(f"### {idx}. [{label}] {title}")
        st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

        paths_guess = _segment_output_paths(cache_dir, seg_slug, p["audio_path"])
        built = os.path.exists(paths_guess["with_audio"])

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            do_gen = st.button("⚙️ Generate", key=f"gen_{idx}_{seg_slug}", type="primary")
        with col2:
            if built:
                with open(paths_guess["with_audio"], "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=f,
                        file_name=os.path.basename(paths_guess["with_audio"]),
                        mime="video/mp4",
                        key=f"dl_{idx}_{seg_slug}",
                    )
            else:
                st.button("⬇️ Download MP4", disabled=True)
        with col3:
            st.success("Ready") if built else st.info("Not generated yet")

        if do_gen:
            try:
                st.warning("Generating segment…")
                generate_segment(p)
                st.success("Segment generated.")
                st.rerun()
            except Exception as e:
                st.error("Segment failed.")
                st.exception(e)

# ============================
# PART 4/5 — Local Artifact Browser + Troubleshooting Tools + Guardrails
# ============================

st.divider()
st.subheader("4) Local Artifacts (For Troubleshooting)")

st.caption(
    "If something fails, the per-segment work folders (when present) can help you debug. "
    "Successful segments usually have their workdir cleared after upload to save disk."
)

work_root = os.path.join(_manifest_dir(cache_dir), "work")
_ensure_dir(work_root)

# List segment workdirs that still exist
existing_workdirs = []
try:
    for name in sorted(os.listdir(work_root)):
        p = os.path.join(work_root, name)
        if os.path.isdir(p):
            existing_workdirs.append(p)
except Exception:
    existing_workdirs = []

if not existing_workdirs:
    st.info("No per-segment work folders found (this is normal if uploads succeeded and cleanup ran).")
else:
    pick = st.selectbox(
        "Select a segment work folder",
        options=existing_workdirs,
        format_func=lambda p: os.path.basename(p),
    )

    # Show files inside
    files = []
    for root, _, fns in os.walk(pick):
        for fn in fns:
            files.append(os.path.join(root, fn))
    files = sorted(files)

    st.write(f"Files in `{pick}`:")
    for fp in files[:120]:
        st.write("-", os.path.relpath(fp, pick))
    if len(files) > 120:
        st.caption(f"...and {len(files) - 120} more")

    # Quick preview buttons for common artifacts
    preview_mp4s = [f for f in files if f.lower().endswith(".mp4")]
    preview_jsons = [f for f in files if f.lower().endswith(".json")]
    preview_txts = [f for f in files if f.lower().endswith((".txt", ".md", ".log"))]

    col1, col2, col3 = st.columns(3)

    with col1:
        if preview_mp4s:
            mp4_pick = st.selectbox("Preview MP4", preview_mp4s, key="prev_mp4")
            if st.button("▶️ Show MP4"):
                st.video(mp4_pick)
        else:
            st.caption("No MP4s in this workdir.")

    with col2:
        if preview_jsons:
            js_pick = st.selectbox("View JSON", preview_jsons, key="prev_json")
            if st.button("🔎 Show JSON"):
                try:
                    with open(js_pick, "r", encoding="utf-8") as f:
                        st.code(f.read()[:20000], language="json")
                except Exception as e:
                    st.error(f"Could not read JSON: {e}")
        else:
            st.caption("No JSON files in this workdir.")

    with col3:
        if preview_txts:
            tx_pick = st.selectbox("View text/log", preview_txts, key="prev_txt")
            if st.button("🧾 Show text/log"):
                try:
                    with open(tx_pick, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read()[-20000:], language="text")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
        else:
            st.caption("No text/log files in this workdir.")


# ----------------------------
# Guardrails / reminders
# ----------------------------
st.divider()
st.subheader("5) Guardrails")

st.markdown(
    """
- **Subtitles are NOT burned in** (by design). You'll handle captions in CapCut.
- **OpenAI API key** is entered in the sidebar and stored only in session memory.
- **Spaces credentials** should be stored in Streamlit Secrets or environment variables — never in GitHub.
- If a segment fails:
  1) open the **Status Dashboard** (Part 4),
  2) inspect its error,
  3) optionally inspect remaining workdir artifacts here,
  4) fix the underlying issue,
  5) **Mark PENDING** and rerun **Generate Videos**.
"""
)

