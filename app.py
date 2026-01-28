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

from video_creator import (
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

# ============================
# PART 3/5 — DigitalOcean Spaces Uploader + Sequential Orchestrator (Generate Videos)
# ============================

# ----------------------------
# Spaces config + upload helper
# ----------------------------
def _get_spaces_config() -> Tuple[str, str, str, str, str]:
    """
    Returns: (do_key, do_secret, do_region, do_bucket, do_public_base)
    Secrets/env only — do NOT request these via UI for a public repo.
    """
    do_key = _sec_get("DO_SPACES_KEY") or os.getenv("DO_SPACES_KEY", "")
    do_secret = _sec_get("DO_SPACES_SECRET") or os.getenv("DO_SPACES_SECRET", "")
    do_region = _sec_get("DO_SPACES_REGION", os.getenv("DO_SPACES_REGION", "nyc3"))
    do_bucket = _sec_get("DO_SPACES_BUCKET") or os.getenv("DO_SPACES_BUCKET", "")
    do_base = _sec_get("DO_SPACES_PUBLIC_BASE") or os.getenv("DO_SPACES_PUBLIC_BASE", "")
    return do_key, do_secret, do_region, do_bucket, do_base


def upload_to_spaces(local_path: str, object_key: str, content_type: str = "video/mp4") -> str:
    """
    Uploads a file to DigitalOcean Spaces and returns a public URL.
    Requires boto3 + botocore in requirements.txt.

    NOTE: If your bucket policy already makes objects public, you can remove ACL below.
    """
    try:
        import boto3
        from botocore.client import Config
    except ModuleNotFoundError:
        raise RuntimeError("Missing boto3/botocore. Add them to requirements.txt and redeploy.")

    do_key, do_secret, do_region, do_bucket, do_base = _get_spaces_config()
    if not all([do_key, do_secret, do_region, do_bucket]):
        raise RuntimeError(
            "Missing DigitalOcean Spaces config. Set DO_SPACES_KEY, DO_SPACES_SECRET, "
            "DO_SPACES_REGION, DO_SPACES_BUCKET (and optionally DO_SPACES_PUBLIC_BASE) in Secrets/env."
        )

    endpoint = f"https://{do_region}.digitaloceanspaces.com"
    s3 = boto3.client(
        "s3",
        region_name=do_region,
        endpoint_url=endpoint,
        aws_access_key_id=do_key,
        aws_secret_access_key=do_secret,
        config=Config(signature_version="s3v4"),
    )

    object_key = (object_key or "").lstrip("/")
    extra = {"ContentType": content_type}

    # If your Space blocks ACLs, remove this and rely on bucket policy.
    extra["ACL"] = "public-read"

    s3.upload_file(local_path, do_bucket, object_key, ExtraArgs=extra)

    if do_base:
        return f"{do_base.rstrip('/')}/{object_key}"
    return f"{endpoint}/{do_bucket}/{object_key}"


# ----------------------------
# Per-segment temp cleanup (keeps manifest)
# ----------------------------
def clear_segment_workdir(cache_dir: str, segment_id: str) -> None:
    """
    Delete only per-segment working artifacts.
    We keep the manifest so the run can resume.
    """
    seg_dir = os.path.join(_manifest_dir(cache_dir), "work", segment_id)
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir, ignore_errors=True)


def segment_object_key(prefix: str, segment_id: str, label: str, title: str) -> str:
    """
    Stable object naming for Spaces.
    """
    safe_name = f"{safe_slug(label)}_{safe_slug(title)}_{segment_id[:10]}.mp4"
    prefix = (prefix or "").strip().strip("/")
    return f"{prefix}/{safe_name}" if prefix else safe_name


# ----------------------------
# UI — Generate Videos
# ----------------------------
st.divider()
st.subheader("2) Generate Videos (Sequential Upload to Spaces)")
st.caption(
    "One-click sequential generation: Intro → Chapters → Outro. "
    "After each MP4: upload to Spaces → clear per-segment cache → continue."
)

# Show basic readiness
do_key, do_secret, do_region, do_bucket, do_base = _get_spaces_config()
spaces_ready = bool(do_key and do_secret and do_region and do_bucket)

if not spaces_ready:
    st.warning(
        "Spaces is not fully configured. Add DO_SPACES_KEY / DO_SPACES_SECRET / DO_SPACES_REGION / DO_SPACES_BUCKET "
        "to Streamlit Secrets or environment variables."
    )

# Options: resume behavior
colR1, colR2, colR3 = st.columns([1, 1, 2])
with colR1:
    resume_done = st.checkbox("Skip segments already marked DONE", value=True)
with colR2:
    stop_on_fail = st.checkbox("Stop on first failure", value=True)
with colR3:
    st.caption("Tip: leave 'Skip DONE' enabled so you can resume after crashes.")

run_btn = st.button("🚀 Generate Videos", type="primary", disabled=not spaces_ready)

# Progress UI placeholders
prog = st.progress(0.0)
status = st.empty()
log_box = st.container(border=True)

if run_btn:
    ordered_urls: List[str] = []
    total = len(pairs)
    manifest = _load_manifest(cache_dir, zip_hash)

    for idx, p in enumerate(pairs, start=1):
        sid = _segment_id(p)
        label = segment_label(p)
        title = p.get("title_guess") or "segment"
        m = manifest.get(sid, {})
        current_status = (m.get("status") or "pending").lower()

        if resume_done and current_status == "done" and (m.get("public_url") or "").strip():
            ordered_urls.append(m["public_url"])
            with log_box:
                st.write(f"✅ Skipping DONE: [{label}] {title}")
            prog.progress(min(1.0, idx / max(1, total)))
            continue

        status.write(f"Generating {idx}/{total}: [{label}] {title}")

        try:
            # Per-segment workdir so we can delete it after upload
            work_root = os.path.join(_manifest_dir(cache_dir), "work")
            seg_workdir = os.path.join(work_root, sid)
            _ensure_dir(seg_workdir)

            # Build MP4 (NO subtitles) — implemented in video_creator.py
            # This function must:
            # - generate visuals
            # - stitch
            # - mux audio
            # - return the local final mp4 path
            mp4_path = build_segment_mp4_no_subs(
                client=client,
                segment=p,
                scripts_by_path=scripts_by_path,
                resolution=resolution,
                cache_dir=cache_dir,
                segment_workdir=seg_workdir,
            )

            if not mp4_path or not os.path.exists(mp4_path):
                raise RuntimeError("Segment builder returned no MP4 path (or file missing).")

            # Upload to Spaces
            obj_key = segment_object_key(upload_prefix, sid, label, title)
            public_url = upload_to_spaces(mp4_path, obj_key)

            # Update manifest
            manifest[sid] = {
                "label": label,
                "title": title,
                "status": "done",
                "public_url": public_url,
                "error": "",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_manifest(cache_dir, zip_hash, manifest)

            ordered_urls.append(public_url)

            # Clear only per-segment cache/workdir to keep disk small
            clear_segment_workdir(cache_dir, sid)

            with log_box:
                st.write(f"✅ DONE: [{label}] {title}")
                st.write(f"   → {public_url}")

        except Exception as e:
            manifest[sid] = {
                "label": label,
                "title": title,
                "status": "failed",
                "public_url": "",
                "error": str(e),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_manifest(cache_dir, zip_hash, manifest)

            with log_box:
                st.write(f"❌ FAILED: [{label}] {title}")
                st.write(f"   error: {e}")

            if stop_on_fail:
                status.write("Stopped due to failure (Stop on fail enabled).")
                break

        prog.progress(min(1.0, idx / max(1, total)))

    st.session_state.job_manifest = manifest
    st.session_state.last_public_urls = ordered_urls

    status.write("Run complete (see URLs below).")

# Persistently show last results
if st.session_state.get("last_public_urls"):
    st.markdown("### Uploaded segment URLs (in order)")
    for u in st.session_state["last_public_urls"]:
        st.write(u)

# ============================
# PART 4/5 — Status Dashboard + Manifest Viewer + Retry Controls
# ============================

st.divider()
st.subheader("3) Status Dashboard (Resume + Troubleshoot)")

st.caption(
    "This dashboard reads the manifest so you can see what completed, what failed, and what’s pending. "
    "Use it to troubleshoot without digging through logs."
)

# Always read from disk (source of truth) so reruns are consistent
manifest = _load_manifest(cache_dir, zip_hash)

# Quick stats
total = len(pairs)
done = sum(1 for v in manifest.values() if (v.get("status") or "").lower() == "done")
failed = sum(1 for v in manifest.values() if (v.get("status") or "").lower() == "failed")
pending = total - done - failed

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("Done", done)
c3.metric("Failed", failed)
c4.metric("Pending", pending)

# Controls
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    show_only = st.selectbox("Filter", ["All", "Pending", "Failed", "Done"], index=0)
with colB:
    if st.button("🔄 Reload manifest"):
        st.rerun()
with colC:
    st.caption("Tip: after fixing an error, mark the failed segment back to PENDING, then rerun Generate Videos.")

# Helper to render a single segment row
def _should_show(status: str) -> bool:
    s = (status or "pending").lower()
    if show_only == "All":
        return True
    if show_only == "Pending":
        return s == "pending"
    if show_only == "Failed":
        return s == "failed"
    if show_only == "Done":
        return s == "done"
    return True


# Display each segment in order (pairs is already ordered Intro → Chapters → Outro)
for i, p in enumerate(pairs, start=1):
    sid = _segment_id(p)
    m = manifest.get(sid, {})
    status = (m.get("status") or "pending").lower()
    if not _should_show(status):
        continue

    label = segment_label(p)
    title = p.get("title_guess") or "segment"
    url = (m.get("public_url") or "").strip()
    err = (m.get("error") or "").strip()
    updated = (m.get("updated_at") or "").strip()

    with st.container(border=True):
        st.markdown(f"**{i}. [{label}] {title}**")
        st.write(f"Status: `{status.upper()}`" + (f" • Updated: {updated}" if updated else ""))

        # Show pairing paths for debugging
        st.code(
            f"SCRIPT: {p.get('script_path','')}\nAUDIO:  {p.get('audio_path','')}",
            language="text",
        )

        if url:
            st.write("Public URL:")
            st.write(url)

        if err:
            st.error("Last error:")
            st.code(err, language="text")

        # Actions
        a1, a2, a3 = st.columns([1, 1, 2])

        with a1:
            if st.button("Mark PENDING", key=f"mk_pending_{sid}"):
                m2 = manifest.get(sid, {})
                m2["status"] = "pending"
                m2["public_url"] = ""
                m2["error"] = ""
                m2["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                manifest[sid] = m2
                _save_manifest(cache_dir, zip_hash, manifest)
                st.success("Marked pending.")
                st.rerun()

        with a2:
            if st.button("Clear Workdir", key=f"clr_work_{sid}"):
                clear_segment_workdir(cache_dir, sid)
                st.success("Workdir cleared.")
                st.rerun()

        with a3:
            st.caption("Use 'Generate Videos' to run sequentially. Mark a failed one to PENDING to retry.")


# Optional: export manifest JSON
st.divider()
st.subheader("Export")

manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
st.download_button(
    "⬇️ Download manifest JSON",
    data=manifest_json.encode("utf-8"),
    file_name=f"manifest_{zip_hash}.json",
    mime="application/json",
)

# ============================
# PART 5/5 — Local Artifact Browser + Troubleshooting Tools + Guardrails
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

