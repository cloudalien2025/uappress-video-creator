# app.py — UAPpress ZIP → Documentary MP4 Studio (ZIP-only, capped intro/outro, stretch-safe)
import os
import json
import math
import shutil
import tempfile
import subprocess
import hashlib
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
    shift_srt,
    renumber_srt_blocks,
    reencode_mp4,
    zip_dir,
)

# ----------------------------
# PATCHED: Safe concat that avoids fragile f-string escaping
# ----------------------------

def _ffmpeg_escape_path(p: str) -> str:
    # FFmpeg concat demuxer expects: file 'path'
    # Inside single quotes, FFmpeg escaping uses: '\''  (close-quote, escaped quote, reopen)
    return (p or "").replace("'", "'\\''")

def safe_concat_mp4s(paths: list[str], out_path: str) -> None:
    if not paths:
        raise ValueError("safe_concat_mp4s: no input paths provided")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"safe_concat_mp4s: missing input file: {p}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in paths:
            safe_path = _ffmpeg_escape_path(os.path.abspath(p))
            f.write(f"file '{safe_path}'\n")

    try:
        cmd = [
            ffmpeg,
            "-y",
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
# Pairing (robust intro/outro + chapter number matching)
# ----------------------------

def _norm_tokens(s: str) -> set[str]:
    import re
    s = (s or "").lower()
    toks = set(re.split(r"[^a-z0-9]+", s))
    toks.discard("")
    return toks

def _chapter_no_from_name(name: str) -> int | None:
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

def pair_segments(scripts: list[str], audios: list[str]) -> list[dict]:
    intro_script = next((s for s in scripts if _is_intro_name(s)), None)
    outro_script = next((s for s in scripts if _is_outro_name(s)), None)

    intro_audio = next((a for a in audios if _is_intro_name(a)), None)
    outro_audio = next((a for a in audios if _is_outro_name(a)), None)

    scripts_left = [s for s in scripts if s not in {intro_script, outro_script}]
    audios_left = [a for a in audios if a not in {intro_audio, outro_audio}]

    audio_by_no: dict[int, list[str]] = {}
    for a in audios_left:
        n = _chapter_no_from_name(a)
        if n is not None:
            audio_by_no.setdefault(n, []).append(a)

    used_audio = set()
    pairs: list[dict] = []

    if intro_script and intro_audio:
        pairs.append({
            "chapter_no": 0,
            "title_guess": os.path.splitext(os.path.basename(intro_script))[0],
            "script_path": intro_script,
            "audio_path": intro_audio,
        })
        used_audio.add(intro_audio)

    if outro_script and outro_audio:
        pairs.append({
            "chapter_no": 9998,
            "title_guess": os.path.splitext(os.path.basename(outro_script))[0],
            "script_path": outro_script,
            "audio_path": outro_audio,
        })
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
            pairs.append({
                "chapter_no": sn,
                "title_guess": os.path.splitext(os.path.basename(s))[0],
                "script_path": s,
                "audio_path": chosen,
            })

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
# Segment logic
# ----------------------------

def is_intro(p: dict) -> bool:
    return _is_intro_name(p.get("title_guess", "")) or _is_intro_name(p.get("script_path", "")) or _is_intro_name(p.get("audio_path", ""))

def is_outro(p: dict) -> bool:
    return _is_outro_name(p.get("title_guess", "")) or _is_outro_name(p.get("script_path", "")) or _is_outro_name(p.get("audio_path", ""))

def is_chapter_one(p: dict) -> bool:
    return p.get("chapter_no") == 1 or "chapter_01" in (p.get("title_guess") or "").lower() or "chapter-01" in (p.get("title_guess") or "").lower()

def segment_label(p: dict) -> str:
    if is_intro(p):
        return "INTRO"
    if is_outro(p):
        return "OUTRO"
    if p.get("chapter_no") is not None:
        return f"CHAPTER {p['chapter_no']}"
    return "SEGMENT"

def segment_scene_cap(title: str, default_cap: int, intro_cap: int, outro_cap: int) -> int:
    if _is_intro_name(title):
        return min(default_cap, intro_cap)
    if _is_outro_name(title):
        return min(default_cap, outro_cap)
    return default_cap


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="UAPpress — Documentary MP4 Studio (ZIP)", layout="wide")
st.title("🛸 UAPpress — Documentary MP4 Studio (ZIP-only)")
st.caption("Upload the ZIP exported by Documentary TTS Studio (scripts + audio). Generates a full YouTube-ready MP4 with subtitles burned in once at the end.")

with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎞 Output")
    resolution = st.selectbox("Resolution", ["1280x720", "1920x1080", "720x1280"], index=0)

    cache_dir = st.text_input("Image/clip cache folder", value=os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache"))
    os.environ["UAPPRESS_CACHE_DIR"] = cache_dir

    if st.button("🧹 Clear cache"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        st.success("Cache cleared.")

    st.divider()
    st.header("🎬 Scenes")
    max_scenes = st.slider("Max scenes per Chapter", 3, 14, 8)
    intro_cap = st.slider("Intro scene cap", 1, 6, 2)
    outro_cap = st.slider("Outro scene cap", 1, 6, 2)

    st.divider()
    st.header("🧠 Models")
    text_model = st.selectbox("Scene planning model", ["gpt-5-mini", "gpt-5"], index=0)
    stt_model = st.selectbox("Transcription model", ["whisper-1"], index=0)
    language = st.text_input("Subtitle language (ISO-639-1)", value="en")

    st.divider()
    st.header("⚙ Run")
    test_mode = st.checkbox("Test Mode (Intro + Chapter 1 + Outro)", value=True)
    force_reencode = st.checkbox("Force re-encode (safer, slower)", value=False)

    # ✅ Option B: make full combined movie default ON
    build_full_movie = st.checkbox("Also build full combined movie", value=True)

    st.divider()
    st.header("🧩 Project")
    if st.button("♻️ Reset Project (re-extract ZIP)"):
        for k in ["pairs", "workdir", "out_dir", "zip_hash"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Project reset.")
        st.rerun()

st.subheader("1) Upload Documentary ZIP")
zip_file = st.file_uploader("Upload ZIP from Documentary TTS Studio", type=["zip"])

st.session_state.setdefault("pairs", [])
st.session_state.setdefault("workdir", None)
st.session_state.setdefault("out_dir", None)
st.session_state.setdefault("zip_hash", None)

# ✅ Extract ZIP ONLY when it changes (rerun-safe)
if zip_file:
    zip_bytes = zip_file.getvalue()
    zip_hash = hashlib.sha256(zip_bytes).hexdigest()

    if st.session_state.zip_hash != zip_hash:
        st.session_state.zip_hash = zip_hash

        workdir, extract_dir = extract_zip_to_temp(zip_bytes)
        st.session_state.workdir = workdir
        st.session_state.out_dir = os.path.join(workdir, "render_out")
        os.makedirs(st.session_state.out_dir, exist_ok=True)

        scripts, audios = find_files(extract_dir)
        pairs = pair_segments(scripts, audios)
        st.session_state.pairs = pairs

        st.success(f"ZIP extracted. Found {len(scripts)} script file(s) and {len(audios)} audio file(s).")
    else:
        st.info("Same ZIP detected — using existing extracted files and outputs.")

pairs = st.session_state.pairs
if not pairs:
    st.info("Upload a ZIP to detect segments.")
    st.stop()

st.subheader("2) Detected Segments (Pairing)")
for i, p in enumerate(pairs, start=1):
    st.write(f"**{i}. [{segment_label(p)}] {p['title_guess']}**")
    st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

# Choose run list
if test_mode:
    intro = next((p for p in pairs if is_intro(p)), None)
    ch1 = next((p for p in pairs if is_chapter_one(p)), None) or (pairs[0] if pairs else None)
    outro = next((p for p in pairs if is_outro(p)), None)
    to_run = [p for p in [intro, ch1, outro] if p is not None]
else:
    to_run = pairs

st.subheader("3) Build")
st.warning("First run costs money (images + transcription). Reruns should be fast/cheap thanks to caching.")

if st.button("🚀 Build MP4(s) from ZIP"):
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar.")
        st.stop()

    client = OpenAI()
    out_dir = st.session_state.out_dir

    progress = st.progress(0.0)
    status = st.empty()

    # We'll build per-segment MP4s with audio (no burned subs),
    # then build ONE full SRT + ONE final MP4 with subs burned in.
    segment_with_audio_mp4s: list[str] = []
    full_srt_parts = []
    cumulative_offset = 0.0

    total_units = max(1, sum(max_scenes for _ in to_run) + (len(to_run) * 6))
    done = 0

    for idx, p in enumerate(to_run, start=1):
        title = p["title_guess"]
        seg_slug = f"seg_{idx:02d}_{safe_slug(title)}"
        seg_dir = os.path.join(out_dir, seg_slug)
        os.makedirs(seg_dir, exist_ok=True)

        status.write(f"### {idx}/{len(to_run)} — {segment_label(p)}: {title}")

        script_text = read_script_file(p["script_path"])
        audio_path = p["audio_path"]

        seg_duration = float(get_media_duration_seconds(audio_path))
        if seg_duration <= 0:
            st.error(f"Could not read duration for: {audio_path}")
            st.stop()

        seg_max_scenes = segment_scene_cap(title, max_scenes, intro_cap, outro_cap)

        # 1) plan scenes
        scenes = plan_scenes(
            client,
            chapter_title=title,
            chapter_text=script_text,
            max_scenes=seg_max_scenes,
            seconds_per_scene=8,
            model=text_model,
        )

        scene_count = max(1, len(scenes))

        # Stretch allocation that prevents cutoff:
        total_needed = int(math.ceil(seg_duration)) + 2
        base = max(6, total_needed // scene_count)
        rem = total_needed % scene_count

        st.info(
            f"{segment_label(p)} audio ≈ {seg_duration:.2f}s | scenes={scene_count} (cap={seg_max_scenes}) "
            f"→ target={total_needed}s (base={base}s, remainder={rem}s)"
        )

        write_text(os.path.join(seg_dir, "scene_plan.json"), json.dumps(scenes, ensure_ascii=False, indent=2))

        # 2) generate clips
        scene_paths = []
        for j, sc in enumerate(scenes, start=1):
            sc_no = int(sc.get("scene", j))
            sc_seconds = base + (1 if j <= rem else 0)
            prompt = str(sc.get("prompt", "")).strip()

            clip_path = os.path.join(seg_dir, f"scene_{sc_no:02d}.mp4")

            status.write(f"Generating visuals — {segment_label(p)} Scene {sc_no} ({sc_seconds}s)")
            if not os.path.exists(clip_path):
                generate_video_clip(
                    client,
                    prompt=prompt,
                    seconds=sc_seconds,
                    size=resolution,
                    model="unused",
                    out_path=clip_path,
                )

            scene_paths.append(clip_path)
            done += 1
            progress.progress(min(done / total_units, 1.0))

        # 3) stitch scenes
        status.write("Stitching scenes…")
        stitched_path = os.path.join(seg_dir, f"{seg_slug}_stitched.mp4")
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

        done += 1
        progress.progress(min(done / total_units, 1.0))

        # 4) mux audio (this is the segment MP4 we keep)
        status.write("Adding narration audio…")
        with_audio = os.path.join(seg_dir, f"{seg_slug}_with_audio.mp4")
        mux_audio(stitched_path, audio_path, with_audio)

        if force_reencode:
            with_audio_re = os.path.join(seg_dir, f"{seg_slug}_with_audio_reencoded.mp4")
            with_audio = reencode_mp4(with_audio, with_audio_re)

        segment_with_audio_mp4s.append(with_audio)

        done += 1
        progress.progress(min(done / total_units, 1.0))

        # 5) subtitles per segment (SRT only; no burn here)
        status.write("Transcribing subtitles…")
        srt_text = transcribe_audio_to_srt(client, audio_path, model=stt_model, language=language)
        srt_path = os.path.join(seg_dir, f"{seg_slug}.srt")
        write_text(srt_path, srt_text)

        shifted = shift_srt(srt_text, cumulative_offset)
        full_srt_parts.append(shifted)
        cumulative_offset += seg_duration

        done += 1
        progress.progress(min(done / total_units, 1.0))

        st.success(f"Ready (segment, no burned subs): {os.path.basename(with_audio)}")

        # Optional segment downloads (still useful as backups)
        with open(with_audio, "rb") as f:
            st.download_button(
                label=f"Download {segment_label(p)} MP4 (audio only)",
                data=f.read(),
                file_name=os.path.basename(with_audio),
                mime="video/mp4",
                key=f"dl_seg_mp4_{idx}",
            )
        with open(srt_path, "rb") as f:
            st.download_button(
                label=f"Download {segment_label(p)} SRT",
                data=f.read(),
                file_name=os.path.basename(srt_path),
                mime="text/plain",
                key=f"dl_seg_srt_{idx}",
            )

        done += 1
        progress.progress(min(done / total_units, 1.0))

    # Build full SRT
    status.write("Building full SRT…")
    full_srt_text = "\n\n".join([p.strip() for p in full_srt_parts if p.strip()])
    full_srt_text = renumber_srt_blocks(full_srt_text)
    full_srt_path = os.path.join(out_dir, "uappress_full_documentary.srt")
    write_text(full_srt_path, full_srt_text)

    # Build full combined movie (now default ON)
    if build_full_movie and len(segment_with_audio_mp4s) >= 1:
        status.write("Concatenating full movie (audio segments)…")
        full_mp4 = os.path.join(out_dir, "uappress_full_documentary.mp4")
        try:
            safe_concat_mp4s(segment_with_audio_mp4s, full_mp4)
        except Exception:
            status.write("Full concat-copy failed; re-encoding segments for compatibility…")
            reencoded = []
            for mp in segment_with_audio_mp4s:
                rp = mp.replace(".mp4", "_reencoded.mp4")
                if not os.path.exists(rp):
                    reencode_mp4(mp, rp)
                reencoded.append(rp)
            safe_concat_mp4s(reencoded, full_mp4)

        status.write("Burning subtitles ONCE into full movie…")
        full_mp4_subs = os.path.join(out_dir, "uappress_full_documentary_subs.mp4")
        embed_srt_softsubs(full_mp4, full_srt_path, full_mp4_subs)

        if force_reencode:
            full_re = os.path.join(out_dir, "uappress_full_documentary_subs_reencoded.mp4")
            full_mp4_subs = reencode_mp4(full_mp4_subs, full_re)

        st.success("✅ Full YouTube-ready movie is ready!")

        with open(full_mp4_subs, "rb") as f:
            st.download_button(
                label="Download FULL Documentary MP4 (burned subs)",
                data=f.read(),
                file_name=os.path.basename(full_mp4_subs),
                mime="video/mp4",
                key="dl_full_mp4",
            )
        with open(full_srt_path, "rb") as f:
            st.download_button(
                label="Download FULL Documentary SRT",
                data=f.read(),
                file_name=os.path.basename(full_srt_path),
                mime="text/plain",
                key="dl_full_srt",
            )

    # Package outputs
    status.write("Packaging outputs into ZIP…")
    out_zip_path = os.path.join(out_dir, "uappress_video_outputs.zip")
    zip_dir(out_dir, out_zip_path)

    st.success("All outputs packaged.")
    with open(out_zip_path, "rb") as f:
        st.download_button(
            label="Download Output ZIP (MP4s + SRTs + plans)",
            data=f.read(),
            file_name="uappress_video_outputs.zip",
            mime="application/zip",
            key="dl_outputs_zip",
        )

    status.write("✅ Done.")
