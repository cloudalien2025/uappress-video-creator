# app.py — UAPpress ZIP → Documentary MP4 Studio (ZIP-Only)
import os
import json
import math
import shutil
import streamlit as st
from openai import OpenAI

from video_pipeline import (
    extract_zip_to_temp,
    find_files,
    best_match_pairs,
    read_script_file,
    plan_scenes,
    generate_video_clip,
    transcribe_audio_to_srt,
    write_text,
    safe_slug,
    concat_mp4s,
    mux_audio,
    embed_srt_softsubs,
    get_media_duration_seconds,
    shift_srt,
    renumber_srt_blocks,
    reencode_mp4,
    zip_dir,
)

# ----------------------------
# Helpers
# ----------------------------

def is_intro(p: dict) -> bool:
    t = (p.get("title_guess") or "").lower()
    return ("intro" in t) and ("outro" not in t)

def is_outro(p: dict) -> bool:
    t = (p.get("title_guess") or "").lower()
    return "outro" in t

def is_chapter_one(p: dict) -> bool:
    if p.get("chapter_no") == 1:
        return True
    t = (p.get("title_guess") or "").lower()
    return ("chapter_01" in t) or ("chapter-01" in t) or ("chapter 01" in t) or ("chapter 1" in t)

def segment_label(p: dict) -> str:
    if is_intro(p):
        return "INTRO"
    if is_outro(p):
        return "OUTRO"
    if p.get("chapter_no") is not None:
        return f"CHAPTER {p['chapter_no']}"
    return "SEGMENT"

def segment_scene_cap(title: str, default_cap: int, intro_cap: int, outro_cap: int) -> int:
    t = (title or "").lower()
    if ("intro" in t) and ("outro" not in t):
        return min(default_cap, intro_cap)
    if "outro" in t:
        return min(default_cap, outro_cap)
    return default_cap


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="UAPpress — Documentary MP4 Studio (ZIP)", layout="wide")
st.title("🛸 UAPpress — Documentary MP4 Studio (ZIP-Only)")
st.caption("Upload the ZIP exported by Documentary TTS Studio (scripts + MP3s). The app builds MP4s with burned-in subtitles.")

with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎞 Output")
    resolution = st.selectbox("Resolution", ["1280x720", "1920x1080", "720x1280"], index=0)

    st.caption("Caching saves money + time on reruns. Your video_pipeline.py uses this folder.")
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

    st.caption("Stretch mode always covers full audio length (no cutoff).")

    st.divider()
    st.header("🧠 Models")
    text_model = st.selectbox("Scene planning model", ["gpt-5-mini", "gpt-5"], index=0)
    stt_model = st.selectbox("Transcription model", ["whisper-1"], index=0)
    language = st.text_input("Subtitle language (ISO-639-1)", value="en")

    st.divider()
    st.header("⚙ Run")
    test_mode = st.checkbox("Test Mode (Intro + Chapter 1 + Outro)", value=True)
    force_reencode = st.checkbox("Force re-encode (safer, slower)", value=False)
    build_full_movie = st.checkbox("Also build full combined movie", value=False)


st.subheader("1) Upload Documentary ZIP")
zip_file = st.file_uploader("Upload ZIP from Documentary TTS Studio", type=["zip"])

if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "workdir" not in st.session_state:
    st.session_state.workdir = None
if "out_dir" not in st.session_state:
    st.session_state.out_dir = None

if zip_file:
    zip_bytes = zip_file.getvalue()
    workdir, extract_dir = extract_zip_to_temp(zip_bytes)
    st.session_state.workdir = workdir
    st.session_state.out_dir = os.path.join(workdir, "render_out")
    os.makedirs(st.session_state.out_dir, exist_ok=True)

    scripts, audios = find_files(extract_dir)
    pairs = best_match_pairs(scripts, audios)
    st.session_state.pairs = pairs

    st.success(f"ZIP extracted. Found {len(scripts)} script file(s) and {len(audios)} audio file(s).")


pairs = st.session_state.pairs
if not pairs:
    st.info("Upload a ZIP to detect segments.")
    st.stop()

st.subheader("2) Detected Segments (Intro / Chapters / Outro)")
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

    chapter_final_mp4s = []
    full_srt_parts = []
    cumulative_offset = 0.0

    # rough progress
    total_units = max(1, sum(max_scenes for _ in to_run) + (len(to_run) * 5))
    done_units = 0

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

        # cap scenes for intro/outro
        seg_max_scenes = segment_scene_cap(title, max_scenes, intro_cap, outro_cap)

        # 1) plan scenes
        scenes = plan_scenes(
            client,
            chapter_title=title,
            chapter_text=script_text,
            max_scenes=seg_max_scenes,
            seconds_per_scene=8,  # hint only; rendering uses stretch allocation
            model=text_model,
        )

        # stretch allocation that guarantees no cutoff
        scene_count = max(1, len(scenes))
        total_needed = int(math.ceil(seg_duration)) + 2  # padding prevents abrupt cutoffs
        base = max(6, total_needed // scene_count)
        rem = total_needed % scene_count  # first rem scenes get +1 sec

        st.info(
            f"{segment_label(p)} duration ≈ {seg_duration:.2f}s | scenes={scene_count} "
            f"(cap={seg_max_scenes}) → total video target={total_needed}s "
            f"(base={base}s, remainder={rem}s)"
        )

        write_text(os.path.join(seg_dir, "scene_plan.json"), json.dumps(scenes, ensure_ascii=False, indent=2))

        # 2) generate clips
        scene_paths = []
        for j, sc in enumerate(scenes, start=1):
            sc_no = sc["scene"]
            sc_seconds = base + (1 if j <= rem else 0)

            prompt = sc["prompt"]
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
            done_units += 1
            progress.progress(min(done_units / total_units, 1.0))

        # 3) stitch scenes
        status.write("Stitching scenes…")
        stitched_path = os.path.join(seg_dir, f"{seg_slug}_stitched.mp4")
        try:
            concat_mp4s(scene_paths, stitched_path)
        except Exception:
            status.write("Concat-copy failed; re-encoding scenes for compatibility…")
            reencoded = []
            for sp in scene_paths:
                rp = sp.replace(".mp4", "_reencoded.mp4")
                if not os.path.exists(rp):
                    reencode_mp4(sp, rp)
                reencoded.append(rp)
            concat_mp4s(reencoded, stitched_path)

        done_units += 1
        progress.progress(min(done_units / total_units, 1.0))

        # 4) mux audio
        status.write("Adding narration audio…")
        with_audio = os.path.join(seg_dir, f"{seg_slug}_audio.mp4")
        mux_audio(stitched_path, audio_path, with_audio)

        done_units += 1
        progress.progress(min(done_units / total_units, 1.0))

        # 5) subtitles (SRT)
        status.write("Transcribing subtitles…")
        srt_text = transcribe_audio_to_srt(client, audio_path, model=stt_model, language=language)
        srt_path = os.path.join(seg_dir, f"{seg_slug}.srt")
        write_text(srt_path, srt_text)

        shifted = shift_srt(srt_text, cumulative_offset)
        full_srt_parts.append(shifted)
        cumulative_offset += seg_duration

        done_units += 1
        progress.progress(min(done_units / total_units, 1.0))

        # 6) burn subs into MP4 (video_pipeline handles burn-in)
        status.write("Embedding subtitles into MP4…")
        final_mp4 = os.path.join(seg_dir, f"{seg_slug}_final_subs.mp4")
        embed_srt_softsubs(with_audio, srt_path, final_mp4)

        if force_reencode:
            final_re = os.path.join(seg_dir, f"{seg_slug}_final_subs_reencoded.mp4")
            final_mp4 = reencode_mp4(final_mp4, final_re)

        chapter_final_mp4s.append(final_mp4)

        st.success(f"Segment ready: {os.path.basename(final_mp4)}")

        with open(final_mp4, "rb") as f:
            st.download_button(
                label=f"Download {segment_label(p)} MP4 (burned subs)",
                data=f,
                file_name=os.path.basename(final_mp4),
                mime="video/mp4",
                key=f"dl_seg_mp4_{idx}",
            )
        with open(srt_path, "rb") as f:
            st.download_button(
                label=f"Download {segment_label(p)} SRT",
                data=f,
                file_name=os.path.basename(srt_path),
                mime="text/plain",
                key=f"dl_seg_srt_{idx}",
            )

        done_units += 1
        progress.progress(min(done_units / total_units, 1.0))

    # Build full SRT
    status.write("Building full SRT…")
    full_srt_text = "\n\n".join([p.strip() for p in full_srt_parts if p.strip()])
    full_srt_text = renumber_srt_blocks(full_srt_text)
    full_srt_path = os.path.join(out_dir, "uappress_full_documentary.srt")
    write_text(full_srt_path, full_srt_text)

    # Optionally build full combined MP4
    if build_full_movie and len(chapter_final_mp4s) >= 2:
        status.write("Concatenating full movie…")
        full_mp4 = os.path.join(out_dir, "uappress_full_documentary.mp4")
        try:
            concat_mp4s(chapter_final_mp4s, full_mp4)
        except Exception:
            status.write("Full concat-copy failed; re-encoding segments for compatibility…")
            reencoded = []
            for mp in chapter_final_mp4s:
                rp = mp.replace(".mp4", "_reencoded.mp4")
                if not os.path.exists(rp):
                    reencode_mp4(mp, rp)
                reencoded.append(rp)
            concat_mp4s(reencoded, full_mp4)

        status.write("Embedding full subtitles into full movie…")
        full_mp4_subs = os.path.join(out_dir, "uappress_full_documentary_subs.mp4")
        embed_srt_softsubs(full_mp4, full_srt_path, full_mp4_subs)

        if force_reencode:
            full_mp4_subs_re = os.path.join(out_dir, "uappress_full_documentary_subs_reencoded.mp4")
            full_mp4_subs = reencode_mp4(full_mp4_subs, full_mp4_subs_re)

        st.success("Full movie ready!")
        with open(full_mp4_subs, "rb") as f:
            st.download_button(
                label="Download FULL Documentary MP4 (burned subs)",
                data=f,
                file_name=os.path.basename(full_mp4_subs),
                mime="video/mp4",
                key="dl_full_mp4",
            )
        with open(full_srt_path, "rb") as f:
            st.download_button(
                label="Download FULL Documentary SRT",
                data=f,
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
            data=f,
            file_name="uappress_video_outputs.zip",
            mime="application/zip",
            key="dl_outputs_zip",
        )

    status.write("✅ Done.")
