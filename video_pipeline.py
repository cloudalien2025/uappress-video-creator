# app.py — UAPpress Video Creator (ZIP → Documentary MP4 with ALWAYS subtitles)
import os
import json
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

st.set_page_config(page_title="UAPpress Video Creator", layout="wide")
st.title("🎬 UAPpress — ZIP → Documentary MP4 (Subtitles Always)")

st.caption(
    "Upload the ZIP exported by Documentary TTS Studio (chapter scripts + MP3s). "
    "This app generates AI visuals per chapter scene, stitches everything into MP4, "
    "and ALWAYS generates + embeds subtitles."
)

with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎥 Video")
    video_model = st.selectbox("Video model", ["sora-2", "sora-2-pro"], index=0)  # kept for UI compatibility
    size = st.selectbox("Resolution", ["1280x720", "720x1280"], index=0)
    seconds = st.selectbox("Seconds per scene (ignored in stretch mode)", [4, 8, 12], index=1)
    max_scenes = st.slider("Max scenes per chapter", 3, 14, 8)

    st.divider()
    st.header("🧠 Scene Planner")
    text_model = st.selectbox("Text model", ["gpt-5-mini", "gpt-5"], index=0)

    st.divider()
    st.header("📝 Subtitles")
    st.caption("Subtitles are ALWAYS generated and embedded.")
    stt_model = st.selectbox("Transcription model", ["whisper-1"], index=0)
    language = st.text_input("Language (ISO-639-1)", value="en")

    st.divider()
    st.header("⚙️ Run Mode")
    only_first_chapter = st.checkbox("Only render first chapter (testing)", value=True)
    force_reencode = st.checkbox("Force re-encode outputs (slower, more compatible)", value=False)

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

st.subheader("2) Review Chapter Pairing")
pairs = st.session_state.pairs
if not pairs:
    st.info("Upload a ZIP to detect chapters.")
    st.stop()

for i, p in enumerate(pairs, start=1):
    st.write(f"**{i}. {p['title_guess']}**")
    st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

st.subheader("3) Build Videos (Subtitles Always)")
st.warning("This pipeline can be expensive if you generate lots of scenes. Testing with 1 chapter first is recommended.")

if st.button("🚀 Build MP4 + Subtitles"):
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar.")
        st.stop()

    client = OpenAI()
    out_dir = st.session_state.out_dir
    to_run = pairs[:1] if only_first_chapter else pairs

    progress = st.progress(0.0)
    status = st.empty()

    chapter_final_mp4s = []
    full_srt_parts = []
    cumulative_offset = 0.0

    total_units = sum(max_scenes for _ in to_run) + (len(to_run) * 4)
    done_units = 0

    for idx, p in enumerate(to_run, start=1):
        title = p["title_guess"]
        chapter_slug = f"chapter_{idx:02d}_{safe_slug(title)}"
        chapter_dir = os.path.join(out_dir, chapter_slug)
        os.makedirs(chapter_dir, exist_ok=True)

        status.write(f"### Chapter {idx}: {title}")

        chapter_text = read_script_file(p["script_path"])
        audio_path = p["audio_path"]

        # ✅ Measure chapter duration (used for stretch method)
        chapter_duration = float(get_media_duration_seconds(audio_path))

        # 1) plan scenes
        scenes = plan_scenes(
            client,
            chapter_title=title,
            chapter_text=chapter_text,
            max_scenes=max_scenes,
            seconds_per_scene=int(seconds),  # planner hint only (we override actual rendering length)
            model=text_model,
        )

        # ✅ Stretch method: divide chapter audio evenly across scenes
        scene_count = max(1, len(scenes))
        # Minimum 6 seconds so a very short chapter doesn't make 1-second clips
        target_scene_seconds = max(6, int(round(chapter_duration / scene_count)))

        st.write("**Scene plan**")
        st.json(scenes)
        write_text(os.path.join(chapter_dir, "scene_plan.json"), json.dumps(scenes, ensure_ascii=False, indent=2))

        st.info(
            f"Stretch mode ON: Chapter audio ≈ {int(chapter_duration)}s, "
            f"{scene_count} scenes → {target_scene_seconds}s per scene."
        )

        # 2) generate clips
        scene_paths = []
        for sc in scenes:
            sc_no = sc["scene"]

            # ✅ OVERRIDE per-scene seconds so visuals cover full chapter length
            sc_seconds = target_scene_seconds

            prompt = sc["prompt"]
            clip_path = os.path.join(chapter_dir, f"scene_{sc_no:02d}.mp4")

            status.write(f"Generating visuals — Chapter {idx} Scene {sc_no} ({sc_seconds}s)")
            if not os.path.exists(clip_path):
                generate_video_clip(
                    client,
                    prompt=prompt,
                    seconds=sc_seconds,
                    size=size,
                    model=video_model,   # kept for compatibility (image pipeline ignores this)
                    out_path=clip_path,
                )

            scene_paths.append(clip_path)
            done_units += 1
            progress.progress(min(done_units / max(total_units, 1), 1.0))

        # 3) stitch scenes
        status.write(f"Stitching scenes — Chapter {idx}")
        stitched_path = os.path.join(chapter_dir, f"{chapter_slug}_stitched.mp4")
        try:
            concat_mp4s(scene_paths, stitched_path)
        except Exception:
            status.write("Concat-copy failed; re-encoding scenes for compatibility…")
            reencoded_scenes = []
            for sp in scene_paths:
                rp = sp.replace(".mp4", "_reencoded.mp4")
                if not os.path.exists(rp):
                    reencode_mp4(sp, rp)
                reencoded_scenes.append(rp)
            concat_mp4s(reencoded_scenes, stitched_path)

        done_units += 1
        progress.progress(min(done_units / max(total_units, 1), 1.0))

        # 4) mux audio
        status.write(f"Adding narration audio — Chapter {idx}")
        chapter_with_audio = os.path.join(chapter_dir, f"{chapter_slug}_audio.mp4")
        mux_audio(stitched_path, audio_path, chapter_with_audio)

        done_units += 1
        progress.progress(min(done_units / max(total_units, 1), 1.0))

        # 5) ALWAYS generate subtitles (SRT) from chapter audio
        status.write(f"Transcribing subtitles — Chapter {idx}")
        srt_text = transcribe_audio_to_srt(client, audio_path, model=stt_model, language=language)
        srt_path = os.path.join(chapter_dir, f"{chapter_slug}.srt")
        write_text(srt_path, srt_text)

        # Add to full SRT with time offset
        shifted = shift_srt(srt_text, cumulative_offset)
        full_srt_parts.append(shifted)
        cumulative_offset += chapter_duration

        done_units += 1
        progress.progress(min(done_units / max(total_units, 1), 1.0))

        # 6) ALWAYS embed subtitles (your video_pipeline embed method controls whether soft or burned-in)
        status.write(f"Embedding subtitles into MP4 — Chapter {idx}")
        chapter_final = os.path.join(chapter_dir, f"{chapter_slug}_final_subs.mp4")
        embed_srt_softsubs(chapter_with_audio, srt_path, chapter_final)

        if force_reencode:
            chapter_final_re = os.path.join(chapter_dir, f"{chapter_slug}_final_subs_reencoded.mp4")
            chapter_final = reencode_mp4(chapter_final, chapter_final_re)

        chapter_final_mp4s.append(chapter_final)

        st.success(f"Chapter ready: {os.path.basename(chapter_final)}")
        with open(chapter_final, "rb") as f:
            st.download_button(
                label=f"Download Chapter {idx} MP4 (with subtitles)",
                data=f,
                file_name=os.path.basename(chapter_final),
                mime="video/mp4",
                key=f"dl_ch_{idx}",
            )
        with open(srt_path, "rb") as f:
            st.download_button(
                label=f"Download Chapter {idx} SRT",
                data=f,
                file_name=os.path.basename(srt_path),
                mime="text/plain",
                key=f"dl_srt_{idx}",
            )

    # Build full SRT
    status.write("Building full SRT…")
    full_srt_text = "\n\n".join([p.strip() for p in full_srt_parts if p.strip()])
    full_srt_text = renumber_srt_blocks(full_srt_text)

    full_srt_path = os.path.join(out_dir, "uappress_documentary_full.srt")
    write_text(full_srt_path, full_srt_text)

    # Concatenate chapters into full documentary MP4
    if len(chapter_final_mp4s) > 1 and not only_first_chapter:
        status.write("Concatenating full documentary MP4…")
        full_mp4 = os.path.join(out_dir, "uappress_documentary_final.mp4")

        try:
            concat_mp4s(chapter_final_mp4s, full_mp4)
        except Exception:
            status.write("Full concat-copy failed; re-encoding chapters for compatibility…")
            reencoded_ch = []
            for mp in chapter_final_mp4s:
                rp = mp.replace(".mp4", "_reencoded.mp4")
                if not os.path.exists(rp):
                    reencode_mp4(mp, rp)
                reencoded_ch.append(rp)
            concat_mp4s(reencoded_ch, full_mp4)

        # Embed full subtitles into final MP4
        status.write("Embedding full subtitles into final MP4…")
        full_mp4_subs = os.path.join(out_dir, "uappress_documentary_final_subs.mp4")
        embed_srt_softsubs(full_mp4, full_srt_path, full_mp4_subs)

        if force_reencode:
            full_mp4_subs_re = os.path.join(out_dir, "uappress_documentary_final_subs_reencoded.mp4")
            full_mp4_subs = reencode_mp4(full_mp4_subs, full_mp4_subs_re)

        st.success("Full documentary ready!")
        with open(full_mp4_subs, "rb") as f:
            st.download_button(
                label="Download Full Documentary MP4 (with subtitles)",
                data=f,
                file_name=os.path.basename(full_mp4_subs),
                mime="video/mp4",
            )
        with open(full_srt_path, "rb") as f:
            st.download_button(
                label="Download Full Documentary SRT",
                data=f,
                file_name=os.path.basename(full_srt_path),
                mime="text/plain",
            )
    else:
        st.info("Rendered only one chapter. Full documentary concat is skipped in testing mode.")

    # Output ZIP
    status.write("Packaging outputs into ZIP…")
    out_zip_path = os.path.join(out_dir, "uappress_video_outputs.zip")
    zip_dir(out_dir, out_zip_path)

    st.success("All outputs packaged.")
    with open(out_zip_path, "rb") as f:
        st.download_button(
            label="Download Output ZIP (MP4s + SRTs + scene plans)",
            data=f,
            file_name="uappress_video_outputs.zip",
            mime="application/zip",
        )

    status.write("✅ Done.")
