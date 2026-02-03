# ============================
    # app.py — UAPpress Video Creator
    # Cinematic Shorts (Sora-driven, prompt-first) — v2
    #
    # Adds:
    # 1) Locked Shorts length presets (7s / 12s / 18s)
    # 2) Sora "house style" controls (camera / lighting / grain-realism)
    # 3) Style toggle: Cinematic realism vs Archival reenactment
    # ============================

import streamlit as st

 import video_pipeline as vp

    st.set_page_config(page_title="UAPpress Video Creator", layout="wide")

    # ---- Sidebar ----
    st.sidebar.header("🔑 API Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key")

    st.sidebar.header("🎞️ Output")
    output_mode = st.sidebar.selectbox(
        "Output mode",
        ["Long-form (16:9)", "Shorts (9:16)"],
        key="output_mode",
    )

    if output_mode == "Long-form (16:9)":
        resolution = st.sidebar.selectbox(
            "Resolution (16:9)",
            ["1280x720", "1920x1080"],
            index=0,
        )
    else:
        resolution = "1080x1920"
        st.sidebar.caption("Shorts are rendered in vertical 9:16 (1080×1920).")

    st.title("UAPpress — Video Generation Studio")

    # ============================
    # Shorts (9:16)
    # ============================
    if output_mode == "Shorts (9:16)":
        st.header("🎬 Generate Shorts")

        shorts_mode = st.radio(
            "Shorts Creation Method",
            ["Documentary (image-based)", "Cinematic (Sora)"],
            index=1,
            horizontal=True,
        )

        if shorts_mode == "Cinematic (Sora)":
            st.subheader("✍️ Sora Prompt Studio")

            # (3) Future toggle — style modes
            style_mode = st.radio(
                "Style Mode",
                ["Cinematic realism", "Archival reenactment"],
                index=0,
                horizontal=True,
                help=(
                    "Cinematic realism: modern documentary-grade photorealism.
"
                    "Archival reenactment: period film stock look (grain, scratches, subtle flicker)."
                ),
            )

            # (1) Locked length presets
            length_preset = st.selectbox(
                "Shorts length preset",
                ["7s (Ultra-hook)", "12s (Narrative beat)", "18s (Mini-story)"],
                index=1,
            )
            target_seconds = {"7s (Ultra-hook)": 7, "12s (Narrative beat)": 12, "18s (Mini-story)": 18}[length_preset.split(" ")[0]]

            # (2) House style controls
            st.markdown("**House Style Controls**")
            c1, c2, c3 = st.columns(3)
            with c1:
                camera_rule = st.selectbox(
                    "Camera motion",
                    ["Slow dolly-in", "Slow lateral drift", "Locked-off (subtle micro-jitter)", "Handheld (restrained)"],
                    index=0,
                )
            with c2:
                lighting_rule = st.selectbox(
                    "Lighting",
                    ["Moody low-key", "Neutral documentary", "Harsh fluorescent (institutional)", "Sodium-vapor night exterior"],
                    index=0,
                )
            with c3:
                grain_rule = st.selectbox(
                    "Grain / realism",
                    ["Clean photoreal", "Subtle film grain", "Heavy archival grain + scratches"],
                    index=1,
                )

            st.markdown("**Core Visual Prompt** (you’ll craft this with ChatGPT and paste it here)")
            user_prompt = st.text_area(
                "Sora Visual Prompt",
                height=210,
                placeholder=(
                    "Describe a short cinematic sequence designed for vertical 9:16.
"
                    "Focus on mood, camera movement, lighting, environment, and atmosphere.
"
                    "Avoid dialogue, subtitles, logos, or on-screen text.
"
                    "Serious investigative documentary tone."
                ),
                key="sora_prompt",
            )

            st.markdown("**Optional outputs**")
            o1, o2, o3 = st.columns(3)
            with o1:
                narration_mode = st.selectbox(
                    "Narration overlay",
                    ["None", "Use uploaded MP3 (trim to fit)"],
                    index=0,
                )
            with o2:
                captions_mode = st.selectbox(
                    "Captions",
                    ["None", "Minimal", "Full"],
                    index=0,
                )
            with o3:
                aspect_note = st.selectbox(
                    "Aspect guidance",
                    ["Native vertical 9:16", "Keep subject centered (safe margins)"],
                    index=0,
                    help="Affects prompt guidance only.",
                )

            # Build final prompt the same way the pipeline would.
            final_prompt = vp.build_sora_prompt(
                user_prompt=user_prompt,
                style_mode=style_mode,
                camera_rule=camera_rule,
                lighting_rule=lighting_rule,
                grain_rule=grain_rule,
                target_seconds=target_seconds,
                aspect_note=aspect_note,
            )

            st.markdown("### ✅ Final Sora Prompt (auto-assembled)")
            st.text_area(
                "Copy this into Sora (or use the Generate button once backend is enabled)",
                value=final_prompt,
                height=220,
                key="final_sora_prompt",
            )

            st.divider()

            if st.button("🚀 Generate Cinematic Short (Sora)", use_container_width=True):
                if not api_key:
                    st.error("Please enter your OpenAI API key in the sidebar.")
                    st.stop()
                if not user_prompt.strip():
                    st.error("Sora prompt is required.")
                    st.stop()

                job = vp.prepare_sora_short_job(
                    api_key=api_key,
                    sora_prompt=final_prompt,
                    target_seconds=target_seconds,
                    resolution=resolution,
                    narration_mode=narration_mode,
                    captions_mode=captions_mode,
                )

                st.success("Sora short job prepared.")
                st.json(job)

                st.info(
                    "If your environment supports the Sora API, the backend can execute this job.
"
                    "If not, copy the Final Sora Prompt above into Sora and export a 9:16 clip."
                )

        else:
            st.info(
                "Documentary Shorts (image-based) should be generated from scratch in 9:16.
"
                "If you want, we can add a dedicated pipeline that splits MP3+script into 7/12/18s Shorts packs."
            )

    # ============================
    # Long-form placeholder (kept minimal here)
    # ============================
    else:
        st.header("🎥 Long-form Documentary")
        st.info("Your existing long-form pipeline remains unchanged in your main app build.")

    # ============================
    # Maintenance
    # ============================
    with st.sidebar.expander("🧹 Maintenance", expanded=False):
        if st.button("Clear Streamlit cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared. Rerun the app.")
