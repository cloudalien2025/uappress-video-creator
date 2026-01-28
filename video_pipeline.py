# ============================
# PART 2/5 — Scene Planning (Text → JSON)
#   (NO Whisper / NO SRT — CapCut handles captions)
# ============================

SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration segment into a list of short visual scenes for AI generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt. "
    "Style: cinematic documentary b-roll / reenactment vibes; realistic lighting; subtle motion notes only. "
    "Avoid brand names, copyrighted characters, celebrity likeness, and explicit violence/gore. "
    "Keep prompts concise (1–3 sentences)."
)

def _extract_json_list(text: str) -> str:
    t = (text or "").strip()
    # strip code fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)

    i = t.find("[")
    j = t.rfind("]")
    if i != -1 and j != -1 and j > i:
        return t[i : j + 1]
    return t

def plan_scenes(
    client: OpenAI,
    chapter_title: str,
    chapter_text: str,
    *,
    max_scenes: int,
    seconds_per_scene: int,
    model: str = "gpt-5-mini",
) -> List[Dict]:
    payload = {
        "chapter_title": str(chapter_title or "").strip(),
        "max_scenes": int(max_scenes),
        "seconds_per_scene": int(seconds_per_scene),
        "chapter_text": str(chapter_text or "").strip(),
        "output": "STRICT_JSON_LIST_ONLY",
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    text = ""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        text = resp.output_text
    else:
        text = str(resp)

    raw = (text or "").strip()
    json_str = _extract_json_list(raw)

    try:
        scenes = json.loads(json_str)
    except Exception as e:
        raise RuntimeError(
            f"Scene planner did not return valid JSON. {type(e).__name__}: {e}\n\nRAW:\n{raw[:1200]}"
        )

    if not isinstance(scenes, list):
        raise RuntimeError("Scene planner returned JSON but it is not a list.")

    out: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        if not isinstance(sc, dict):
            continue

        prompt = str(sc.get("prompt", "")).strip()
        if not prompt:
            continue

        sec = sc.get("seconds", seconds_per_scene)
        try:
            sec_i = int(sec)
        except Exception:
            sec_i = int(seconds_per_scene)
        sec_i = max(1, sec_i)

        out.append({"scene": i, "seconds": sec_i, "prompt": prompt})

    if not out:
        raise RuntimeError("Scene planner returned no usable prompts.")
    return out

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

# ============================
# PART 3/5 — ffmpeg Assembly (Concat + Mux) — NO SUBS
# ============================
# This is the only assembly you need for "segments only → CapCut does everything else".
# - concat scene MP4s into one silent segment video
# - mux narration audio onto that video
# - output ONE clean MP4 per segment

def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    """
    Concatenate MP4 clips using ffmpeg concat demuxer with stream copy.
    Assumes all MP4 clips were generated with consistent codec/settings.
    """
    if not mp4_paths:
        raise ValueError("concat_mp4s: no input mp4_paths")

    for p in mp4_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"concat_mp4s: missing input: {p}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # concat demuxer list file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        list_path = tf.name
        for p in mp4_paths:
            # escape single quotes for concat file format
            safe_p = p.replace("'", "'\\''")
            tf.write(f"file '{safe_p}'\n")

    try:
        # -fflags +genpts helps when clips come from image loops
        run_cmd([
            ff, "-y",
            "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """
    Mux narration audio onto the concatenated video.
    Video stream is copied; audio is encoded to AAC for broad compatibility.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"mux_audio: missing video_path: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"mux_audio: missing audio_path: {audio_path}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # If your narration is longer than video (or vice versa), -shortest trims to the shorter.
    run_cmd([
        ff, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path


def reencode_mp4(in_path: str, out_path: str) -> str:
    """
    Fallback utility ONLY if you ever hit concat-copy incompatibility.
    (Not normally used if your clips are produced consistently.)
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"reencode_mp4: missing input: {in_path}")

    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    run_cmd([
        ff, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-c:v", "libx264",
        "-preset", os.environ.get("UAPPRESS_X264_PRESET", "veryfast"),
        "-crf", os.environ.get("UAPPRESS_X264_CRF", "23"),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", os.environ.get("UAPPRESS_AAC_BITRATE", "192k"),
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path

# ============================
# PART 4/5 — Image → Ken Burns MP4 (ZOOM ONLY) + Tier-1 Caching + Disk Safety
# ============================

def _parse_size(size: str) -> Tuple[int, int]:
    if "x" not in (size or ""):
        return 1280, 720
    w, h = str(size).lower().split("x", 1)
    try:
        return int(w), int(h)
    except Exception:
        return 1280, 720


def _best_image_size_for_video(w: int, h: int) -> str:
    """
    gpt-image-1 supported sizes:
    - 1024x1024
    - 1536x1024 (landscape)
    - 1024x1536 (portrait)
    - auto
    """
    return "1536x1024" if w >= h else "1024x1536"


def _build_motion_vf(W: int, H: int, fps: int, frames: int) -> str:
    """
    ZOOM ONLY (no shake, no drift, no grain).
    Continuous zoom for full duration.
    """
    z0 = _safe_float_env("UAPPRESS_ZOOM_START", 1.00)
    z1 = _safe_float_env("UAPPRESS_ZOOM_END", 1.08)
    z0 = max(1.0, min(1.25, z0))
    z1 = max(z0, min(1.35, z1))

    # Smooth linear zoom over full frames using 'on' (output frame index)
    zoom_expr = f"{z0}+({z1}-{z0})*on/max(1\\,{frames-1})"

    # Dead-center crop tracking (no drift)
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    # IMPORTANT:
    # - We keep scale+crop BEFORE zoompan for consistent frame geometry
    # - No noise, no rotate, no extra filters
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={W}x{H},"
        f"fps={fps},"
        f"format=yuv420p"
    )

    return vf


def generate_video_clip(
    client: OpenAI,
    *,
    prompt: str,
    seconds: int,
    size: str,
    model: str,  # kept for compatibility with app.py (ignored here)
    out_path: str,
) -> str:
    """
    prompt -> OpenAI image -> ffmpeg zoom-only -> mp4
    - NO shake
    - NO grain
    - NO drift
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cache_dir = _get_cache_dir()
    _ensure_dir(cache_dir)
    _prune_cache(cache_dir)

    img_prompt = (
        (prompt or "").strip()
        + "\n\nStyle notes: cinematic documentary b-roll, realistic lighting, photorealistic, "
          "no text overlays, no logos, no watermarks."
    )

    seconds = max(1, int(seconds))
    W, H = _parse_size(size)
    img_size = _best_image_size_for_video(W, H)

    # FPS: your pipeline default is 15 unless you override UAPPRESS_KB_FPS
    fps = _safe_int_env("UAPPRESS_KB_FPS", 15)
    fps = max(10, min(30, fps))
    frames = max(1, int(seconds * fps))

    cache_id = _cache_key("gpt-image-1", img_prompt, str(seconds), str(size), f"fps={fps}")
    cached_png = os.path.join(cache_dir, f"{cache_id}.png")
    cached_mp4 = os.path.join(cache_dir, f"{cache_id}.mp4")

    if os.path.exists(cached_mp4):
        if not os.path.exists(out_path):
            _copy_file(cached_mp4, out_path)
        png_path = out_path.replace(".mp4", ".png")
        if os.path.exists(cached_png) and not os.path.exists(png_path):
            _copy_file(cached_png, png_path)
        return out_path

    png_path = out_path.replace(".mp4", ".png")
    if os.path.exists(cached_png) and not os.path.exists(png_path):
        _copy_file(cached_png, png_path)

    if not os.path.exists(png_path):
        img = None
        last_err = None
        for attempt in range(1, 4):
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size=img_size,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)

        if img is None:
            try:
                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="auto",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Image generation failed (gpt-image-1). {type(e).__name__}: {e}"
                ) from e

        try:
            b64 = img.data[0].b64_json
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            raise RuntimeError(f"Could not decode image bytes. {type(e).__name__}: {e}") from e

        with open(png_path, "wb") as f:
            f.write(img_bytes)
        with open(cached_png, "wb") as f:
            f.write(img_bytes)

    ff = ffmpeg_exe()
    vf = _build_motion_vf(W, H, fps=fps, frames=frames)

    preset = os.environ.get("UAPPRESS_X264_PRESET", "veryfast")
    crf = os.environ.get("UAPPRESS_X264_CRF", "23")

    run_cmd([
        ff, "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-loop", "1",
        "-i", png_path,
        "-t", str(seconds),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
        "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-g", str(max(30, fps * 2)),
        "-movflags", "+faststart",
        out_path,
    ])

    _copy_file(out_path, cached_mp4)
    _prune_cache(cache_dir)

    return out_path

# ============================
# PART 5/5 — Packaging Outputs (ZIP) + (Optional) Cleanup Helper
# ============================

def zip_dir(dir_path: str, zip_path: str) -> str:
    """
    Zip a directory to a file path (used by some legacy flows).
    """
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                z.write(full, rel)
    return zip_path


def safe_rmtree(path: str) -> None:
    """
    Best-effort cleanup (used by the new sequential generator after each upload).
    Never throws — cleanup should not crash the run.
    """
    try:
        if not path:
            return
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn))
                    except Exception:
                        pass
            try:
                os.rmdir(path)
            except Exception:
                pass
        elif os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        return
