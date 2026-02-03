# ============================
# video_pipeline.py — UAPpress
# Cinematic Shorts (Sora-driven, prompt-first) — v2
#
# Adds:
# - build_sora_prompt(): applies house style + toggles + presets
# - prepare_sora_short_job(): returns an execution-ready payload
#
# NOTE:
# The actual Sora execution call is intentionally not implemented here
# because Sora API endpoints and SDK surface may vary by environment.
# This keeps the app stable and lets you copy the final prompt into Sora today.
# ============================

from __future__ import annotations

from dataclasses import dataclass, asdict


def build_sora_prompt(
    *,
    user_prompt: str,
    style_mode: str,
    camera_rule: str,
    lighting_rule: str,
    grain_rule: str,
    target_seconds: int,
    aspect_note: str = "Native vertical 9:16",
) -> str:
    """Assemble a brand-consistent Sora prompt (prompt-first, no ambiguity)."""

    base_rules = [
        "Vertical video, 9:16 composition, designed for YouTube Shorts.",
        f"Duration: ~{int(target_seconds)} seconds.",
        f"Camera motion: {camera_rule}.",
        f"Lighting: {lighting_rule}.",
        f"Texture / realism: {grain_rule}.",
        "Serious investigative documentary tone. Credibility-first. No fantasy look.",
        "No on-screen text, no subtitles, no logos, no UI, no watermarks.",
        "No dialogue or lip-synced speaking. Ambient motion only.",
    ]

    if aspect_note and "safe" in aspect_note.lower():
        base_rules.append("Keep subject centered with safe margins for mobile UI overlays.")

    if style_mode.lower().startswith("archival"):
        style_header = [
            "ARCHIVAL REENACTMENT STYLE:",
            "Photoreal reenactment with period authenticity.",
            "Film stock look: subtle flicker, gate weave, dust, scratches, vignette as appropriate.",
            "Era-appropriate color science, slightly muted, documentary archival feel.",
        ]
    else:
        style_header = [
            "CINEMATIC REALISM STYLE:",
            "Modern documentary-grade photorealism.",
            "Natural color science, restrained contrast, cinematic but believable.",
            "Shallow depth-of-field only when appropriate, no glamor look.",
        ]

    # User prompt goes last so it remains the creative anchor.
    user_prompt = (user_prompt or "").strip()

    parts = []
    parts.extend(style_header)
    parts.append("")
    parts.append("HOUSE RULES:")
    parts.extend([f"- {r}" for r in base_rules])
    parts.append("")
    parts.append("SCENE DESCRIPTION:")
    parts.append(user_prompt if user_prompt else "(No user prompt provided.)")

    return "\n".join(parts).strip()


@dataclass
class SoraShortJob:
    # The key is included for runtime execution; you may omit it for exporting prompts.
    api_key: str
    prompt: str
    target_seconds: int
    resolution: str = "1080x1920"
    aspect_ratio: str = "9:16"
    narration_mode: str = "None"
    captions_mode: str = "None"
    # Future toggles
    style_mode: str | None = None
    camera_rule: str | None = None
    lighting_rule: str | None = None
    grain_rule: str | None = None


def prepare_sora_short_job(
    *,
    api_key: str,
    sora_prompt: str,
    target_seconds: int,
    resolution: str,
    narration_mode: str,
    captions_mode: str,
    style_mode: str | None = None,
    camera_rule: str | None = None,
    lighting_rule: str | None = None,
    grain_rule: str | None = None,
) -> dict:
    """Return an execution-ready payload for a Sora cinematic short."""
    job = SoraShortJob(
        api_key=str(api_key or ""),
        prompt=str(sora_prompt or ""),
        target_seconds=int(target_seconds),
        resolution=str(resolution or "1080x1920"),
        narration_mode=str(narration_mode or "None"),
        captions_mode=str(captions_mode or "None"),
        style_mode=style_mode,
        camera_rule=camera_rule,
        lighting_rule=lighting_rule,
        grain_rule=grain_rule,
    )
    return asdict(job)


def generate_sora_short(**kwargs):
    """
    Placeholder for executing a Sora job.

    When you’re ready, we’ll replace this with the real Sora API call:
    - Send prompt + duration + aspect_ratio
    - Receive MP4
    - Optionally mux narration + captions
    """
    raise NotImplementedError(
        "Sora execution is not implemented in this build. "
        "Use prepare_sora_short_job() + the Final Sora Prompt in the app."
    )
