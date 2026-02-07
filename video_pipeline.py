# ============================
# video_pipeline.py (GODMODE compatible)
# ============================
# Designed to work with your current app.py (Streamlit Cloud) and its call signature:
#   render_segment_mp4(..., burn_subtitles=bool, subtitle_style=str, export_srt=bool)
#
# Guarantees:
# - Explicit stream mapping + audio decode preflight (no silent/no-audio MP4s).
# - 9:16 ALWAYS fills frame (scale+crop). Never pads.
# - Subtitle burn-in optional (default controlled by app checkbox).
# - Deterministic pairing (stem → numeric prefix → stable order).

from __future__ import annotations

import io
import os
import re
import time
import shutil
import zipfile
import base64
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

_SCRIPT_EXTS = {'.txt', '.md'}
_SUB_EXTS = {'.srt', '.vtt', '.ass'}
_AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}

_DEFAULT_IMAGE_MODEL = os.environ.get('UAPPRESS_IMAGE_MODEL', 'gpt-image-1')
_DEFAULT_IMAGE_SIZE_169 = os.environ.get('UAPPRESS_IMAGE_SIZE_169', '1536x1024')
_DEFAULT_IMAGE_SIZE_916 = os.environ.get('UAPPRESS_IMAGE_SIZE_916', '1024x1536')
_IMAGE_CACHE_DIRNAME = os.environ.get('UAPPRESS_IMAGE_CACHE_DIRNAME', '_images_cache_godmode')

def build_sora_prompt(scene_text: str, style: str = 'photorealistic') -> str:
    scene_text = (scene_text or '').strip()
    scene_text = re.sub(r'\s+', ' ', scene_text).strip()
    if style.lower().startswith('photo'):
        return (
            'Photorealistic documentary still. Natural light, realistic materials, cinematic composition. '
            'No text, no captions, no logos, no watermarks.\n'
            f'Scene: {scene_text}'
        )
    if style.lower().startswith('cinematic'):
        return (
            'Cinematic stylized realism, grounded documentary look, subtle film grain. '
            'No text, no captions, no logos, no watermarks.\n'
            f'Scene: {scene_text}'
        )
    return f'Scene: {scene_text}'

def safe_slug(text: str, max_len: int = 60) -> str:
    s = (text or '').strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    if not s:
        s = 'untitled'
    return (s[:max_len].strip('-') or 'untitled')

def _stable_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1((s or '').encode('utf-8', errors='ignore')).hexdigest()[: max(6, int(n))]

def ffmpeg_exe() -> str:
    return shutil.which('ffmpeg') or 'ffmpeg'

def _ffprobe_exe() -> Optional[str]:
    return shutil.which('ffprobe') or None

def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or '').strip()[-4000:]
        raise RuntimeError(f'Command failed (code {proc.returncode}). Tail:\n{tail}')

def _run_cmd_rc(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout or '', proc.stderr or ''

def _parse_duration_from_ffmpeg_stderr(stderr_text: str) -> float:
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', stderr_text or '')
    if not m:
        return 0.0
    hh = float(m.group(1)); mm = float(m.group(2)); ss = float(m.group(3))
    return hh*3600.0 + mm*60.0 + ss

def ffprobe_duration_seconds(path: Union[str, Path]) -> float:
    p = str(path)
    fp = _ffprobe_exe()
    if fp:
        rc, out, err = _run_cmd_rc([fp, '-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1', p])
        if rc == 0:
            try:
                return float((out or '').strip())
            except Exception:
                pass
    ff = ffmpeg_exe()
    rc, out, err = _run_cmd_rc([ff, '-hide_banner', '-i', p])
    return _parse_duration_from_ffmpeg_stderr(err or '')

def has_audio_stream(path: Union[str, Path]) -> bool:
    p = str(path)
    fp = _ffprobe_exe()
    if fp:
        rc, out, err = _run_cmd_rc([fp,'-v','error','-select_streams','a','-show_entries','stream=index','-of','csv=p=0', p])
        if rc == 0:
            return bool((out or '').strip())
    ff = ffmpeg_exe()
    rc, out, err = _run_cmd_rc([ff, '-hide_banner', '-i', p])
    return bool(re.search(r'Audio:\s', err or ''))

def extract_zip_to_temp(zip_bytes_or_path: Union[bytes, str, Path]) -> Tuple[str, str]:
    workdir = tempfile.mkdtemp(prefix='uappress_zip_')
    extract_dir = str(Path(workdir) / 'extracted')
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(zip_bytes_or_path, (str, Path)):
        data = Path(zip_bytes_or_path).read_bytes()
    else:
        data = bytes(zip_bytes_or_path)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)
    return workdir, extract_dir

def find_files(extract_dir: Union[str, Path]) -> Tuple[List[str], List[str], List[str]]:
    ed = Path(extract_dir)
    scripts: List[str] = []; audios: List[str] = []; subs: List[str] = []
    for p in ed.rglob('*'):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _SCRIPT_EXTS:
            scripts.append(str(p))
        elif ext in _AUDIO_EXTS:
            audios.append(str(p))
        elif ext in _SUB_EXTS:
            subs.append(str(p))
    scripts.sort(key=lambda x: Path(x).name.lower())
    audios.sort(key=lambda x: Path(x).name.lower())
    subs.sort(key=lambda x: Path(x).name.lower())
    return scripts, audios, subs

def _leading_int(name: str) -> Optional[int]:
    m = re.match(r'^\s*0*(\d+)', (name or ''))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _guess_kind_from_name(name: str) -> str:
    n = (name or '').lower()
    if 'intro' in n: return 'INTRO'
    if 'outro' in n: return 'OUTRO'
    m = re.search(r'(chapter|ch)[\s_\-]*0*(\d+)', n)
    if m: return f'CHAPTER {int(m.group(2))}'
    return 'SEGMENT'

def segment_label(pair: Dict[str, Any]) -> str:
    kind = str(pair.get('kind_guess') or '').strip().upper()
    if kind.startswith('CHAPTER'): return kind
    if kind in ('INTRO','OUTRO'): return kind
    combo = (str(pair.get('script_path') or '') + ' ' + str(pair.get('audio_path') or '')).lower()
    if 'intro' in combo: return 'INTRO'
    if 'outro' in combo: return 'OUTRO'
    return 'SEGMENT'

def pair_segments(scripts: List[str], audios: List[str], subs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    subs = subs or []
    audios = list(audios or [])
    scripts = list(scripts or [])
    used_audio: set[str] = set()
    audio_by_stem = {Path(a).stem.lower(): a for a in audios}
    sub_by_stem = {Path(s).stem.lower(): s for s in subs}
    audio_by_num: Dict[int, str] = {}
    for a in audios:
        n = _leading_int(Path(a).stem)
        if n is not None and n not in audio_by_num:
            audio_by_num[n] = a
    sub_by_num: Dict[int, str] = {}
    for s in subs:
        n = _leading_int(Path(s).stem)
        if n is not None and n not in sub_by_num:
            sub_by_num[n] = s
    pairs: List[Dict[str, Any]] = []
    for s in scripts:
        sp = Path(s); stem = sp.stem.lower()
        a_match = audio_by_stem.get(stem, '')
        if (not a_match) or (a_match in used_audio):
            n = _leading_int(sp.stem)
            if n is not None and n in audio_by_num and audio_by_num[n] not in used_audio:
                a_match = audio_by_num[n]
        if (not a_match) or (a_match in used_audio):
            for a in audios:
                if a not in used_audio:
                    a_match = a; break
        if a_match:
            used_audio.add(a_match)
        sub_match = sub_by_stem.get(stem, '')
        if not sub_match:
            n = _leading_int(sp.stem)
            if n is not None:
                sub_match = sub_by_num.get(n, '')
        kind = _guess_kind_from_name(sp.name)
        title_guess = ''
        try:
            preview = sp.read_text(encoding='utf-8', errors='ignore').strip()
            for line in preview.splitlines():
                raw = line.strip()
                if not raw: continue
                if raw.startswith('#'):
                    title_guess = raw.lstrip('#').strip()[:120]; break
                if raw.lower().startswith('title:'):
                    title_guess = raw.split(':',1)[1].strip()[:120]; break
        except Exception:
            pass
        seed = f"script:{str(sp.resolve())}|audio:{str(Path(a_match).resolve()) if a_match else ''}"
        uid = _stable_hash(seed, 14)
        pairs.append({'script_path': str(sp), 'audio_path': str(a_match), 'subtitle_path': str(sub_match) if sub_match else '', 'kind_guess': kind, 'title_guess': title_guess, 'uid': uid})
    return pairs

_SHOTS = [
    'Wide establishing frame, eye-level, stable tripod feel.',
    'Medium documentary framing, natural perspective.',
    'Close detail shot, shallow depth of field.',
    'Wide with foreground framing (fence/window/doorway).',
    'Medium from behind (observer POV), faces not visible.',
    'Close-up of objects (papers, radios, maps), practical lighting.',
]

def _sanitize_scene(text: str, max_chars: int = 320) -> str:
    s = (text or '').strip()
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + '…'
    return s

def _choose_beats(script_text: str, n: int) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n\s*\n+', script_text or '') if p.strip()]
    if not paras:
        paras = [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+', (script_text or '').strip()) if p.strip()]
    if not paras: return []
    n = max(1, int(n))
    if len(paras) <= n: return paras
    step = len(paras) / float(n)
    idxs = [min(len(paras)-1, int(i*step)) for i in range(n)]
    out: List[str] = []; last = -1
    for i in idxs:
        if i == last: i = min(len(paras)-1, i+1)
        out.append(paras[i]); last = i
    return out[:n]

def _image_size_for_wh(width: int, height: int) -> str:
    return _DEFAULT_IMAGE_SIZE_916 if height > width else _DEFAULT_IMAGE_SIZE_169

def _build_prompts(script_text: str, max_scenes: int) -> List[str]:
    beats = _choose_beats(script_text, max_scenes)
    prompts: List[str] = []
    for i, b in enumerate(beats):
        snippet = _sanitize_scene(b, 300)
        cam = _SHOTS[i % len(_SHOTS)]
        prompts.append(build_sora_prompt(f"{cam} {snippet}", style='photorealistic'))
    return prompts

def _segment_cache_dir(extract_dir: str, uid: str) -> Path:
    root = Path(extract_dir) / _IMAGE_CACHE_DIRNAME / uid
    root.mkdir(parents=True, exist_ok=True)
    return root

def _is_good_image(path: Path, min_bytes: int = 50_000) -> bool:
    try:
        return path.exists() and path.stat().st_size >= int(min_bytes)
    except Exception:
        return False

def _openai_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Add 'openai' to requirements.txt.")
    api_key = (api_key or '').strip()
    if not api_key:
        raise RuntimeError('OpenAI API key is empty.')
    client = OpenAI(api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, 3):
        try:
            r = client.images.generate(model=_DEFAULT_IMAGE_MODEL, prompt=prompt, size=size)
            b64 = r.data[0].b64_json
            if not b64:
                raise RuntimeError('Images API returned empty b64_json.')
            return base64.b64decode(b64)
        except Exception as e:
            last_err = e
            time.sleep(0.25 * attempt)
    raise RuntimeError(f'Images API failed: {last_err}')

def _ensure_images(*, extract_dir: str, uid: str, api_key: str, prompts: List[str], size: str) -> List[Path]:
    out: List[Path] = []
    cache = _segment_cache_dir(extract_dir, uid)
    for i, prompt in enumerate(prompts, start=1):
        img_path = cache / f'img_{i:02d}.png'
        if _is_good_image(img_path):
            out.append(img_path); continue
        data = _openai_image_bytes(api_key, prompt, size)
        img_path.write_bytes(data)
        out.append(img_path)
    return out

def _sanitize_caption_text(s: str) -> str:
    s = (s or '')
    s = s.replace('\\\\', ' ')
    s = s.replace('\\', ' ')
    s = s.replace('“','"').replace('”','"').replace('’',"'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _format_srt_time(t: float) -> str:
    t = max(0.0, float(t))
    ms = int(round((t - int(t)) * 1000.0))
    s = int(t) % 60; m = (int(t)//60)%60; h = int(t)//3600
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def _build_srt_from_text(text: str, duration: float) -> str:
    raw = _sanitize_caption_text(text)
    parts = [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+', raw) if p.strip()]
    if not parts:
        parts = [raw] if raw else []
    if not parts:
        return ''
    max_cues = min(140, max(12, int(duration / 1.2)))
    if len(parts) > max_cues:
        step = len(parts) / float(max_cues)
        parts = [parts[min(len(parts)-1, int(i*step))] for i in range(max_cues)]
    cue_dur = max(0.8, duration / float(len(parts)))
    cues: List[str] = []
    t = 0.0
    for i, p in enumerate(parts, start=1):
        start = t; end = min(duration, t + cue_dur); t = end
        cues.append(str(i))
        cues.append(f'{_format_srt_time(start)} --> {_format_srt_time(end)}')
        cues.append(p)
        cues.append('')
        if t >= duration: break
    return '\n'.join(cues).strip() + '\n'

def _write_subs_for_segment(extract_dir: str, uid: str, audio_path: Path, script_text: str, subtitle_path: str) -> Path:
    cache = _segment_cache_dir(extract_dir, uid)
    out_srt = cache / 'captions.srt'
    if subtitle_path:
        sp = Path(subtitle_path)
        if sp.exists():
            txt = sp.read_text(encoding='utf-8', errors='ignore')
            txt = _sanitize_caption_text(txt)
            out_srt.write_text(txt, encoding='utf-8')
            return out_srt
    dur = float(ffprobe_duration_seconds(audio_path))
    out_srt.write_text(_build_srt_from_text(script_text, dur), encoding='utf-8')
    return out_srt

def _escape_filter_path(p: Union[str, Path]) -> str:
    s = str(p)
    s = s.replace('\\\\', '\\\\\\\\')
    s = s.replace(':', '\\:')
    s = s.replace("'", "\\'")
    return s

def _concat_list_file(images: List[Path], seconds_per: float) -> Path:
    seconds_per = max(0.5, float(seconds_per))
    p = Path(tempfile.mkstemp(prefix='uappress_concat_', suffix='.txt')[1])
    lines: List[str] = []
    for img in images:
        lines.append(f"file '{img.as_posix()}'")
        lines.append(f'duration {seconds_per:.3f}')
    lines.append(f"file '{images[-1].as_posix()}'")
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return p

def _vf_fill_frame(width: int, height: int) -> str:
    return f'scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p'

def _assert_audio_decodes(audio_path: Path) -> None:
    ff = ffmpeg_exe()
    rc, out, err = _run_cmd_rc([ff, '-hide_banner', '-v','error', '-i', str(audio_path), '-f','null','-'])
    if rc != 0:
        tail = (err or '').strip()[-2000:]
        raise RuntimeError(f'Audio decode failed for {audio_path.name}. Tail:\n{tail}')
    dur = float(ffprobe_duration_seconds(audio_path))
    if dur < 1.0:
        raise RuntimeError(f'Audio too short (<1.0s) or invalid: {audio_path.name} dur={dur:.2f}s')

def render_segment_mp4(*, pair: Dict[str, Any], extract_dir: str, out_path: str, api_key: str, fps: int = 30, width: int = 1280, height: int = 720, zoom_strength: float = 0.0, max_scenes: int = 12, min_scene_seconds: int = 6, max_scene_seconds: int = 120, burn_subtitles: bool = True, subtitle_style: str = 'Auto', export_srt: bool = False) -> None:
    sp = Path(pair.get('script_path') or '')
    ap = Path(pair.get('audio_path') or '')
    subp = str(pair.get('subtitle_path') or '').strip()
    uid = str(pair.get('uid') or _stable_hash(str(sp) + '|' + str(ap), 14))
    if not sp.exists(): raise RuntimeError(f'Missing script file: {sp}')
    if not ap.exists(): raise RuntimeError(f'Missing audio file: {ap}')
    script_text = sp.read_text(encoding='utf-8', errors='ignore').strip()
    if not script_text: raise RuntimeError(f'Empty script file: {sp.name}')
    _assert_audio_decodes(ap)
    dur = float(ffprobe_duration_seconds(ap))
    max_scenes = max(1, min(60, int(max_scenes or 12)))
    prompts = _build_prompts(script_text, max_scenes)
    if not prompts: raise RuntimeError('Unable to derive prompts from script.')
    size = _image_size_for_wh(int(width), int(height))
    imgs = _ensure_images(extract_dir=extract_dir, uid=uid, api_key=str(api_key), prompts=prompts, size=size)
    sec_per = dur / float(len(imgs))
    sec_per = max(float(min_scene_seconds), min(float(max_scene_seconds), sec_per))
    lst = _concat_list_file(imgs, sec_per)
    outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
    vf = _vf_fill_frame(int(width), int(height))
    preset = 'veryfast'
    is_vertical = int(height) > int(width)
    crf = '20' if is_vertical else '19'
    tmp_base = outp.with_suffix('.base.mp4')
    cmd = [ffmpeg_exe(), '-hide_banner', '-loglevel','error', '-f','concat','-safe','0','-i', str(lst), '-i', str(ap), '-map','0:v:0','-map','1:a:0', '-vf', vf, '-r', str(int(fps)), '-c:v','libx264','-preset', preset, '-tune','stillimage','-crf', crf, '-pix_fmt','yuv420p', '-c:a','aac','-b:a','192k', '-shortest','-movflags','+faststart', str(tmp_base)]
    run_cmd(cmd)
    if (not tmp_base.exists()) or tmp_base.stat().st_size < 80000: raise RuntimeError('Base render produced invalid/too-small MP4.')
    if not has_audio_stream(str(tmp_base)): raise RuntimeError('Base render produced MP4 with NO AUDIO STREAM.')
    if burn_subtitles:
        srt_path = _write_subs_for_segment(extract_dir, uid, ap, script_text, subp)
        if export_srt:
            try: Path(str(outp) + '.srt').write_text(srt_path.read_text(encoding='utf-8', errors='ignore'), encoding='utf-8')
            except Exception: pass
        if str(subtitle_style or 'Auto').lower() == 'none':
            tmp_base.replace(outp)
        else:
            font_size = 46 if is_vertical and int(height) >= 1920 else (38 if is_vertical else 28)
            margin_v = 110 if is_vertical else 60
            style = f'Alignment=2,MarginV={margin_v},Fontsize={font_size},Outline=2,Shadow=1'
            filt = f"subtitles='{_escape_filter_path(srt_path)}':force_style='{style}'"
            cmd2 = [ffmpeg_exe(), '-hide_banner','-loglevel','error','-i', str(tmp_base), '-vf', filt, '-c:v','libx264','-preset', preset, '-crf', crf, '-pix_fmt','yuv420p', '-c:a','copy', '-movflags','+faststart', str(outp)]
            run_cmd(cmd2)
            try: tmp_base.unlink(missing_ok=True)
            except Exception: pass
    else:
        tmp_base.replace(outp)
    if (not outp.exists()) or outp.stat().st_size < 80000: raise RuntimeError('Final render produced invalid/too-small MP4.')
    if float(ffprobe_duration_seconds(outp)) <= 0.05: raise RuntimeError('Final render produced zero-duration MP4.')
    if not has_audio_stream(str(outp)): raise RuntimeError('Final render has no audio stream.')
