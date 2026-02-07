# video_pipeline_GODMODE.txt
# Deterministic, fast, cost-aware video engine

from __future__ import annotations
import os, io, re, math, time, json, zipfile, tempfile, subprocess, shutil
from pathlib import Path
from typing import List, Dict, Tuple
from hashlib import sha1

try:
    from openai import OpenAI
except Exception:
    OpenAI=None

# ---------------- Utilities ----------------
SCRIPT_EXT={".txt",".md"}
AUDIO_EXT={".mp3",".wav",".m4a",".aac"}

def extract_zip_to_temp(data:bytes)->Tuple[str,str]:
    wd=tempfile.mkdtemp(prefix="god_vp_")
    ed=os.path.join(wd,"extracted"); os.makedirs(ed,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(ed)
    return wd,ed

def find_files(ed)->Tuple[List[str],List[str]]:
    s,a=[],[]
    for p in Path(ed).rglob("*"):
        if p.suffix.lower() in SCRIPT_EXT: s.append(str(p))
        if p.suffix.lower() in AUDIO_EXT: a.append(str(p))
    return sorted(s),sorted(a)

def safe_slug(t,max_len=40):
    s=re.sub(r"[^a-z0-9]+","-", (t or "").lower()).strip("-")
    return s[:max_len] or "segment"

def segment_label(pair:Dict)->str:
    n=(Path(pair.get("script_path","")).name+Path(pair.get("audio_path","")).name).lower()
    if "intro" in n: return "INTRO"
    if "outro" in n: return "OUTRO"
    return "SEGMENT"

def pair_segments(scripts:List[str], audios:List[str])->List[Dict]:
    amap={Path(a).stem.lower():a for a in audios}
    out=[]
    for s in scripts:
        stem=Path(s).stem.lower()
        a=amap.get(stem) or audios[len(out)%len(audios)]
        uid=sha1((s+a).encode()).hexdigest()[:10]
        out.append(dict(script_path=s,audio_path=a,uid=uid))
    return out

def resolution_wh(mode,res169,res916):
    if mode.startswith("Short"):
        return (720,1280) if res916=="720x1280" else (1080,1920)
    return (1280,720) if res169=="1280x720" else (1920,1080)

def default_scenes(is_vertical:bool)->int:
    return 6 if is_vertical else 12

# ---------------- Image Generation ----------------
VISUAL_STYLE = (
    "Photorealistic documentary still, natural light, realistic materials, "
    "no text, no logos, cinematic composition."
)

def image_prompts(script:str, n:int)->List[str]:
    paras=[p.strip() for p in re.split(r"\n\n+",script) if p.strip()]
    if not paras: return []
    step=max(1,len(paras)//n)
    picks=[paras[i] for i in range(0,len(paras),step)][:n]
    return [f"{VISUAL_STYLE} Scene inspired by: {p[:280]}" for p in picks]

def gen_image(api_key,prompt,size)->Path:
    if OpenAI is None: raise RuntimeError("OpenAI SDK missing")
    c=OpenAI(api_key=api_key)
    r=c.images.generate(model="gpt-image-1",prompt=prompt,size=size)
    import base64
    b=base64.b64decode(r.data[0].b64_json)
    fd,fp=tempfile.mkstemp(suffix=".png"); os.write(fd,b); os.close(fd)
    return Path(fp)

# ---------------- Video ----------------
def ffmpeg()->str:
    return shutil.which("ffmpeg") or "ffmpeg"

def audio_duration(p)->float:
    cmd=[ffmpeg(),"-i",p]
    pr=subprocess.run(cmd,stderr=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    m=re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)",pr.stderr or "")
    if not m: return 0.0
    h,mn,s=map(float,m.groups()); return h*3600+mn*60+s

def render_segment_mp4(*,pair,extract_dir,out_path,api_key,width,height,fps,max_scenes,**_):
    script=Path(pair["script_path"]).read_text(encoding="utf-8",errors="ignore")
    dur=max(1.0,audio_duration(pair["audio_path"]))
    scenes=image_prompts(script,max_scenes)
    size="1024x1536" if height>width else "1536x1024"
    imgs=[]
    for p in scenes:
        imgs.append(gen_image(api_key,p,size))
    per=max(0.5, dur/len(imgs))
    lst=Path(tempfile.mkstemp(suffix=".txt")[1])
    with open(lst,"w") as f:
        for i in imgs:
            f.write(f"file '{i.as_posix()}'\n")
            f.write(f"duration {per:.3f}\n")
    cmd=[ffmpeg(),"-y","-r",str(fps),"-f","concat","-safe","0","-i",str(lst),
         "-i",pair["audio_path"],"-c:v","libx264","-pix_fmt","yuv420p",
         "-shortest",out_path]
    subprocess.run(cmd,check=True)
