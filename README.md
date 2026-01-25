# UAPpress Video Creator (Zip → MP4)

## What it does
Upload a ZIP from Documentary TTS Studio (chapter scripts + MP3s).
This app can:
- Create a scene plan per chapter (OpenAI text model)
- Generate short video clips per scene (OpenAI Sora Video API)
- Stitch clips + mux MP3 into chapter MP4s
- Optionally concatenate chapters into one final MP4

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
