import os
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="NOVA")

# "ollama" runs locally (default). "groq" is used on the cloud host.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Voice needs local audio packages, so it is off on the cloud host.
ENABLE_VOICE = os.getenv("ENABLE_VOICE", "1") == "1"

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    personality: str = "serious"
    emotion: str = "neutral"
    history: list[dict] = []


class SpeakRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


def build_prompt(req: ChatRequest) -> str:
    style = {
        "serious": "calm and empathetic",
        "sarcastic": "witty, with light sarcasm but never rude",
    }.get(req.personality, "calm and empathetic")

    convo = "".join(
        f"{'user' if m['role'] == 'user' else 'NOVA'}: {m['text']}\n"
        for m in req.history[-6:]
    )

    return (
        f"You are NOVA, a companion with a {style} personality. "
        f"the user's face currently shows the emotion:{req.emotion}."
        "let that shape your tone. "
        "Reply in 2 to 4 sentences, using exactly this format:\n"
        "Emotion: <one word>\n"
        "Response: <your reply>\n\n"
        f"conversation so far:\n{convo}\n"
        f"User: {req.message}"
    )


def ask_ollama(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


def ask_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@app.post("/chat")
def chat(req: ChatRequest):
    prompt = build_prompt(req)
    if LLM_PROVIDER == "groq":
        reply = ask_groq(prompt)
    else:
        reply = ask_ollama(prompt)
    return {"reply": reply}


@app.post("/speak")
def speak(req: SpeakRequest):
    if not ENABLE_VOICE:
        raise HTTPException(status_code=503, detail="Voice is disabled on this server")
    import pyttsx3

    filename = f"{uuid.uuid4().hex}.wav"
    path = AUDIO_DIR / filename
    engine = pyttsx3.init()
    engine.save_to_file(req.text, str(path))
    engine.runAndWait()
    return {"audio_url": f"/audio/{filename}"}


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = AUDIO_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")


_stt_model = None


def get_stt_model():
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel

        _stt_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _stt_model


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    if not ENABLE_VOICE:
        raise HTTPException(status_code=503, detail="Voice is disabled on this server")
    tmp = AUDIO_DIR / f"{uuid.uuid4().hex}_in.wav"
    tmp.write_bytes(audio.file.read())
    segments, _ = get_stt_model().transcribe(str(tmp))
    text = " ".join(seg.text.strip() for seg in segments)
    tmp.unlink(missing_ok=True)
    return {"text": text}