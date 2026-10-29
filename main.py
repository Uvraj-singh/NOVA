import os
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="NOVA")

# Text model: "ollama" (local, default) or "groq" (cloud).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# Voice engine: "local" (pyttsx3 + faster-whisper, default) or "groq" (cloud).
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "local")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "troy")

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


def groq_post(path: str, **kwargs) -> requests.Response:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    r = requests.post(
        f"{GROQ_BASE}{path}",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=120,
        **kwargs,
    )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"Groq error {r.status_code}: {r.text[:200]}"
        )
    return r


def ask_ollama(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


def ask_groq(prompt: str) -> str:
    r = groq_post(
        "/chat/completions",
        json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]},
    )
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
    filename = f"{uuid.uuid4().hex}.wav"
    path = AUDIO_DIR / filename

    if VOICE_PROVIDER == "groq":
        r = groq_post(
            "/audio/speech",
            json={
                "model": GROQ_TTS_MODEL,
                "input": req.text,
                "voice": GROQ_TTS_VOICE,
                "response_format": "wav",
            },
        )
        path.write_bytes(r.content)
    else:
        import pyttsx3

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
    tmp = AUDIO_DIR / f"{uuid.uuid4().hex}_in.wav"
    tmp.write_bytes(audio.file.read())
    try:
        if VOICE_PROVIDER == "groq":
            with open(tmp, "rb") as f:
                r = groq_post(
                    "/audio/transcriptions",
                    files={"file": (tmp.name, f, "audio/wav")},
                    data={"model": GROQ_STT_MODEL, "response_format": "json"},
                )
            text = r.json()["text"]
        else:
            segments, _ = get_stt_model().transcribe(str(tmp))
            text = " ".join(seg.text.strip() for seg in segments)
    finally:
        tmp.unlink(missing_ok=True)
    return {"text": text}