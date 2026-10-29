import requests 
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
import pyttsx3
from fastapi import File, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title ="NOVA")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

class ChatRequest(BaseModel):
    message: str
    personality: str = "serious"
    emotion: str = "neutral"
    history: list[dict] = []

@app.get ("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    style ={
        "serious": "calm and empathetic",
        "sarcastic": "witty, with light sarcasm but never rude"
    }.get(req.personality, "calm and empathetic")

    convo ="".join(
        f"{'user' if m['role'] == 'user' else 'NOVA'}: {m['text']}\n"
        for m in req.history[-6:]
    )

    prompt =(f"You are NOVA, a companion with a {style} personality. "
        f"the user's face currently shows the emotion:{req.emotion}."
        "let that shape your tone. "
        "Reply in 2 to 4 sentences, using exactly this format:\n"
        "Emotion: <one word>\n"
        "Response: <your reply>\n\n"
        f"conversation so far:\n{convo}\n"
        f"User: {req.message}"
    )

    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt":prompt,"stream":False},
        timeout=120,
    )
    r.raise_for_status()
    return {"reply": r.json()["response"]}

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
def speak(req: SpeakRequest):
    filename = f"{uuid.uuid4().hex}.wav"
    path = AUDIO_DIR / filename
    engine = pyttsx3.init()
    engine.save_to_file(req.text, str(path))
    engine.runAndWait()
    return {"audio_url":f"/audio/{filename}"}

@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = AUDIO_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")

stt_model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    tmp =AUDIO_DIR / f"{uuid.uuid4().hex}_in.wav"
    tmp.write_bytes(audio.file.read())
    segments, _=stt_model.transcribe(str(tmp))
    text =" ".join(seg.text.strip() for seg in segments)
    tmp.unlink(missing_ok=True)
    return{"text": text}