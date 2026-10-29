import hashlib

import requests
import streamlit as st


def get_backend_url() -> str:
    # On Streamlit Cloud, the address comes from the app's Secrets.
    # On your laptop there's no secrets file, so use the local backend.
    try:
    except Exception:
        return "http://127.0.0.1:8000"


BACKEND_URL = get_backend_url()
CHAT_URL = f"{BACKEND_URL}/chat"

st.set_page_config(page_title="NOVA", page_icon="🤖")
st.title("NOVA")

emotion = st.selectbox(
    "Detected emotion (placeholder)",
    ["neutral", "happy", "sad", "angry", "surprised"],
)

personality = st.radio(
    "personality", ["serious", "sarcastic"], horizontal=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

mic = st.audio_input("Or speak to NOVA")
spoken_text = None
if mic is not None:
    key = hashlib.sha1(mic.getvalue()).hexdigest()
    if key != st.session_state.get("last_audio"):
        st.session_state.last_audio = key
        try:
            t = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"audio": ("speech.wav", mic.getvalue(), "audio/wav")},
                timeout=120,
            )
            t.raise_for_status()
            spoken_text = t.json()["text"]
        except requests.RequestException as e:
            st.caption(f"Transcription unavailable: {e}")

typed_text = st.chat_input("say something to NOVA")
user_text = typed_text or spoken_text
if user_text:
    st.session_state.messages.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    try:
        r = requests.post(
            CHAT_URL,
            json={
                "message": user_text,
                "personality": personality,
                "emotion": emotion,
                "history": st.session_state.messages[:-1],
            },
            timeout=120,
        )
        r.raise_for_status()
        reply = r.json()["reply"]
    except requests.RequestException as e:
        reply = f"could not reach the NOVA backend: {e}"

    st.session_state.messages.append({"role": "assistant", "text": reply})
    with st.chat_message("assistant"):
        st.write(reply)
        try:
            s = requests.post(
                f"{BACKEND_URL}/speak",
                json={"text": reply},
                timeout=120,
            )
            s.raise_for_status()
            audio_url = s.json()["audio_url"]
            st.audio(f"{BACKEND_URL}{audio_url}", format="audio/wav")
        except requests.RequestException as e:
            st.caption(f"voice unavailable: {e}")