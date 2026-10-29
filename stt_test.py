from faster_whisper import WhisperModel         

model = WhisperModel("base", device="cpu", compute_type="int8")  
segments, info = model.transcribe("speech.m4a")  

text = " ".join(seg.text.strip() for seg in segments) 
print("transcript:", text)                       