import librosa                                        
import soundfile as sf                                

y, sr = librosa.load("test.wav", sr=None)          
shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)  
sf.write("test_happy.wav", shifted, sr)                
print("saved test_happy.wav, sample rate", sr)         