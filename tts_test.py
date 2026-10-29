import pyttsx3                                        # 1: the text-to-speech engine

engine = pyttsx3.init()                               # 2: starts the voice engine
engine.save_to_file("Hello from NOVA", "test.wav")    # 3: writes speech to a file
engine.runAndWait()                                   # 4: actually does the writing
print("saved test.wav")   