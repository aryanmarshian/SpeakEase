import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading

# Load Whisper model
model = whisper.load_model("base")

# Queue for live audio
audio_queue = queue.Queue()

# Function to process live audio and transcribe it
def transcribe_live_audio(fs=16000, chunk_duration=3):
    audio_data = []
    chunk_samples = chunk_duration * fs

    while True:
        data = audio_queue.get()
        if data is None:
            break

        audio_data.extend(data)

        if len(audio_data) >= chunk_samples:
            # Process last chunk for transcription
            audio_chunk = np.array(audio_data[-chunk_samples:])
            audio_chunk = np.squeeze(audio_chunk)
            
            # Transcribe current chunk
            result = model.transcribe(audio_chunk, fp16=False)
            st.session_state["live_transcription"] += result['text'] + " "
            st.experimental_rerun()  # Update live transcription

# Function to capture audio from the microphone in real time
def capture_audio_in_real_time(duration=30, fs=16000):
    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())  # Add audio data to queue

    # Initialize live transcription
    st.session_state["live_transcription"] = ""
    audio_stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)

    with audio_stream:
        st.info("Recording... Speak now!")
        sd.sleep(duration * 1000)  # Capture audio for 30 seconds

    # Stop transcription thread after recording ends
    audio_queue.put(None)

# Streamlit UI
st.title("Live Speech-to-Text Transcription")

# Start live transcription on button click
if "live_transcription" not in st.session_state:
    st.session_state["live_transcription"] = ""

if st.button("Start Recording"):
    # Start transcription thread
    threading.Thread(target=transcribe_live_audio, daemon=True).start()

    # Start capturing audio
    capture_audio_in_real_time(duration=30)

# Show live transcription in real time
st.subheader("Live Transcription:")
st.write(st.session_state["live_transcription"])
