import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import random
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Load Whisper model
model = whisper.load_model("base")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize conversation session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to get Gemini response
def get_gemini_response(user_response, is_first_interaction=False):
    practice_instruction = (
        "Respond with a corrected version of the user's answer, followed by a follow-up question. Keep responses simple for a child audience."
    )
    prompt = f"{practice_instruction}\n\nUser: {user_response}\nBot:"
    
    chat = genai.GenerativeModel("gemini-pro").start_chat(history=[])
    response = chat.send_message(prompt, stream=False)

    if response.candidates and response.candidates[0].content:
        return response.candidates[0].content.parts[0].text
    return "I'm sorry, I didn't catch that. Could you please repeat?"

# Helper function to record and transcribe audio
def record_and_transcribe(duration=8, fs=16000):
    st.info("Recording... 🎙️")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete! 🌟")
    result = model.transcribe(np.squeeze(audio), fp16=False)
    return result['text']

# Function to calculate fluency score
def calculate_score(reference_text, transcribed_text):
    vectorizer = TfidfVectorizer().fit_transform([reference_text, transcribed_text])
    vectors = vectorizer.toarray()
    return round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)

# UI
st.title("Welcome to SpeakEase 🎙️")
st.markdown("Practice speaking with interactive exercises designed to build confidence and fluency.")

# Conversation Practice Section
st.subheader("Conversation Practice 💬")
st.write("Speak your answers, and I'll help you with follow-up questions!")

# Start the conversation
if st.button("Start Conversation" if not st.session_state.conversation_history else "Continue Conversation"):
    # Check if it's the first interaction
    if not st.session_state.conversation_history:
        question = "Hi! Can you introduce yourself?"
    else:
        last_response = st.session_state.conversation_history[-1]['user']
        question = get_gemini_response(last_response)

    # Record the user's answer
    user_response = record_and_transcribe(duration=10)
    st.write("Your Answer:")
    st.markdown(f"<p style='color: #33CC99;'>{user_response}</p>", unsafe_allow_html=True)

    # Append conversation turn to history
    st.session_state.conversation_history.append({"question": question, "user": user_response})

    # Get chatbot response
    bot_response = get_gemini_response(user_response)
    st.session_state.conversation_history[-1]['bot'] = bot_response

    st.write("Chatbot says:")
    st.markdown(f"<p style='color: #3399FF;'>{bot_response}</p>", unsafe_allow_html=True)

# Display conversation history for context
if st.session_state.conversation_history:
    st.write("### Conversation History")
    for turn in st.session_state.conversation_history:
        st.write(f"**Q: {turn['question']}**")
        st.write(f"**A: {turn['user']}**")
        st.write(f"**Chatbot**: {turn['bot']}")

# End the conversation
if st.button("End Conversation"):
    st.session_state.conversation_history.clear()
    st.success("Conversation ended. Thank you for practicing!")
