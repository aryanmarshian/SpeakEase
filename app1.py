import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

# Load Whisper model
model = whisper.load_model("base")

# Sample text for fluency practice
sample_text = "The quick brown fox jumps over the lazy dog."

# Load environment variables for Gemini API
from dotenv import load_dotenv
load_dotenv()  # loading all the environment variables

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(user_response, is_first_interaction=False):
    if is_first_interaction:
        follow_up_question = "Hi! I'm here to help you practice speaking. Can you please introduce yourself?"
    else:
        practice_instruction = "This is a conversation practice for a child Repond with a corrected version of the user reponse AND in the next line follow-up question based on the user's answer. "
        prompt = f"{practice_instruction}\n\nUser: {user_response}\nBot:"

        chat = genai.GenerativeModel("gemini-pro").start_chat(history=[])
        response = chat.send_message(prompt, stream=False)

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            follow_up_question = response.candidates[0].content.parts[0].text
        else:
            follow_up_question = "I'm sorry, I didn't catch that. Could you please repeat?"

    return follow_up_question


def record_audio(duration, fs):
    st.info("Recording... 🎙️ Please speak:")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  
    st.success("Recording complete! 🌟")
    return np.squeeze(audio)

# Function to calculate similarity score
def calculate_score(reference_text, transcribed_text):
    vectorizer = TfidfVectorizer().fit_transform([reference_text, transcribed_text])
    vectors = vectorizer.toarray()
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(score * 100, 2)  # Convert to percentage


# Generic responses for the chatbot
def generate_generic_response():
    responses = [
        "That's wonderful! Thanks for sharing!",
        "Oh, that sounds really interesting!",
        "Wow, I hadn't thought of it that way!",
        "Nice! I can see why you enjoy that.",
        "That’s amazing! Tell me more!"
    ]
    return random.choice(responses)

st.title("Welcome to SpeakEase 🎙️")
st.markdown("<p style='font-size: 20px;'>SpeakEase helps kids practice speaking with fun and interactive activities. Improve your confidence and fluency by speaking aloud. Let's get started!</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'><img src='https://myfirst.tech/wp-content/uploads/2023/02/PublicSpeaking1000.jpg' width='700'></div>", unsafe_allow_html=True)

st.markdown("<h2 style=''>Fluency Practice 🗣️</h2>", unsafe_allow_html=True)
passage = st.text_area("Enter a passage to practice:", sample_text, height=100)
if st.button("Start Fluency Recording"):
    duration = 8  # Recording duration for reading
    fs = 16000  # Sample rate
    audio = record_audio(duration, fs)
    
    # Transcribe using Whisper
    with st.spinner("Transcribing... 🎤"):
        result = model.transcribe(audio, fp16=False)
        transcribed_text = result['text']

    st.subheader("Your Speech:")
    st.write(transcribed_text)

    # Calculate and display similarity score
    score = calculate_score(passage, transcribed_text)
    st.write(f"**Fluency Score:** {score}%")
    if score > 80:
        st.success("Great job! Your fluency is excellent!")
    elif score > 50:
        st.info("Good job! Keep practicing to improve even more!")
    else:
        st.warning("Don't worry, keep practicing to build confidence!")

# Conversation Practice Section with multiple questions displayed
st.markdown("<h2 style=''>Conversation Practice 💬</h2>", unsafe_allow_html=True)
st.write("Answer the questions below by speaking. Let's chat!")

# Initialize session state to keep track of answered questions
if "answered_questions" not in st.session_state:
    st.session_state.answered_questions = {}

# Display the first question (Introduce Yourself)
if "first_question" not in st.session_state:
    st.session_state.first_question = True

if st.session_state.first_question:
    question = "Please introduce yourself."
    st.markdown(f"<p style='font-size: 18px; color: #3399FF;'>{question}</p>", unsafe_allow_html=True)
    st.session_state.first_question = False

    # Record the user's response
    if st.button("Record Answer for Introduction"):
        duration = 10  # Recording duration for answers
        fs = 16000
        audio = record_audio(duration, fs)
        
        # Transcribe using Whisper
        with st.spinner("Transcribing your answer..."):
            result = model.transcribe(audio, fp16=False)
        
        user_response = result['text']
        st.write("Your Answer:")
        st.markdown(f"<p style='color: #33CC99;'>{user_response}</p>", unsafe_allow_html=True)

        # Get a follow-up question from Gemini
        bot_response = get_gemini_response(user_response)
        st.write("Chatbot says:")
        st.markdown(f"<p style='color: #3399FF;'>{bot_response}</p>", unsafe_allow_html=True)

        # Save the answer in session state
        st.session_state.answered_questions[question] = user_response

# Continuous conversation loop (cycle through the questions and responses)
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = True

if st.session_state.conversation_active:
    # Get the last user response and generate a new question
    # Check if there are any answered questions before accessing the last one
    if st.session_state.answered_questions:
        last_user_response = list(st.session_state.answered_questions.values())[-1]
    else:
        last_user_response = None  # Or set a default value if no questions have been answered

    follow_up_question = get_gemini_response(last_user_response)

    # Display the follow-up question
    st.write(f"Next question: {follow_up_question}")

    # Allow the user to record another answer
    if st.button(f"Record Answer for Question {len(st.session_state.answered_questions) + 1}"):
        duration = 10  # Recording duration for answers
        fs = 16000
        audio = record_audio(duration, fs)
        
        # Transcribe the new answer
        with st.spinner("Transcribing your answer..."):
            result = model.transcribe(audio, fp16=False)
        
        user_response = result['text']
        st.write("Your Answer:")
        st.markdown(f"<p style='color: #33CC99;'>{user_response}</p>", unsafe_allow_html=True)

        # Get the new follow-up question from Gemini
        bot_response = get_gemini_response(user_response)
        st.write("Chatbot says:")
        st.markdown(f"<p style='color: #3399FF;'>{bot_response}</p>", unsafe_allow_html=True)

        # Save the answer in session state
        st.session_state.answered_questions[follow_up_question] = user_response + "1"

    # Option to stop the conversation
    if st.button("End Conversation"):
        st.session_state.conversation_active = False
        st.success("Conversation ended! Thank you for practicing.")
