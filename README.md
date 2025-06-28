
# 🗣️ SpeakEase – AI-Powered Speech Therapy Assistant

**SpeakEase** is an interactive AI-powered **speech therapy assistant** designed to support **children with stammering**. It uses real-time **speech transcription**, **emotion recognition**, and **motivational feedback** to help kids build fluency and confidence.

## 🚀 Features

- 🎤 **Fluency Practice**  
  Read aloud custom or sample passages and get instant feedback on how closely your speech matches the original text.

- 🤖 **Conversational Therapy with AI**  
  Practice speaking with a chatbot that gives corrected versions of your answers and asks smart follow-up questions using Google Gemini Pro.

- ✍️ **Real-time Transcription**  
  Convert recorded speech into text using OpenAI's Whisper model.

- 💡 **Progress Feedback**  
  AI-based scoring (TF-IDF cosine similarity) shows fluency level and gives encouraging tips.

- 📊 **Emotion & Sentiment-aware Response** *(planned)*  
  Emotion recognition and personalized feedback help create a positive speech environment.

---

## 🧠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)  
- **NLP & Scoring**: `scikit-learn`, TF-IDF + Cosine Similarity  
- **Conversational AI**: [Google Gemini Pro](https://ai.google.dev) (via Generative AI SDK)  
- **Audio Recording**: `sounddevice`, `numpy`  
- **Environment Management**: `python-dotenv`

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/aryanmarshian/speakease
   cd speakease

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**

   * Create a `.env` file in the root directory:

     ```
     GOOGLE_API_KEY=your_google_gemini_api_key
     ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

---

## 🖼️ Screenshots

<img src="https://myfirst.tech/wp-content/uploads/2023/02/PublicSpeaking1000.jpg" width="600"/>

---

## 📚 Example Use Case

* The child reads: “The quick brown fox jumps over the lazy dog.”
* The app transcribes their speech and evaluates fluency.
* Then, the chatbot responds with corrections and asks a new question like:
  *"Great try! What do you like doing after school?"*

---

## 📌 Future Improvements

* Emotion detection from audio
* Personalized weekly progress reports
* Interactive avatar or voice assistant

---

## 🤝 Contributing

Contributions are welcome! Please open issues or PRs to suggest features, improvements, or bug fixes.

---

## 📄 License

MIT License © 2024 Aryan Singh


