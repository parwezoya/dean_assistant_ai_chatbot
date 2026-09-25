import pyttsx3
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pandas as pd
import random
import json
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
#  TTS ENGINE SETUP
# ─────────────────────────────────────────────
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 175)
 
# ─────────────────────────────────────────────
#  WHISPER MODEL  (loads once at startup)
#  Use "base" for speed; swap to "small" or
#  "medium" for better accuracy on noisy audio
# ─────────────────────────────────────────────
print("🔄 Loading Whisper model... (one-time startup)")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded.")
 
SAMPLE_RATE   = 16000   # Whisper expects 16 kHz
RECORD_SECONDS = 6      # seconds to record per turn
 
# ─────────────────────────────────────────────
#  CORE HELPERS
# ─────────────────────────────────────────────
def speak(audio: str):
    print(f"🧠 Dean: {audio}")
    engine.say(audio)
    engine.runAndWait()
 
 
def wishMe():
    hour = datetime.datetime.now().hour
    if hour < 12:
        speak("Good Morning!")
    elif hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Dean, your virtual assistant. How may I help you?")
 
 
def record_audio(seconds: int = RECORD_SECONDS) -> np.ndarray:
    """Record mono audio from the default microphone."""
    print("🎤 Listening...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()
 
 
def takeCommand(seconds: int = RECORD_SECONDS) -> str:
    """
    Record audio and transcribe with OpenAI Whisper.
    Returns the lowercase transcript or "none" on failure.
    """
    audio_np = record_audio(seconds)
 
    # Whisper needs a .wav file path  ─ use a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.write(tmp_path, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
 
    try:
        print("🧠 Transcribing with Whisper...")
        result = whisper_model.transcribe(tmp_path, language="en", fp16=False)
        query = result["text"].strip()
        print(f"👤 You said: {query}")
        return query.lower() if query else "none"
    except Exception as e:
        print(f"Whisper error: {e}")
        return "none"
    finally:
        os.remove(tmp_path)
 
 
def parse_email(spoken_email: str) -> str:
    spoken_email = spoken_email.lower()
    for spoken, symbol in [
        (" at ", "@"), (" dot ", "."),
        (" underscore ", "_"), (" dash ", "-"), (" space ", ""),
    ]:
        spoken_email = spoken_email.replace(spoken, symbol)
    return spoken_email.replace(" ", "")
 
 
def sendEmail(to: str, content: str) -> bool:
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('your_email@gmail.com', 'your_app_password')   # ← update
        server.sendmail('your_email@gmail.com', to, content)
        server.close()
        return True
    except Exception as e:
        print(e)
        return False
 
# ─────────────────────────────────────────────
#  CONTACTS
# ─────────────────────────────────────────────
CONTACTS_FILE = 'contacts.json'
try:
    with open(CONTACTS_FILE, 'r') as f:
        contacts = json.load(f)
except FileNotFoundError:
    contacts = {}
 
# ─────────────────────────────────────────────
#  EMAIL DATA
# ─────────────────────────────────────────────
CSV_PATH = "gmail_emails_classified.csv"
if os.path.exists(CSV_PATH):
    df_real = pd.read_csv(CSV_PATH)
    df_real['subject'] = df_real['subject'].fillna('')
    df_real['body']    = df_real['body'].fillna('')
    df_real['text']    = df_real['subject'] + " " + df_real['body']
else:
    df_real = pd.DataFrame(columns=['subject', 'body', 'text', 'Label', 'Predicted_Category'])
 
# ─────────────────────────────────────────────
#  MUSIC
# ─────────────────────────────────────────────
songs = [
    r"D:\\music\\song1.mp3",
    r"D:\\music\\song2.mp3",
]
 
# ─────────────────────────────────────────────
#  CALENDAR  (imported from calendar_manager)
# ─────────────────────────────────────────────
from calendar_manager import CalendarManager
calendar = CalendarManager()
 
# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    wishMe()
    speak("Voice spam filter is activated. Say spam to check emails, schedule a meeting to use the calendar, or exit to stop.")
 
    while True:
        query = takeCommand()
 
        # ── EXIT ──────────────────────────────
        if 'exit' in query or 'quit' in query:
            speak("Deactivating. Goodbye!")
            break
 
        # ── WIKIPEDIA ─────────────────────────
        elif 'wikipedia' in query:
            speak('Searching Wikipedia...')
            q = query.replace("wikipedia", "").strip()
            try:
                results = wikipedia.summary(q, sentences=2)
                speak("According to Wikipedia: " + results)
            except Exception:
                speak("Sorry, I couldn't find that on Wikipedia.")
 
        # ── YOUTUBE ───────────────────────────
        elif 'open youtube' in query:
            speak("What should I search on YouTube?")
            topic = takeCommand()
            webbrowser.open(f"https://www.youtube.com/results?search_query={topic}")
 
        # ── GOOGLE ────────────────────────────
        elif 'open google' in query:
            speak("What should I search on Google?")
            topic = takeCommand()
            webbrowser.open(f"https://www.google.com/search?q={topic}")
 
        # ── MUSIC ─────────────────────────────
        elif 'play music' in query:
            song = random.choice(songs)
            os.startfile(song)
 
        # ── TIME ──────────────────────────────
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")
 
        # ── VS CODE ───────────────────────────
        elif 'open code' in query:
            codePath = r"C:\Users\Elite BooK\AppData\Local\Programs\Microsoft VS Code\Code.exe"
            os.startfile(codePath)
 
        # ── SEND EMAIL ────────────────────────
        elif 'send email' in query:
            speak("Who should I send the email to?")
            recipient_name = takeCommand()
            if recipient_name in contacts:
                to = contacts[recipient_name]
            else:
                speak("Please tell me their email. Say it like: name at domain dot com")
                spoken_email = takeCommand()
                to = parse_email(spoken_email)
                contacts[recipient_name] = to
                with open(CONTACTS_FILE, 'w') as f:
                    json.dump(contacts, f)
 
            speak("What should I say?")
            content = takeCommand()
            speak(f"You said: {content}. Sending to {to}. Say 'send email' to confirm or 'cancel'.")
            confirmation = takeCommand()
            if 'send email' in confirmation:
                speak("Email sent!" if sendEmail(to, content) else "Sorry, email could not be sent.")
 
        # ── SPAM CHECK ────────────────────────
        elif 'spam' in query:
            if 'Label' in df_real.columns:
                spam_emails = df_real[df_real['Label'].str.lower() == 'spam']
                if spam_emails.empty:
                    speak("You are safe. No spam emails today.")
                else:
                    speak(f"You have {len(spam_emails)} spam emails.")
                    for _, row in spam_emails.iterrows():
                        speak(f"Spam alert: {row['subject']}")
            else:
                speak("Spam labels not available.")
 
        # ── EMAIL CATEGORIES ──────────────────
        elif any(cat in query for cat in ['urgent', 'meeting email', 'schedule email',
                                          'personal email', 'information email']):
            for cat in ['urgent', 'meeting', 'schedule', 'personal', 'information']:
                if cat in query:
                    if 'Predicted_Category' in df_real.columns:
                        filtered = df_real[df_real['Predicted_Category'] == cat]
                        if filtered.empty:
                            speak(f"You have no {cat} emails.")
                        else:
                            speak(f"You have {len(filtered)} {cat} emails.")
                            for _, row in filtered.iterrows():
                                speak(f"Subject: {row['subject']}")
                    else:
                        speak("Email categories not available. Please run the classifier first.")
                    break
 
        # ── CALENDAR ──────────────────────────
        elif any(kw in query for kw in ['schedule', 'meeting', 'calendar',
                                        'remind', 'cancel meeting', 'meetings today',
                                        'meetings this week', 'delete meeting']):
            calendar.handle_voice_command(query, speak, takeCommand)
 
        else:
            speak("I didn't quite catch that. Could you please repeat?")