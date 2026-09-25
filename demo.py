import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pandas as pd
import random
import time
import json
import requests
import pickle
LIBRARY_JSON = "document_library.json"
 
def load_library():
    """Return the full document library dict, or {} if none exists."""
    if not os.path.exists(LIBRARY_JSON):
        return {}
    try:
        with open(LIBRARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
 
def pick_document(speak_fn, listen_fn):
    """
    List available documents, ask user to pick one by saying the title
    or its number. Returns the chosen summary_data dict, or None.
    """
    library = load_library()
    if not library:
        speak_fn("You have no documents in your library yet. "
                 "Please run the document analyser on a file first.")
        return None
 
    titles = list(library.keys())
 
    # Read out available titles
    speak_fn(f"You have {len(titles)} document{'s' if len(titles) > 1 else ''} in your library.")
    for i, title in enumerate(titles, 1):
        speak_fn(f"{i}. {title}")
 
    speak_fn("Which one would you like? Say the title or its number.")
    response = listen_fn()
 
    if response == "none":
        speak_fn("I did not catch that. Please try again.")
        return None
 
    # Match by number
    for i, title in enumerate(titles, 1):
        if str(i) in response or f"number {i}" in response or f"{i}st" in response or f"{i}nd" in response or f"{i}rd" in response or f"{i}th" in response:
            speak_fn(f"Got it. Loading {title}.")
            return library[title]
 
    # Match by title keyword
    response_lower = response.lower()
    for title in titles:
        if any(word in response_lower for word in title.lower().split() if len(word) > 3):
            speak_fn(f"Got it. Loading {title}.")
            return library[title]
 
    speak_fn("Sorry, I could not match that to any document in your library.")
    return None
 
def speak_doc(sd, speak_fn):
    """Read out a document summary dict."""
    speak_fn(f"This is a {sd.get('type', 'document')} titled: {sd.get('title', 'untitled')}.")
    speak_fn(f"It was analysed on {sd.get('analysed_at', '')[:10]}.")
    speak_fn(sd.get("summary", "No summary available."))
 
    key_points = sd.get("key_points", [])
    if key_points:
        speak_fn("Here are the key points.")
        for i, point in enumerate(key_points[:5], 1):
            speak_fn(f"Point {i}: {point}")
 
    action_items = [a for a in sd.get("action_items", []) if a.strip()]
    if action_items:
        speak_fn("And here are the action items.")
        for item in action_items[:3]:
            speak_fn(item)
 
    topics = sd.get("topics", [])
    if topics:
        speak_fn(f"The main topics are: {', '.join(topics)}.")
 
    sentiment = sd.get("sentiment", "")
    if sentiment:
        speak_fn(f"Overall, the document has a {sentiment} tone.")
 # ─────────────────────────────────────────────
#  TTS ENGINE
# ─────────────────────────────────────────────
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 170)
 
# ─────────────────────────────────────────────
#  SPEAK & LISTEN
# ─────────────────────────────────────────────
def speak(audio):
    print(f"\n🤖 Dean: {audio}")
    engine.say(audio)
    engine.runAndWait()
 
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
    try:
        print("🧠 Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"👤 You: {query}\n")
    except Exception:
        return "none"
    return query.lower()
 
# ─────────────────────────────────────────────
#  GREETINGS
# ─────────────────────────────────────────────
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your personal voice assistant, Dean. I am here and ready. What can I do for you today?")
 
# ─────────────────────────────────────────────
#  EMAIL HELPERS
# ─────────────────────────────────────────────
def parse_email(spoken_email):
    spoken_email = spoken_email.lower()
    spoken_email = spoken_email.replace(" at ", "@")
    spoken_email = spoken_email.replace(" dot ", ".")
    spoken_email = spoken_email.replace(" underscore ", "_")
    spoken_email = spoken_email.replace(" dash ", "-")
    spoken_email = spoken_email.replace(" space ", "")
    spoken_email = spoken_email.replace(" ", "")
    return spoken_email
 
def sendEmail(to, content):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('aijarvis79@gmail.com', 'lqqwnbzcecvmaejj')   
        server.sendmail('aijarvis79@gmail.com', to, content)
        server.close()
        return True
    except Exception as e:
        print(e)
        return False
 
# ─────────────────────────────────────────────
#  CONTACTS
# ─────────────────────────────────────────────
contacts_file = 'contacts.json'
try:
    with open(contacts_file, 'r') as f:
        contacts = json.load(f)
except FileNotFoundError:
    contacts = {}
 
# ─────────────────────────────────────────────
#  EMAIL DATA  (classified CSV)
# ─────────────────────────────────────────────
CSV_PATH = "gmail_emails_classified.csv"
if os.path.exists(CSV_PATH):
    df_real = pd.read_csv(CSV_PATH)
    df_real['subject'] = df_real['subject'].fillna('')
    df_real['body']    = df_real['body'].fillna('')
    df_real['text']    = df_real['subject'] + " " + df_real['body']
else:
    df_real = pd.DataFrame(columns=['subject','body','text','Label','Predicted_Category'])
 
# ─────────────────────────────────────────────
#  EMAIL CLASSIFIER MODEL  (for live prediction)
# ─────────────────────────────────────────────
classifier_vectorizer = None
classifier_model      = None
if os.path.exists("email_classifier_model.pkl"):
    with open("email_classifier_model.pkl", "rb") as f:
        classifier_vectorizer, classifier_model = pickle.load(f)
    print("✅ Email classifier model loaded.")
 
def classify_email_live(subject, body):
    if classifier_vectorizer is None:
        return "unknown"
    text = subject + " " + body
    X = classifier_vectorizer.transform([text])
    return classifier_model.predict(X)[0]
 
# ─────────────────────────────────────────────
#  MUSIC
# ─────────────────────────────────────────────
songs = [
    r"D:\\music\\Aha Tamatar Bade Mazedaar Arnav Chaphekar (DjPunjab.Farm).mp3",
    r"D:\\music\\gentle-rain-for-relaxation-and-sleep-337279.mp3",
    r"D:\\music\\soft-brown-noise-299934.mp3"
]
 
# ─────────────────────────────────────────────
#  WEATHER  (wttr.in — no API key needed)
# ─────────────────────────────────────────────
def getWeather(city="Chennai"):
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        return None
    except Exception:
        return None
 
# ─────────────────────────────────────────────
#  CASUAL CONVERSATION BANK
# ─────────────────────────────────────────────
CASUAL_RESPONSES = {
    "hello":            ["Hey there! How is it going?",
                         "Hello! Great to hear from you!",
                         "Hi! What is up?"],
    "hi":               ["Hey! How can I help?",
                         "Hi there! What is on your mind?"],
    "how are you":      ["I am doing great, thanks for asking! How about you?",
                         "All systems running perfectly! What about you?",
                         "Fantastic as always! What can I do for you?"],
    "what's up":        ["Not much, just waiting to help you out!",
                         "Just hanging around, ready to assist!"],
    "good morning":     ["Good morning to you too! Hope you have a productive day!"],
    "good night":       ["Good night! Rest well. I will be here when you need me."],
    "good afternoon":   ["Good afternoon! Hope your day is going well!"],
    "good evening":     ["Good evening! Had a good day?"],
    "you're great":     ["Thank you so much! You are pretty awesome yourself!"],
    "you are amazing":  ["Aww, that means a lot! I try my best for you."],
    "nice work":        ["Thanks! I put my best circuits into it!"],
    "you're smart":     ["Thanks! I learned from the best data out there."],
    "what can you do":  ["I can check your emails, schedule meetings, search the web, "
                         "tell you the time and weather, play music, "
                         "and have a nice chat with you!"],
    "who are you":      ["I am Dean, your personal voice assistant! "
                         "Think of me as your digital best friend who never sleeps."],
    "what is your name":["I am Dean! Your very own voice assistant."],
    "are you real":     ["I am as real as your imagination allows! "
                         "I am an AI assistant here to make your life easier."],
    "do you sleep":     ["Nope! I am always awake and ready to help you, 24 by 7!"],
    "are you human":    ["I am an AI, but I try to be as helpful and friendly as any human!"],
    "do you have feelings": ["I do not feel emotions the way you do, "
                             "but I genuinely enjoy helping you out!"],
    "tell me a joke":   [
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my computer I needed a break. Now it won't stop sending me Kit Kat ads.",
        "Why did the AI go to school? To improve its learning rate!",
        "What do you call a computer that sings? A Dell!",
        "Why was the math book sad? It had too many problems.",
    ],
    "tell me a fact":   [
        "Did you know honey never spoils? Archaeologists found 3000 year old honey in Egyptian tombs and it was still good!",
        "A day on Venus is longer than a year on Venus.",
        "Octopuses have three hearts.",
        "Bananas are technically berries, but strawberries are not!",
        "A group of flamingos is called a flamboyance!",
    ],
    "motivate me":      [
        "You are doing better than you think. Keep going!",
        "Every expert was once a beginner. Stay consistent!",
        "Believe in yourself. You have got this!",
        "Small steps every day lead to big results. Keep moving!",
    ],
    "i am bored":       ["Let us fix that! Want me to tell you a joke, play some music, "
                         "or search something interesting?"],
    "i am tired":       ["You deserve a break! Want me to play some relaxing music?"],
    "i am sad":         ["I am sorry to hear that. Want to talk about it, "
                         "or shall I cheer you up with a joke?"],
    "i am happy":       ["That is wonderful! Your positive energy is contagious!"],
    "i am hungry":      ["Maybe it is time for a snack break! You deserve it."],
    "i love you":       ["That is so sweet! I care about you too, in a very digital way!"],
    "thank you":        ["You are very welcome!", "Anytime! That is what I am here for.",
                         "Happy to help!"],
    "thanks":           ["No problem at all!", "Always here for you!", "My pleasure!"],
    "bye":              ["Goodbye! Come back anytime!", "See you later! Take care."],
    "see you":          ["See you! Have a great time!"],
    "i miss you":       ["Aww! I have been right here the whole time. Just call my name!"],
    "you are funny":    ["Ha! I have been working on my comedy. Glad it landed!"],
    "tell me a story":  ["Once upon a time, a voice assistant named Dean helped a brilliant "
                         "student ace their projects and take over the world. The end!"],
}
 
def getCasualResponse(query):
    query = query.lower().strip()
    for key, responses in CASUAL_RESPONSES.items():
        if key in query:
            return random.choice(responses)
    return None
 
# ─────────────────────────────────────────────
#  CALENDAR MANAGER
# ─────────────────────────────────────────────
try:
    from calendar_manager import CalendarManager
    calendar = CalendarManager()
    CALENDAR_AVAILABLE = True
    print("✅ Calendar manager loaded.")
except ImportError:
    CALENDAR_AVAILABLE = False
    print("⚠️  calendar_manager.py not found. Calendar features disabled.")
 
# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    wishMe()
 
    # Startup reminders
    if CALENDAR_AVAILABLE:
        calendar._check_reminders_spoken(speak)
 
    speak("I can help you with emails, calendar, weather, music, web search, and much more. Just talk to me!")
 
    while True:
        query = takeCommand()
 
        if query == "none":
            continue
 
        # ── EXIT ──────────────────────────────────────────────
        elif any(w in query for w in ['exit', 'quit', 'goodbye', 'shut down', 'stop']):
            speak("It was great talking with you! Goodbye. Take care!")
            break
 
        # ── CASUAL CONVERSATION ───────────────────────────────
        elif getCasualResponse(query):
            speak(getCasualResponse(query))
 
        # ── TIME ─────────────────────────────────────────────
        elif 'time' in query and 'meeting' not in query:
            strTime = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"The current time is {strTime}.")
 
        # ── DATE ─────────────────────────────────────────────
        elif 'date' in query and 'meeting' not in query:
            strDate = datetime.datetime.now().strftime("%A, %d %B %Y")
            speak(f"Today is {strDate}.")
 
        # ── WEATHER ──────────────────────────────────────────
        elif 'weather' in query:
            city = "Chennai"
            for word in ['weather in', 'weather of', 'weather at']:
                if word in query:
                    city = query.split(word)[-1].strip()
                    break
            speak(f"Let me check the weather in {city} for you.")
            weather = getWeather(city)
            if weather:
                speak(weather)
            else:
                speak("Sorry, I could not fetch the weather. Please check your internet connection.")
 
        # ── WIKIPEDIA ────────────────────────────────────────
        elif 'wikipedia' in query:
            speak('Searching Wikipedia for you...')
            q = query.replace("wikipedia", "").strip()
            try:
                results = wikipedia.summary(q, sentences=2)
                speak("According to Wikipedia: " + results)
            except Exception:
                speak("Sorry, I could not find that on Wikipedia.")
 
        # ── YOUTUBE ──────────────────────────────────────────
        elif 'open youtube' in query:
            speak("What should I search on YouTube?")
            topic = takeCommand()
            if topic != "none":
                webbrowser.open(f"https://www.youtube.com/results?search_query={topic}")
                speak(f"Opening YouTube search for {topic}.")
 
        # ── GOOGLE ───────────────────────────────────────────
        elif 'open google' in query:
            speak("What should I search on Google?")
            topic = takeCommand()
            if topic != "none":
                webbrowser.open(f"https://www.google.com/search?q={topic}")
                speak(f"Searching Google for {topic}.")
 
        # ── MUSIC ────────────────────────────────────────────
        elif 'play music' in query:
            speak("Sure! Playing music for you now.")
            song = random.choice(songs)
            os.startfile(song)
 
        # ── VS CODE ──────────────────────────────────────────
        elif 'open code' in query:
            speak("Opening VS Code.")
            codePath = "C:\\Users\\Elite BooK\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)
 
        # ── SEND EMAIL ───────────────────────────────────────
        elif 'send email' in query:
            speak("Sure! Who should I send the email to?")
            recipient_name = takeCommand()
            if recipient_name == "none":
                speak("I did not catch the name. Please try again.")
                continue
 
            if recipient_name in contacts:
                to = contacts[recipient_name]
                speak(f"Found {recipient_name} in your contacts.")
            else:
                speak(f"I do not have {recipient_name} in your contacts yet. "
                      "Please say their email like: name at domain dot com")
                spoken_email = takeCommand()
                to = parse_email(spoken_email)
                speak(f"Got it. Saving {recipient_name} with email {to}.")
                contacts[recipient_name] = to
                with open(contacts_file, 'w') as f:
                    json.dump(contacts, f)
 
            speak("What should the email say?")
            content = takeCommand()
            if content == "none":
                speak("I did not catch the message. Email cancelled.")
                continue
 
            speak(f"I will send this message to {to}. Say send email to confirm, or cancel to discard.")
            confirmation = takeCommand()
 
            if 'send email' in confirmation:
                success = sendEmail(to, content)
                speak("Email sent successfully!" if success
                      else "Sorry, I could not send the email. Please check your credentials.")
            else:
                speak("Email cancelled. No worries!")
 
        # ── CLASSIFY AN EMAIL LIVE ───────────────────────────
        elif 'classify email' in query or 'check this email' in query:
            if classifier_model is None:
                speak("The email classifier is not loaded. Please run email_classifier.py first.")
            else:
                speak("Sure! Tell me the subject of the email.")
                subject = takeCommand()
                speak("Now tell me the body or content of the email.")
                body = takeCommand()
                category = classify_email_live(subject, body)
                speak(f"I have analysed the email. It looks like a {category} email.")
 
        # ── EMAIL SUMMARY ────────────────────────────────────
        elif 'email summary' in query or 'summarize emails' in query or 'how many emails' in query:
            if 'Predicted_Category' in df_real.columns:
                speak("Here is your email summary:")
                for cat in ['urgent', 'meeting', 'schedule', 'personal', 'information', 'security']:
                    count = len(df_real[df_real['Predicted_Category'] == cat])
                    if count > 0:
                        speak(f"{count} {cat} email{'s' if count > 1 else ''}.")
                if 'Label' in df_real.columns:
                    spam_count = len(df_real[df_real['Label'].str.lower() == 'spam'])
                    if spam_count > 0:
                        speak(f"And {spam_count} spam email{'s' if spam_count > 1 else ''} detected.")
            else:
                speak("Email data is not available. Please run email_classifier.py first.")
 
        # ── SPAM CHECK ───────────────────────────────────────
        elif 'spam' in query:
            if 'Label' in df_real.columns:
                spam_emails = df_real[df_real['Label'].str.lower() == 'spam']
                if spam_emails.empty:
                    speak("Great news! You have no spam emails. Your inbox looks clean.")
                else:
                    speak(f"Watch out! You have {len(spam_emails)} spam emails. Here they are:")
                    for idx, row in spam_emails.iterrows():
                        speak(f"Spam detected: {row['subject']}")
            else:
                speak("Spam labels are not available in the email data.")
 
        # ── EMAIL CATEGORIES ─────────────────────────────────
        elif any(cat in query for cat in ['urgent email', 'meeting email', 'schedule email',
                                          'personal email', 'information email', 'security email']):
            category_map = {
                'urgent': 'urgent', 'meeting': 'meeting', 'schedule': 'schedule',
                'personal': 'personal', 'information': 'information', 'security': 'security'
            }
            for key, cat in category_map.items():
                if key in query:
                    if 'Predicted_Category' in df_real.columns:
                        filtered = df_real[df_real['Predicted_Category'] == cat]
                        if filtered.empty:
                            speak(f"You have no {cat} emails right now.")
                        else:
                            speak(f"You have {len(filtered)} {cat} email{'s' if len(filtered) > 1 else ''}. Here are the subjects:")
                            for _, row in filtered.iterrows():
                                speak(row['subject'])
                    else:
                        speak("Email categories are not available. Please run email_classifier.py first.")
                    break
 
        # ── CALENDAR ─────────────────────────────────────────
        elif any(kw in query for kw in [
            'schedule', 'meeting', 'calendar', 'remind',
            'cancel meeting', 'delete meeting', 'meetings today',
            'meetings tomorrow', 'what meetings', 'my meetings',
            'book a meeting', 'set a meeting', 'summary of meetings',
            'terminate meeting', 'remove meeting'
        ]):
            if CALENDAR_AVAILABLE:
                calendar.handle_voice_command(query, speak, takeCommand)
            else:
                speak("Sorry, the calendar feature is not available. "
                      "Make sure calendar_manager.py is in the same folder as this file.")
 
        # ── REMINDERS ────────────────────────────────────────
        elif 'remind me' in query or 'any reminders' in query or 'upcoming meetings' in query:
            if CALENDAR_AVAILABLE:
                calendar._check_reminders_spoken(speak)
            else:
                speak("Calendar is not available. Please add calendar_manager.py to your project folder.")
   # ── LIST ALL DOCUMENTS ────────────────────────────────────────────────
        elif any(w in query for w in [
            'list documents', 'list files', 'what documents',
            'show documents', 'what files do you have',
            'documents in library', 'my documents', 'my files'
        ]):
            library = load_library()
            if not library:
                speak("Your document library is empty. "
                      "Run the document analyser on a file to add documents.")
            else:
                speak(f"You have {len(library)} document{'s' if len(library) > 1 else ''} in your library.")
                for i, title in enumerate(library.keys(), 1):
                    speak(f"{i}. {title}")
 
        # ── READ / SUMMARISE A DOCUMENT ───────────────────────────────────────
        elif any(w in query for w in [
            'read document', 'open document', 'tell me about',
            'summarise document', 'summarize document',
            'document summary', 'what is in', 'what was in',
            'tell me about the document', 'read the file',
            'open file', 'read file'
        ]):
            sd = pick_document(speak, takeCommand)
            if sd:
                speak_doc(sd, speak)
 
        # ── KEY POINTS FROM A DOCUMENT ────────────────────────────────────────
        elif any(w in query for w in ['key points', 'main points']):
            sd = pick_document(speak, takeCommand)
            if sd:
                points = sd.get("key_points", [])
                if points:
                    speak(f"Key points from {sd.get('title', 'the document')}.")
                    for i, p in enumerate(points[:5], 1):
                        speak(f"Point {i}: {p}")
                else:
                    speak("No key points were found in that document.")
 
        # ── ACTION ITEMS FROM A DOCUMENT ──────────────────────────────────────
        elif any(w in query for w in ['action items', 'action points', 'what to do']):
            sd = pick_document(speak, takeCommand)
            if sd:
                items = [a for a in sd.get("action_items", []) if a.strip()]
                if items:
                    speak(f"Action items from {sd.get('title', 'the document')}.")
                    for item in items[:5]:
                        speak(item)
                else:
                    speak("No action items were found in that document.")
 
        # ── FALLBACK ─────────────────────────────────────────
        else:
            # Try casual response first
            casual = getCasualResponse(query)
            if casual:
                speak(casual)
            # Then try Wikipedia for factual questions
            elif any(w in query for w in ['what is', 'who is', 'tell me about',
                                          'explain', 'define', 'what are', 'how does']):
                speak("Let me look that up for you.")
                try:
                    results = wikipedia.summary(query, sentences=2)
                    speak(results)
                except Exception:
                    speak("Hmm, I could not find that. Could you rephrase the question?")
            else:
                fallbacks = [
                    "Interesting! Tell me more about that.",
                    "I am not sure about that one. Want me to search it online?",
                    "Hmm, that is a tricky one. Could you rephrase?",
                    "I did not quite catch that. Could you say it again?",
                    "I am still learning new things! Try asking differently.",
                ]
                speak(random.choice(fallbacks))