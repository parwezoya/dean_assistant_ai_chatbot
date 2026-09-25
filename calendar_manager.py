import json
import os
import re
import uuid
import datetime
 
MEETINGS_FILE = "meetings.json"
 
# ─────────────────────────────────────────────
#  FILE HELPERS
# ─────────────────────────────────────────────
def _load():
    if os.path.exists(MEETINGS_FILE):
        try:
            with open(MEETINGS_FILE, "r") as f:
                data = json.load(f)
                print(f"📂 Loaded {len(data)} meetings from {MEETINGS_FILE}")
                return data
        except Exception as e:
            print(f"⚠️  Could not load meetings file: {e}")
            return []
    print(f"📂 No meetings file found. Starting fresh.")
    return []
 
 
def _save(meetings):
    try:
        with open(MEETINGS_FILE, "w") as f:
            json.dump(meetings, f, indent=2)
        print(f"✅ Saved {len(meetings)} meeting(s) to {MEETINGS_FILE}")
    except Exception as e:
        print(f"❌ SAVE FAILED: {e}")
 
 
def _gen_id():
    return str(uuid.uuid4())[:8]
 
 
# ─────────────────────────────────────────────
#  DATE PARSER
# ─────────────────────────────────────────────
def _parse_date(text):
    if not text or text.strip() == "":
        return None
 
    text = text.lower().strip()
    today = datetime.date.today()
 
    # natural shortcuts
    if "today"    in text: return today
    if "tomorrow" in text: return today + datetime.timedelta(days=1)
    if "day after tomorrow" in text: return today + datetime.timedelta(days=2)
 
    # weekday names
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, wd in enumerate(weekdays):
        if wd in text:
            days_ahead = (i - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7   # next occurrence
            return today + datetime.timedelta(days=days_ahead)
 
    # month name spellings
    months = {
        "january":"01","february":"02","march":"03","april":"04",
        "may":"05","june":"06","july":"07","august":"08",
        "september":"09","october":"10","november":"11","december":"12",
        "jan":"01","feb":"02","mar":"03","apr":"04","jun":"06",
        "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12",
    }
 
    # "15 april 2025" or "april 15 2025" or "15 april" etc.
    for month_word, month_num in months.items():
        if month_word in text:
            # extract numbers from text
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                # try day + year
                day  = nums[0].zfill(2)
                year = nums[1] if len(nums[1]) == 4 else str(today.year)
                try:
                    return datetime.date(int(year), int(month_num), int(day))
                except ValueError:
                    pass
            elif len(nums) == 1:
                day  = nums[0].zfill(2)
                year = str(today.year)
                try:
                    d = datetime.date(int(year), int(month_num), int(day))
                    if d < today:
                        d = datetime.date(today.year + 1, int(month_num), int(day))
                    return d
                except ValueError:
                    pass
 
    # numeric formats: dd-mm-yyyy, dd/mm/yyyy, yyyy-mm-dd
    for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y", "%d/%m/%y"]:
        try:
            return datetime.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
 
    return None
 
 
# ─────────────────────────────────────────────
#  TIME PARSER  (the main fix)
# ─────────────────────────────────────────────
def _parse_time(text):
    if not text or text.strip() == "":
        return None
 
    original = text
    text = text.lower().strip()
 
    # ── Step 1: word → digit replacements ─────
    word_nums = {
        "twelve":   "12", "eleven": "11", "ten":    "10",
        "nine":      "9", "eight":   "8", "seven":   "7",
        "six":       "6", "five":    "5", "four":    "4",
        "three":     "3", "two":     "2", "one":     "1",
        "zero":      "0",
        "thirty":   "30", "fifteen": "15", "forty five": "45",
        "forty-five":"45","quarter": "15", "half":   "30",
        "o'clock":  "00", "oclock":  "00", "o clock":"00",
        "oh":       "0",
    }
    for word, num in word_nums.items():
        text = text.replace(word, num)
 
    # ── Step 2: handle "half past X" → X:30 and "quarter to X" → (X-1):45
    half_past = re.search(r'30\s*past\s*(\d+)', text)
    if half_past:
        h = int(half_past.group(1))
        text = f"{h}:30"
 
    quarter_to = re.search(r'15\s*to\s*(\d+)', text)
    if quarter_to:
        h = int(quarter_to.group(1)) - 1
        text = f"{h}:45"
 
    quarter_past = re.search(r'15\s*past\s*(\d+)', text)
    if quarter_past:
        h = int(quarter_past.group(1))
        text = f"{h}:15"
 
    # ── Step 3: clean up spacing around AM/PM ─
    text = re.sub(r'\s*(a\.?m\.?)', ' a.m.', text)
    text = re.sub(r'\s*(p\.?m\.?)', ' p.m.', text)
    text = text.strip()
 
    # ── Step 4: try standard strptime formats ─
    formats = [
        "%I:%M %p",   # 10:30 AM
        "%H:%M",      # 14:30
        "%I %p",      # 10 AM
        "%I:%M%p",    # 10:30AM
        "%I%p",       # 10AM
        "%H.%M",      # 14.30
        "%I.%M %p",   # 10.30 AM
    ]
    for fmt in formats:
        try:
            t = datetime.datetime.strptime(text.strip(), fmt).time()
            print(f"🕐 Parsed time: '{original}' → {t.strftime('%I:%M %p')}")
            return t
        except ValueError:
            continue
 
    # ── Step 5: regex fallback — extract H:M + AM/PM ─
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(a.m.|p.m.)?', text)
    if match:
        hour   = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm   = match.group(3)
        if ampm == 'p.m.' and hour != 12:
            hour += 12
        elif ampm == 'a.m.' and hour == 12:
            hour = 0
        try:
            t = datetime.time(hour, minute)
            print(f"🕐 Parsed time (regex): '{original}' → {t.strftime('%I:%M %p')}")
            return t
        except ValueError:
            pass
 
    print(f"❌ Could not parse time: '{original}'")
    return None
 
 
# ─────────────────────────────────────────────
#  FORMAT HELPERS
# ─────────────────────────────────────────────
def _fmt_date(d):
    return d.strftime("%d %B %Y, %A")
 
def _fmt_time(t):
    return t.strftime("%I:%M %p").lstrip("0")
 
 
# ─────────────────────────────────────────────
#  CLASH DETECTION
# ─────────────────────────────────────────────
def _check_clash(meetings, date, start, end, exclude_id=None):
    clashes = []
    for m in meetings:
        if m.get("id") == exclude_id:
            continue
        if m["date"] != date.isoformat():
            continue
        ms = datetime.time.fromisoformat(m["start_time"])
        me = datetime.time.fromisoformat(m["end_time"])
        if not (end <= ms or start >= me):
            clashes.append(m)
    return clashes
 
 
# ─────────────────────────────────────────────
#  CALENDAR MANAGER CLASS
# ─────────────────────────────────────────────
class CalendarManager:
 
    def __init__(self):
        self.meetings = _load()
        self._check_reminders_silent()
 
    # ── ROUTER ──────────────────────────────
    def handle_voice_command(self, query, speak, listen):
        q = query.lower()
 
        if any(kw in q for kw in ["cancel", "delete", "remove", "terminate"]):
            self._cancel_flow(speak, listen)
 
        elif any(kw in q for kw in ["today", "tomorrow", "summary",
                                    "what meetings", "my meetings", "meetings on"]):
            self._summary_flow(speak, listen)
 
        elif "remind" in q or "upcoming" in q:
            self._check_reminders_spoken(speak)
 
        else:
            speak("Let us schedule a meeting.")
            self._schedule_flow(speak, listen)
 
    # ── SCHEDULE FLOW ───────────────────────
    def _schedule_flow(self, speak, listen):
 
        # ── TITLE ─────────────────────────────
        speak("What is the title of the meeting?")
        title = listen().strip()
        if not title or title == "none":
            speak("I did not catch the title. Please try again.")
            return
        print(f"📝 Title: {title}")
 
        # ── DATE ──────────────────────────────
        date = None
        for attempt in range(3):
            speak("What date? Say today, tomorrow, a weekday, or a date like 15 April.")
            date_str = listen()
            print(f"📅 Date input: '{date_str}'")
            date = _parse_date(date_str)
            if date is None:
                speak("Sorry, I could not get that date. Please try again.")
            elif date < datetime.date.today():
                speak("That date is in the past. Please give a future date.")
                date = None
            else:
                speak(f"Date set to {_fmt_date(date)}.")
                break
 
        if date is None:
            speak("Could not get a valid date. Please try again later.")
            return
 
        # ── START TIME ────────────────────────
        start_time = None
        for attempt in range(3):
            speak("What time should the meeting start? For example: 10 AM, or 2 30 PM.")
            t_str = listen()
            print(f"⏰ Time input: '{t_str}'")
            start_time = _parse_time(t_str)
            if start_time is None:
                speak("I could not understand that time. Try saying something like 10 AM or 3 30 PM.")
            else:
                speak(f"Start time set to {_fmt_time(start_time)}.")
                break
 
        if start_time is None:
            speak("Could not get a valid start time. Please try again later.")
            return
 
        # ── DURATION ──────────────────────────
        duration_min = None
        for attempt in range(3):
            speak("How many minutes will the meeting last? Say 30, 45, or 60.")
            dur_str = listen()
            print(f"⏱️  Duration input: '{dur_str}'")
            nums = re.findall(r'\d+', dur_str)
            if nums:
                duration_min = int(nums[0])
                speak(f"Duration set to {duration_min} minutes.")
                break
            else:
                speak("Please say just a number like 30 or 60.")
 
        if duration_min is None:
            speak("Could not get the duration. Please try again later.")
            return
 
        # calculate end time
        end_dt   = datetime.datetime.combine(date, start_time) + datetime.timedelta(minutes=duration_min)
        end_time = end_dt.time()
        speak(f"Meeting will run from {_fmt_time(start_time)} to {_fmt_time(end_time)}.")
 
        # ── CLASH CHECK ───────────────────────
        clashes = _check_clash(self.meetings, date, start_time, end_time)
        if clashes:
            speak(f"Warning! You already have {len(clashes)} meeting at that time.")
            for c in clashes:
                cs = datetime.time.fromisoformat(c["start_time"])
                ce = datetime.time.fromisoformat(c["end_time"])
                speak(f"{c['title']} from {_fmt_time(cs)} to {_fmt_time(ce)}.")
            speak("Do you still want to schedule this? Say yes or no.")
            ans = listen()
            if "yes" not in ans.lower():
                speak("Okay, meeting not scheduled. Try a different time.")
                return
 
        # ── ATTENDEES ─────────────────────────
        speak("Who are the attendees? Name them or say skip.")
        att_raw = listen()
        if "skip" in att_raw.lower() or att_raw == "none":
            attendees = []
        else:
            attendees = [a.strip() for a in att_raw.replace(" and ", ",").split(",") if a.strip()]
        print(f"👥 Attendees: {attendees}")
 
        # ── NOTES ─────────────────────────────
        speak("Any notes or agenda? Say skip to leave blank.")
        notes_raw = listen()
        notes = "" if ("skip" in notes_raw.lower() or notes_raw == "none") else notes_raw.strip()
        print(f"📓 Notes: {notes}")
 
        # ── CONFIRM ───────────────────────────
        att_str = ", ".join(attendees) if attendees else "none"
        speak(f"Confirming: {title}, on {_fmt_date(date)}, "
              f"from {_fmt_time(start_time)} to {_fmt_time(end_time)}, "
              f"attendees: {att_str}. "
              f"Say confirm to save or cancel to discard.")
        confirm = listen().lower()
        print(f"✅ Confirmation response: '{confirm}'")
 
        if "confirm" not in confirm:
            speak("Meeting discarded.")
            return
 
        # ── SAVE ──────────────────────────────
        meeting = {
            "id":         _gen_id(),
            "title":      title,
            "date":       date.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time":   end_time.isoformat(),
            "duration":   duration_min,
            "attendees":  attendees,
            "notes":      notes,
            "created_at": datetime.datetime.now().isoformat(),
        }
        self.meetings.append(meeting)
        _save(self.meetings)
 
        speak(f"Done! {title} has been saved for {_fmt_date(date)} at {_fmt_time(start_time)}.")
 
    # ── CANCEL FLOW ─────────────────────────
    def _cancel_flow(self, speak, listen):
        today    = datetime.date.today()
        upcoming = sorted(
            [m for m in self.meetings
             if datetime.date.fromisoformat(m["date"]) >= today],
            key=lambda m: (m["date"], m["start_time"])
        )
 
        if not upcoming:
            speak("You have no upcoming meetings to cancel.")
            return
 
        speak(f"You have {len(upcoming)} upcoming meeting{'s' if len(upcoming)>1 else ''}.")
        for i, m in enumerate(upcoming, 1):
            d = datetime.date.fromisoformat(m["date"])
            t = datetime.time.fromisoformat(m["start_time"])
            speak(f"Option {i}: {m['title']} on {_fmt_date(d)} at {_fmt_time(t)}.")
 
        speak("Say the number of the meeting to cancel, or say back.")
        choice = listen()
 
        if "back" in choice.lower():
            speak("No changes made.")
            return
 
        nums = re.findall(r'\d+', choice)
        if not nums:
            speak("I did not catch a number. No changes made.")
            return
 
        idx = int(nums[0]) - 1
        if idx < 0 or idx >= len(upcoming):
            speak("That number is out of range.")
            return
 
        chosen = upcoming[idx]
        speak(f"Say yes to confirm cancelling {chosen['title']}.")
        confirm = listen().lower()
 
        if "yes" in confirm:
            self.meetings = [m for m in self.meetings if m["id"] != chosen["id"]]
            _save(self.meetings)
            speak(f"Done. {chosen['title']} has been removed from your calendar.")
        else:
            speak("Cancellation aborted.")
 
    # ── SUMMARY FLOW ────────────────────────
    def _summary_flow(self, speak, listen):
        speak("Which date? Say today, tomorrow, or a specific date.")
        date_str = listen()
        date     = _parse_date(date_str)
 
        if date is None:
            speak("I could not understand that date.")
            return
 
        day_meetings = sorted(
            [m for m in self.meetings if m["date"] == date.isoformat()],
            key=lambda m: m["start_time"]
        )

        if not day_meetings:
            speak(f"You have no meetings on {_fmt_date(date)}.")
            return
 
        speak(f"You have {len(day_meetings)} meeting{'s' if len(day_meetings)>1 else ''} on {_fmt_date(date)}.")
        for i, m in enumerate(day_meetings, 1):
            st  = datetime.time.fromisoformat(m["start_time"])
            et  = datetime.time.fromisoformat(m["end_time"])
            att = ", ".join(m["attendees"]) if m["attendees"] else "no attendees listed"
            speak(f"Meeting {i}: {m['title']}, from {_fmt_time(st)} to {_fmt_time(et)}, attendees: {att}.")
            if m.get("notes"):
                speak(f"Notes: {m['notes']}")
 
    # ── REMINDERS (spoken) ──────────────────
    def _check_reminders_spoken(self, speak):
        today    = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        now      = datetime.datetime.now()
        found    = False
 
        for m in self.meetings:
            d = datetime.date.fromisoformat(m["date"])
            t = datetime.time.fromisoformat(m["start_time"])
 
            if d == today:
                diff = (datetime.datetime.combine(d, t) - now).total_seconds() / 60
                if 0 <= diff <= 60:
                    speak(f"Heads up! {m['title']} starts in {int(diff)} minutes at {_fmt_time(t)}.")
                    found = True
                elif diff < 0 and abs(diff) < m.get("duration", 60):
                    speak(f"Your meeting {m['title']} is currently ongoing!")
                    found = True
 
            elif d == tomorrow:
                speak(f"Tomorrow you have {m['title']} at {_fmt_time(t)}.")
                found = True
 
        if not found:
            speak("No meetings coming up in the next 60 minutes.")
 
    # ── REMINDERS (silent startup) ──────────
    def _check_reminders_silent(self):
        today = datetime.date.today()
        now   = datetime.datetime.now()
        for m in self.meetings:
            d = datetime.date.fromisoformat(m["date"])
            t = datetime.time.fromisoformat(m["start_time"])
            if d == today:
                diff = (datetime.datetime.combine(d, t) - now).total_seconds() / 60
                if 0 <= diff <= 60:
                    print(f"⏰  Upcoming in ~{int(diff)} min → '{m['title']}' at {_fmt_time(t)}")