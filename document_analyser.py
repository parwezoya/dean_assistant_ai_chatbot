import os
import sys
import json
import re
import datetime
from collections import Counter
 
# ── nltk setup ───────────────────────────────────────────────────────────────
import nltk
for pkg in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)
 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
 
# ── sumy ─────────────────────────────────────────────────────────────────────
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False
    print("⚠️  sumy not installed. Run: pip install sumy")
 
# ── python-docx ──────────────────────────────────────────────────────────────
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx not installed. Run: pip install python-docx")
 
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LIBRARY_JSON = "document_library.json"   # all documents stored here
LANGUAGE     = "english"
 
ACTION_KEYWORDS = [
    'need to', 'must', 'should', 'will', 'action', 'follow up',
    'deadline', 'due', 'assign', 'complete', 'finish', 'submit',
    'send', 'schedule', 'plan', 'prepare', 'review', 'update',
    'check', 'confirm', 'ensure', 'provide', 'deliver', 'fix',
    'resolve', 'contact', 'implement'
]
 
DOC_TYPE_HINTS = {
    'meeting notes':  ['meeting', 'attendees', 'agenda', 'minutes', 'discussed'],
    'report':         ['report', 'analysis', 'findings', 'conclusion', 'results'],
    'letter':         ['dear', 'sincerely', 'regards', 'yours truly'],
    'invoice':        ['invoice', 'amount due', 'payment', 'bill', 'total'],
    'contract':       ['agreement', 'parties', 'terms', 'clause', 'hereby'],
    'resume':         ['experience', 'education', 'skills', 'objective', 'linkedin'],
    'research paper': ['abstract', 'introduction', 'methodology', 'hypothesis'],
    'notes':          ['note', 'notes', 'todo', 'to-do', 'reminder'],
}
 
POSITIVE_WORDS = {
    'good','great','excellent','success','positive','achieved','approved',
    'completed','well','happy','improve','benefit','agree','confirmed',
    'resolved','effective','efficient','outstanding','productive'
}
NEGATIVE_WORDS = {
    'bad','issue','problem','fail','negative','concern','risk','delay',
    'reject','error','conflict','cancel','loss','urgent','critical',
    'overdue','missed','incomplete','blocked','stuck'
}
 
# ─────────────────────────────────────────────────────────────────────────────
#  LIBRARY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def load_library():
    """Load existing document library or return empty dict."""
    if not os.path.exists(LIBRARY_JSON):
        return {}
    try:
        with open(LIBRARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
 
def save_library(library):
    with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
 
# ─────────────────────────────────────────────────────────────────────────────
#  TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 
def extract_text_from_docx(path):
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
 
def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .txt or .docx")
 
# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def detect_title(text, filename):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and len(lines[0]) < 100:
        return lines[0]
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
 
def detect_doc_type(text, filename):
    lower = text.lower() + " " + filename.lower()
    scores = {dt: sum(1 for kw in kws if kw in lower)
              for dt, kws in DOC_TYPE_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "document"
 
def build_summary(text, num_sentences=4):
    sentences = sent_tokenize(text)
    if not sentences:
        return "The document appears to be empty."
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    if SUMY_AVAILABLE:
        try:
            parser     = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
            stemmer    = Stemmer(LANGUAGE)
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words(LANGUAGE)
            result  = summarizer(parser.document, num_sentences)
            summary = " ".join(str(s) for s in result)
            if summary.strip():
                return summary
        except Exception:
            pass
    return " ".join(sentences[:num_sentences])
 
def extract_key_points(text, max_points=5):
    lines  = text.splitlines()
    points = []
    for line in lines:
        s = line.strip()
        if re.match(r'^(\-|\*|•|\d+[\.\)])\s+.{8,}', s):
            points.append(re.sub(r'^(\-|\*|•|\d+[\.\)])\s+', '', s))
        elif s.isupper() and 5 < len(s) < 80:
            points.append(s.title())
    if len(points) < 3:
        stop  = set(stopwords.words(LANGUAGE))
        sents = sent_tokenize(text)
        freq  = Counter(w for w in word_tokenize(text.lower())
                        if w not in stop and w.isalpha() and len(w) > 2)
        ranked = sorted(sents, key=lambda s: sum(freq.get(w,0) for w in word_tokenize(s.lower())), reverse=True)
        for s in ranked:
            if s not in points:
                points.append(s)
            if len(points) >= max_points:
                break
    return points[:max_points]
 
def extract_action_items(text):
    return [s.strip() for s in sent_tokenize(text)
            if any(kw in s.lower() for kw in ACTION_KEYWORDS)][:5]
 
def extract_topics(text, top_n=6):
    stop  = set(stopwords.words(LANGUAGE))
    words = word_tokenize(text.lower())
    content = [w for w in words if w not in stop and w.isalpha() and len(w) > 3]
    return [w for w, _ in Counter(content).most_common(top_n)]
 
def detect_sentiment(text):
    words = set(word_tokenize(text.lower()))
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
    if pos > neg:  return "positive"
    if neg > pos:  return "negative"
    return "neutral"
 
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  –  analyse & save into library
# ─────────────────────────────────────────────────────────────────────────────
 
def analyse_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
 
    filename = os.path.basename(file_path)
    print(f"\n📄 Reading: {file_path}")
    raw_text   = extract_text(file_path)
    word_count = len(raw_text.split())
    print(f"   Extracted ~{word_count} words")
    print("🧠 Analysing locally (no internet needed)...")
 
    title = detect_title(raw_text, filename)
 
    summary_data = {
        "title":             title,
        "filename":          filename,
        "source_file":       os.path.abspath(file_path),
        "analysed_at":       datetime.datetime.now().isoformat(),
        "type":              detect_doc_type(raw_text, filename),
        "summary":           build_summary(raw_text, num_sentences=4),
        "key_points":        extract_key_points(raw_text, max_points=5),
        "word_count_approx": word_count,
        "topics":            extract_topics(raw_text, top_n=6),
        "sentiment":         detect_sentiment(raw_text),
        "action_items":      extract_action_items(raw_text),
    }
 
    # Load existing library and add/overwrite this document by title
    library = load_library()
    library[title] = summary_data
    save_library(library)
 
    print(f"\n✅ Saved as '{title}' in {LIBRARY_JSON}")
    print(f"   Library now contains {len(library)} document(s):")
    for i, t in enumerate(library.keys(), 1):
        print(f"   {i}. {t}")
    print()
    return summary_data
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to your .txt or .docx file: ").strip().strip('"')
 
    try:
        analyse_document(path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
        