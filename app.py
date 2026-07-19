# import streamlit as st 
# import pickle 
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer as ps
# import string

# # FIX 1: Initialize the PorterStemmer object here
# stemmer = ps()

# # Speed Optimization: Convert stopwords to a set for faster processing in Streamlit
# STOP_WORDS = set(stopwords.words('english'))
# PUNCTUATION = set(string.punctuation)

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in STOP_WORDS and i not in PUNCTUATION:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         # FIX 2: Use the initialized 'stemmer' object instead of the 'ps' alias
#         y.append(stemmer.stem(i))
    
#     return " ".join(y)

# # Load your pickle files
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message") # Tip: text_area is better for multi-line messages

# if st.button("Predict"):
#     if input_sms.strip() == "":
#         st.warning("Please enter a message first!")
#     else:
#         transformed_sms = transform_text(input_sms)
#         vector_input = tfidf.transform([transformed_sms])
#         result = model.predict(vector_input)[0]

#         if result == 1:
#             st.error("🚨 Spam")
#         else:
#             st.success("✅ Not Spam")


import streamlit as st
import pickle
import time
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# One-time setup: NLTK data + heavy resources (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), PorterStemmer()

STOP_WORDS, stemmer = setup_nltk()
PUNCTUATION = set(string.punctuation)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_artifacts()

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in PUNCTUATION]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens), tokens

# ---------------------------------------------------------------------------
# Session state — running history + stats
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: text, label, confidence
if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""

EXAMPLES = {
    "🎁 Prize scam": "Congratulations! You've WON a $1000 Walmart gift card. Click here now to claim your prize before it expires!!!",
    "📅 Normal message": "Hey, are we still on for lunch tomorrow at 1pm? Let me know if that works.",
    "🏦 Phishing": "URGENT: Your bank account has been suspended. Verify your details immediately at this link to avoid permanent closure.",
}

def use_example(text):
    st.session_state.sms_input = text

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(-45deg, #050914, #0d1b2a, #16213e, #0f3460);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main .block-container {
    animation: fadeUp 0.8s ease-out;
    padding-top: 2rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-title {
    text-align: center;
    font-size: 2.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite, popIn 0.7s ease-out;
    margin-bottom: 0.2rem;
}

@keyframes shine { to { background-position: 200% center; } }

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    60% { opacity: 1; transform: scale(1.03); }
    100% { transform: scale(1); }
}

.shield-icon {
    display: inline-block;
    animation: guard 2.4s ease-in-out infinite;
}

@keyframes guard {
    0%, 100% { transform: rotate(0deg) scale(1); }
    25% { transform: rotate(-6deg) scale(1.06); }
    75% { transform: rotate(6deg) scale(1.06); }
}

.hero-subtitle {
    text-align: center;
    color: #cfd8ea;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 1.6rem;
    animation: fadeUp 1s ease-out;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 18px !important;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    animation: fadeUp 0.9s ease-out;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 28px rgba(58, 123, 213, 0.20);
    border: 1px solid rgba(58, 123, 213, 0.4) !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
    color: #e6e6f0 !important;
}

label, .stMarkdown p, .stMarkdown li { color: #e8e8f2 !important; }

textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.04) !important;
    color: #f2f2f8 !important;
    transition: border 0.3s ease, box-shadow 0.3s ease;
}

textarea:focus {
    border: 1px solid #3a7bd5 !important;
    box-shadow: 0 0 0 3px rgba(58, 123, 213, 0.25) !important;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.1rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    box-shadow: 0 6px 20px rgba(58, 123, 213, 0.35);
    transition: all 0.35s ease;
}

div.stButton > button:hover {
    background-position: right center;
    transform: scale(1.02) translateY(-2px);
    box-shadow: 0 10px 28px rgba(58, 123, 213, 0.55);
}

div.stButton > button:active { transform: scale(0.98); }

.result-card {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: resultPop 0.6s cubic-bezier(0.26, 1.36, 0.44, 1);
    margin-top: 1rem;
}

@keyframes resultPop {
    0% { opacity: 0; transform: scale(0.8) translateY(15px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.result-spam {
    background: linear-gradient(135deg, rgba(255, 65, 108, 0.22), rgba(255, 75, 43, 0.10));
    border: 1px solid rgba(255, 90, 90, 0.45);
    box-shadow: 0 0 40px rgba(255, 65, 108, 0.25);
}

.result-safe {
    background: linear-gradient(135deg, rgba(46, 213, 115, 0.20), rgba(0, 210, 211, 0.10));
    border: 1px solid rgba(46, 213, 115, 0.45);
    box-shadow: 0 0 40px rgba(46, 213, 115, 0.25);
}

.result-icon { font-size: 3rem; animation: bounceIn 0.8s ease-out; }

@keyframes bounceIn {
    0% { transform: scale(0); }
    60% { transform: scale(1.25); }
    100% { transform: scale(1); }
}

.result-heading { font-size: 1.5rem; font-weight: 700; margin-top: 0.4rem; color: white; }
.result-sub { color: #d8d8e6; font-size: 0.92rem; margin-top: 0.3rem; }

.token-chip {
    display: inline-block;
    background: rgba(58, 123, 213, 0.18);
    border: 1px solid rgba(58, 123, 213, 0.4);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    color: #cfe8ff;
    margin: 0.15rem;
    animation: chipIn 0.4s ease-out backwards;
}

@keyframes chipIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.history-item {
    border-left: 3px solid;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.5rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.04);
    font-size: 0.85rem;
    animation: fadeUp 0.4s ease-out;
}

.history-spam { border-color: #ff416c; }
.history-safe { border-color: #2ed573; }

.stat-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}

.stat-num { font-size: 1.4rem; font-weight: 700; color: #00d2ff; }
.stat-label { font-size: 0.72rem; color: #9fb3d8; }

.footer-note {
    text-align: center;
    color: #8f8fae;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    animation: fadeUp 1.2s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ About Spam Shield")
    st.markdown(
        "This app uses a trained **ML text classifier** (TF-IDF + your "
        "model) to flag whether an email or SMS message is spam."
    )
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["label"] == "spam")
    safe_count = total - spam_count

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat-chip'><div class='stat-num'>{total}</div><div class='stat-label'>Checked</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-chip'><div class='stat-num'>{spam_count}</div><div class='stat-label'>Spam Found</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.history:
        st.markdown("### 🕓 Recent Checks")
        for h in reversed(st.session_state.history[-6:]):
            css_class = "history-spam" if h["label"] == "spam" else "history-safe"
            icon = "🚨" if h["label"] == "spam" else "✅"
            preview = h["text"][:45] + ("…" if len(h["text"]) > 45 else "")
            st.markdown(
                f"<div class='history-item {css_class}'>{icon} {preview}</div>",
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            "<span style='font-size:0.82rem;color:#9a9ac0;'>No messages checked yet.</span>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero-title'><span class='shield-icon'>🛡️</span> Spam Shield</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-subtitle'>Paste any email or SMS below and get an "
    "instant spam check</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example quick-fill buttons
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header' style='color:#9fb3d8;font-size:0.9rem;margin-bottom:0.4rem;'>Try an example</div>", unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    with col:
        st.button(
            label,
            key=f"ex_{label}",
            use_container_width=True,
            on_click=use_example,
            args=(text,),
        )

# ---------------------------------------------------------------------------
# Input card
# ---------------------------------------------------------------------------
input_card = st.container(border=True)
with input_card:
    input_sms = st.text_area(
        "Enter the message",
        height=150,
        placeholder="Paste your email or SMS text here...",
        key="sms_input",
    )

    word_count = len(input_sms.split())
    char_count = len(input_sms)
    st.markdown(
        f"<span style='font-size:0.78rem;color:#8f9fc0;'>📝 {word_count} words · {char_count} characters</span>",
        unsafe_allow_html=True,
    )

    def clear_input():
        st.session_state.sms_input = ""

    b1, b2 = st.columns([3, 1])
    with b1:
        predict_clicked = st.button("🔍 Check for Spam", use_container_width=True)
    with b2:
        st.button("✖ Clear", use_container_width=True, on_click=clear_input)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        with st.spinner("Scanning message..."):
            transformed_sms, tokens = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector_input)[0]
                confidence = proba[int(result)] * 100

            time.sleep(0.4)

        label = "spam" if result == 1 else "safe"
        st.session_state.history.append({"text": input_sms, "label": label})

        conf_text = f"Confidence: {confidence:.1f}%" if confidence is not None else ""

        if result == 1:
            st.markdown(f"""
            <div class="result-card result-spam">
                <div class="result-icon">🚨</div>
                <div class="result-heading">Spam Detected</div>
                <div class="result-sub">{conf_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if confidence is not None:
                st.progress(min(int(confidence), 100))
        else:
            st.markdown(f"""
            <div class="result-card result-safe">
                <div class="result-icon">✅</div>
                <div class="result-heading">Not Spam</div>
                <div class="result-sub">{conf_text}</div>
            </div>
            """, unsafe_allow_html=True)
            if confidence is not None:
                st.progress(min(int(confidence), 100))
            st.balloons()

        if tokens:
            st.markdown("<div style='margin-top:1rem;font-size:0.85rem;color:#9fb3d8;'>Key processed tokens:</div>", unsafe_allow_html=True)
            chips_html = "".join(f"<span class='token-chip'>{t}</span>" for t in tokens[:20])
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

        with st.expander("🔧 See cleaned text sent to the model"):
            st.code(transformed_sms if transformed_sms else "(empty after cleaning)")

st.markdown(
    "<div class='footer-note'>Built with Streamlit · TF-IDF + ML Classifier · Spam Shield</div>",
    unsafe_allow_html=True,
)