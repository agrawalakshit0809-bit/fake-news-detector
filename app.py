import streamlit as st
import pickle
import re

# Page Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

# Load Models
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction Function
def predict_news(news_text):
    if not news_text.strip():
        return None, None, None
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    if prediction == 0:
        confidence = round(probability[0] * 100, 2)
        return "FAKE", confidence, "🔴"
    else:
        confidence = round(probability[1] * 100, 2)
        return "REAL", confidence, "🟢"

# Custom CSS for UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f172a; }
.stApp { background-color: #0f172a; }
h1 { color: #ffffff !important; }
p { color: #94a3b8; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 24px; }
.badge { background: #1e293b; border: 1px solid #2d3f55; border-radius: 20px; padding: 5px 12px; font-size: 12px; }
.result-box { background: #1e293b; border: 1px solid #2d3f55; border-radius: 14px; padding: 24px; margin-top: 16px; }
.stTextArea textarea { background-color: #1e293b !important; color: #f1f5f9 !important; border: 1.5px solid #2d3f55 !important; border-radius: 10px !important; }
.stButton > button { background-color: #6366f1 !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; padding: 12px !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <h1 style="font-size:2.4em; font-weight:700; margin-bottom:8px;">🔍 Fake News Detector</h1>
    <p style="color:#94a3b8; margin-bottom:16px;">AI-powered · 44,898 articles trained · 99% Accuracy</p>
    <div class="badge-row">
        <span class="badge" style="color:#a5b4fc;">🤖 NLP Powered</span>
        <span class="badge" style="color:#86efac;">✅ 99% Accuracy</span>
        <span class="badge" style="color:#fbbf24;">📰 44,898 Articles</span>
        <span class="badge" style="color:#f9a8d4;">⚡ Instant Results</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Layout: Input and Analysis
col1, col2 = st.columns(2)

with col1:
    news_input = st.text_area(
        "Paste News Here",
        placeholder="Type or paste any news headline or article...",
        height=200,
        key="news_input"
    )
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("Clear"):
            st.session_state.news_input = ""
            st.rerun()
    with bcol2:
        analyze = st.button("🔍 Analyze News", type="primary")

with col2:
    if analyze and news_input:
        label, confidence, emoji = predict_news(news_input)
        if label == "FAKE":
            st.markdown(f"""
            <div class="result-box" style="border-color:#ef4444;">
                <h2 style="color:#ef4444;">{emoji} FAKE NEWS DETECTED</h2>
                <p style="color:#94a3b8; font-size:1.1em;">Confidence: <strong style="color:#f1f5f9;">{confidence}%</strong></p>
                <p style="color:#94a3b8;">This article shows strong signs of being FAKE or misleading.</p>
                <p style="color:#64748b; font-size:0.85em;">Always verify with trusted sources like Reuters, BBC, or AP News.</p>
            </div>
            """, unsafe_allow_html=True)
        elif label == "REAL":
            st.markdown(f"""
            <div class="result-box" style="border-color:#22c55e;">
                <h2 style="color:#22c55e;">{emoji} REAL NEWS VERIFIED</h2>
                <p style="color:#94a3b8; font-size:1.1em;">Confidence: <strong style="color:#f1f5f9;">{confidence}%</strong></p>
                <p style="color:#94a3b8;">This article matches patterns of legitimate, verified journalism.</p>
                <p style="color:#64748b; font-size:0.85em;">Content aligns with real news language and reporting standards.</p>
            </div>
            """, unsafe_allow_html=True)
    elif analyze and not news_input:
        st.warning("⚠️ Please enter some news text to analyze!")
    else:
        st.markdown("""
        <div class="result-box">
            <p style="color:#475569; text-align:center; margin-top:40px;">
                AI Analysis Result will appear here...
            </p>
        </div>
        """, unsafe_allow_html=True)

# Example Section
st.markdown("""
<p style="color:#94a3b8; font-size:0.78em; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin:20px 0 12px;">
    📋 Click Any Example to Test
</p>
""", unsafe_allow_html=True)

examples = [
    "BREAKING: You won't believe what Obama just did! WATCH NOW before deleted",
    "Republican senator says he will vote for new healthcare bill",
    "Hillary Clinton SECRET video EXPOSED — Share before it gets removed!",
    "Trump signs executive order on immigration at the White House",
    "VIDEO: Watch what happens when reporter confronts Obama on live TV",
    "US military to accept transgender recruits after federal court ruling"
]

ecol1, ecol2 = st.columns(2)
for i, example in enumerate(examples):
    col = ecol1 if i % 2 == 0 else ecol2
    with col:
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.news_input = example
            st.rerun()
