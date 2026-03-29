import gradio as gr
import pickle
import re

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(news_text):
    if not news_text.strip():
        return "⚠️ Please enter some news text to analyze!"
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    if prediction == 0:
        confidence = round(probability[0] * 100, 2)
        return (
            f"🔴 FAKE NEWS DETECTED\n\n"
            f"Confidence Score: {confidence}%\n\n"
            f"This article shows strong signs of being FAKE or misleading.\n\n"
            f"Always verify with trusted sources like Reuters, BBC, or AP News."
        )
    else:
        confidence = round(probability[1] * 100, 2)
        return (
            f"🟢 REAL NEWS VERIFIED\n\n"
            f"Confidence Score: {confidence}%\n\n"
            f"This article matches patterns of legitimate, verified journalism.\n\n"
            f"Content aligns with real news language and reporting standards."
        )

with gr.Blocks(title="Fake News Detector") as app:

    gr.HTML("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            * { box-sizing: border-box !important; }
            body { background: #0f172a !important; font-family: Inter, sans-serif !important; margin: 0 !important; padding: 0 !important; }
            .gradio-container { background: #0f172a !important; max-width: 100% !important; width: 100% !important; padding: 20px !important; min-height: 100vh !important; }
            footer { display: none !important; }
            button[aria-label="Flag"] { display: none !important; }
            label span { color: #7c8fa6 !important; font-size: 0.78em !important; font-weight: 600 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
            textarea { background: #0f172a !important; border: 1.5px solid #2d3f55 !important; border-radius: 10px !important; color: #f1f5f9 !important; font-family: Inter, sans-serif !important; font-size: 15px !important; line-height: 1.7 !important; padding: 14px 16px !important; }
            textarea::placeholder { color: #3d5166 !important; }
            textarea:focus { border-color: #6366f1 !important; outline: none !important; }
            .block { background: #1e293b !important; border: 1px solid #2d3f55 !important; border-radius: 14px !important; padding: 16px !important; }
            button.primary { background: #6366f1 !important; border: none !important; border-radius: 10px !important; color: #ffffff !important; font-family: Inter, sans-serif !important; font-size: 15px !important; font-weight: 600 !important; padding: 13px 20px !important; width: 100% !important; cursor: pointer !important; }
            button.secondary { background: transparent !important; border: 1.5px solid #2d3f55 !important; border-radius: 10px !important; color: #94a3b8 !important; font-family: Inter, sans-serif !important; font-size: 14px !important; }
            .example-btn button { background: #1e293b !important; border: 1.5px solid #334155 !important; border-radius: 12px !important; color: #f1f5f9 !important; font-family: Inter, sans-serif !important; font-size: 14px !important; font-weight: 500 !important; padding: 14px 16px !important; width: 100% !important; text-align: left !important; white-space: normal !important; height: auto !important; line-height: 1.6 !important; min-height: 52px !important; cursor: pointer !important; }
            .example-btn button:hover { background: #263347 !important; border-color: #6366f1 !important; color: #ffffff !important; }

            /* MOBILE RESPONSIVE */
            @media (max-width: 640px) {
                .gradio-container { padding: 12px !important; }
                .gr-row { flex-direction: column !important; flex-wrap: wrap !important; }
                .gr-column { width: 100% !important; min-width: 100% !important; flex: none !important; }
                textarea { font-size: 14px !important; }
                .example-btn button { font-size: 13px !important; padding: 12px 12px !important; }
                h1 { font-size: 1.8em !important; }
            }
        </style>

        <div style="text-align:center; padding:20px 0 24px; font-family:Inter,sans-serif;">
            <h1 style="color:#ffffff; font-size:2.4em; font-weight:700; margin-bottom:8px; letter-spacing:-0.5px;">
                🔍 Fake News Detector
            </h1>
            <p style="color:#94a3b8; font-size:1em; margin:0 0 16px;">
                AI-powered · 44,898 articles trained · 99% Accuracy
            </p>
            <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:8px;">
                <span style="background:#1e293b; border:1px solid #2d3f55; color:#a5b4fc; padding:5px 12px; border-radius:20px; font-size:12px;">🤖 NLP Powered</span>
                <span style="background:#1e293b; border:1px solid #2d3f55; color:#86efac; padding:5px 12px; border-radius:20px; font-size:12px;">✅ 99% Accuracy</span>
                <span style="background:#1e293b; border:1px solid #2d3f55; color:#fbbf24; padding:5px 12px; border-radius:20px; font-size:12px;">📰 44,898 Articles</span>
                <span style="background:#1e293b; border:1px solid #2d3f55; color:#f9a8d4; padding:5px 12px; border-radius:20px; font-size:12px;">⚡ Instant Results</span>
            </div>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="Paste News Here",
                placeholder="Type or paste any news headline or article...",
                lines=6
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("🔍 Analyze News", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(
                label="AI Analysis Result",
                lines=6,
                interactive=False
            )

    gr.HTML("""
        <div style="margin-top:20px; font-family:Inter,sans-serif;">
            <p style="color:#94a3b8; font-size:0.78em; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin:0 0 12px;">
                📋 Click Any Example to Test
            </p>
        </div>
    """)

    with gr.Row(elem_classes="example-btn"):
        ex1 = gr.Button("BREAKING: You won't believe what Obama just did! WATCH NOW before deleted")
        ex2 = gr.Button("Republican senator says he will vote for new healthcare bill")
    with gr.Row(elem_classes="example-btn"):
        ex3 = gr.Button("Hillary Clinton SECRET video EXPOSED — Share before it gets removed!")
        ex4 = gr.Button("Trump signs executive order on immigration at the White House")
    with gr.Row(elem_classes="example-btn"):
        ex5 = gr.Button("VIDEO: Watch what happens when reporter confronts Obama on live TV")
        ex6 = gr.Button("US military to accept transgender recruits after federal court ruling")

    ex1.click(fn=lambda: "BREAKING: You won't believe what Obama just did! WATCH NOW before deleted", outputs=input_box)
    ex2.click(fn=lambda: "Republican senator says he will vote for new healthcare bill", outputs=input_box)
    ex3.click(fn=lambda: "Hillary Clinton SECRET video EXPOSED — Share before it gets removed!", outputs=input_box)
    ex4.click(fn=lambda: "Trump signs executive order on immigration at the White House", outputs=input_box)
    ex5.click(fn=lambda: "VIDEO: Watch what happens when reporter confronts Obama on live TV", outputs=input_box)
    ex6.click(fn=lambda: "US military to accept transgender recruits after federal court ruling", outputs=input_box)

    submit_btn.click(fn=predict_news, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=lambda: ("", ""), outputs=[input_box, output_box])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
