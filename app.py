import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
import gdown

from groq import Groq
from sentence_transformers import SentenceTransformer

# ===============================
# CONFIG
# ===============================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Set in HF Secrets
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in Hugging Face Secrets.")

# Google Drive file ID for CSV
FILE_ID = "1jiFnxMOghMQDxDEHHXmLiI5jTvTCicbU"
LOCAL_CSV = "knowledge.csv"

TOP_K = 5
MAX_DISTANCE = 1.2  # Adjust for MiniLM + L2

# ===============================
# Load Models
# ===============================
client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Load Knowledge Base
# ===============================
if not os.path.exists(LOCAL_CSV):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        LOCAL_CSV,
        quiet=True,
        fuzzy=True
    )

df = pd.read_csv(LOCAL_CSV)
documents = [" | ".join([str(v) for v in row.values]) for _, row in df.iterrows()]

# ===============================
# Build FAISS Index
# ===============================
embeddings = embed_model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)

print(f"✅ Knowledge Base Loaded: {len(documents)} rows")

# ===============================
# RAG Question Answering
# ===============================
def ask_question(question):
    query_emb = embed_model.encode([question]).astype("float32")
    distances, indices = faiss_index.search(query_emb, TOP_K)

    if distances[0][0] > MAX_DISTANCE:
        return (
            "<div class='answer-box'>"
            "<h2>Answer</h2>"
            "<p><b>I don’t know.</b><br>"
            "This question is outside my knowledge base.</p>"
            "</div>"
        )

    context = "\n".join([documents[i] for i in indices[0]])

    prompt = f"""
You are a factual assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return (
        "<div class='answer-box'>"
        "<h2>Answer</h2>"
        f"<p>{answer}</p>"
        "</div>"
    )

# ===============================
# Gradio UI
# ===============================
css = """
body {
  background-color: #0f172a;
  color: #f9fafb;
  font-family: Inter, system-ui, sans-serif;
}

.gradio-container {
  max-width: 900px;
  margin: auto;
}

h1 {
  font-size: 28px;
  font-weight: 600;
  text-align: center;
  margin-bottom: 8px;
}

.subtitle {
  text-align: center;
  color: #9ca3af;
  margin-bottom: 32px;
}

input, textarea {
  background-color: #020617 !important;
  color: #f9fafb !important;
  border: 1px solid #1f2937 !important;
  border-radius: 8px !important;
}

label {
  font-weight: 500;
  color: #e5e7eb;
}

.answer-box {
  background-color: #020617;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 18px;
  margin-top: 12px;
}

.answer-box h2 {
  margin-top: 0;
  color: #3b82f6;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # 🧠 Knowledge Base Assistant
        <div class="subtitle">
        Ask questions strictly from the internal knowledge base.
        </div>
        """
    )

    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask a question related to the knowledge base…",
        lines=2
    )

    answer = gr.Markdown(
        value="""
        <div class="answer-box">
        <h2>Answer</h2>
        <p>Your answer will appear here.</p>
        </div>
        """
    )

    question.submit(fn=ask_question, inputs=question, outputs=answer)

demo.launch()
