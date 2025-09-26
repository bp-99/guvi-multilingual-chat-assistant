import gradio as gr
from langdetect import detect
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import faiss
import numpy as np
import json

# Load FAQ data
with open("guvi_qa.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)


# Build embeddings with normalized questions
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def normalize_text(text):
    return text.strip().lower()

faq_questions = [normalize_text(f["Question"]) for f in faq_data]
faq_embeddings = model.encode(faq_questions)

# Build FAISS index
dim = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(faq_embeddings))

# Chat function
def chatbot(user_text, chat_history):

    # Detect language, fallback to English for short/ambiguous input
    try:
        lang = detect(user_text)
    except Exception:
        lang = "en"
    # Fallback for short or ambiguous input
    if len(user_text.strip()) < 5 or lang not in ["en", "hi", "ta", "te", "fr", "es", "de", "zh", "ar", "ru", "ja", "ko"]:
        lang = "en"
    original_user_text = user_text

    # Normalize input
    user_text_norm = normalize_text(user_text)

    # Translate only for embedding search
    if lang != "en":
        user_text_for_search = GoogleTranslator(source=lang, target="en").translate(user_text)
        user_text_for_search = normalize_text(user_text_for_search)
    else:
        user_text_for_search = user_text_norm

    query_embedding = model.encode([user_text_for_search])
    D, I = index.search(np.array(query_embedding), k=3)

    # Show top-1 if score is high, else show top-3
    answers = []
    threshold = 50.0  # Lower is more similar (L2 distance)
    for rank in range(3):
        idx = I[0][rank]
        dist = D[0][rank]
        faq_item = faq_data[idx]
        # Do not modify FAQ data
        if lang != "en":
            answer = GoogleTranslator(source="en", target=lang).translate(faq_item["Answer"])
        else:
            answer = faq_item["Answer"]
        answers.append((dist, faq_item["Question"], answer))

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": original_user_text})

    if D[0][0] < threshold:
        # Only show the best answer
        chat_history.append({"role": "assistant", "content": f"[{lang}] {answers[0][2]}"})
    else:
        # Show top-3 answers with their questions
        response = f"[{lang}] I found multiple possible answers:\n"
        for i, (dist, q, a) in enumerate(answers, 1):
            response += f"\n{i}. Q: {q}\nA: {a}\n"
        chat_history.append({"role": "assistant", "content": response})
    return chat_history, chat_history

# Gradio interface
with gr.Blocks() as demo:
    # ✅ Add title here
    gr.Markdown("<h1 style='text-align: center; color: green;'>Guvi Multilingual Chat Assistant 🤖</h1>")

    chatbot_ui = gr.Chatbot(type="messages")

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything...", lines=2, scale=8)
        send = gr.Button("Send", scale=1)

    clear = gr.Button("Clear")

    def respond(message, chat_history):
        return chatbot(message, chat_history)

    # Press Enter OR click Send → same behavior
    msg.submit(respond, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    send.click(respond, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])

    clear.click(lambda: [], None, chatbot_ui, queue=False)

if __name__ == "__main__":
    demo.launch()
