import json
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr
import os
from huggingface_hub import login

hf_token = os.environ.get("hf_token")

if hf_token:
    login(token=hf_token)

with open("guvi_qa.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

def normalize_text(text):
    return text.strip().lower()

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
faq_questions = [normalize_text(f["Question"]) for f in faq_data]
faq_embeddings = model.encode(faq_questions)
dim = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(faq_embeddings, dtype=np.float32))

# Generator
gen_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def rag_chatbot(user_text, chat_history):
    try:
        lang = detect(user_text)
    except Exception:
        lang = "en"

    original_user_text = user_text
    user_text_norm = normalize_text(user_text)

    if lang != "en":
        user_text_for_search = GoogleTranslator(source=lang, target="en").translate(user_text)
        user_text_for_search = normalize_text(user_text_for_search)
    else:
        user_text_for_search = user_text_norm

    query_embedding = model.encode([user_text_for_search])
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_embedding, k=3)

    context = "\n".join([faq_data[I[0][j]]["Answer"] for j in range(len(I[0]))])

    prompt = f"Context: {context}\nQuestion: {user_text_for_search}\nAnswer:"
    gen_output = gen_pipe(prompt, max_new_tokens=100)[0]["generated_text"]

    answer = gen_output.split("Answer:")[-1].strip()

    if lang != "en":
        answer = GoogleTranslator(source="en", target=lang).translate(answer)

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": original_user_text})
    chat_history.append({"role": "assistant", "content": f"[{lang}] {answer}"})
    return chat_history

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; color: green;'>Guvi Multilingual Chat Assistant</h1>")
    chatbot_ui = gr.Chatbot(type="messages")
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything...", lines=2, scale=8)
        send = gr.Button("Send", scale=1)
    clear = gr.ClearButton(components=[msg, chatbot_ui])

    def respond(message, chat_history):
        return rag_chatbot(message, chat_history)

    msg.submit(respond, [msg, chatbot_ui], chatbot_ui)
    send.click(respond, [msg, chatbot_ui], chatbot_ui)

    clear.click(lambda: [], None, chatbot_ui, queue=False)

if __name__ == "__main__":
    demo.launch()
