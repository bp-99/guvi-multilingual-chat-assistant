# Guvi Multilingual Chat Assistant 🤖

A multilingual chatbot built with Gradio, Sentence Transformers, FAISS, and Deep Translator. It answers user questions based on a FAQ dataset, supporting multiple languages and returning the most relevant answers.

## Features
- Multilingual support: Ask questions in various languages
- Fast semantic search using FAISS
- Accurate answer retrieval using sentence embeddings
- Automatic translation of user queries and answers
- Gradio web interface for easy interaction

## How It Works
1. User enters a question in any supported language.
2. The question is translated to English (if needed) for semantic search.
3. The chatbot finds the most relevant FAQ answers using embeddings and FAISS.
4. The answer is translated back to the user's language (if needed).
5. The chat history is displayed in the interface.

## Setup
1. Clone this repository or copy the files to your local machine.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the following files in the project directory:
   - `app.py` (main chatbot code)
   - `guvi_qa.json` (FAQ data)
   - `guvi_dataset_formatted.txt` (optional, for reference)
4. Run the chatbot:
   ```bash
   python app.py
   ```
5. Access the Gradio interface in your browser (URL will be shown in the terminal).

## File Structure
- `app.py` — Main application code
- `guvi_qa.json` — FAQ dataset in JSON format
- `guvi_dataset_formatted.txt` — Optional formatted dataset
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.7+
- See `requirements.txt` for all required packages

## Customization
- Update `guvi_qa.json` to add or modify FAQ entries.
- Change the Gradio UI in `app.py` for a different look or features.

## License
This project is for educational purposes. Please check individual package licenses for commercial use.

Huggingface spaces : https://huggingface.co/spaces/bp-99/guvi-multilingual-chat-assistant
