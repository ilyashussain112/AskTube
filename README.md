# YouTube Q&A Chatbot with LangChain + Groq + FAISS

This project lets you ask questions about a YouTube video.  
It extracts the transcript, splits it into chunks, stores embeddings in FAISS,  
and answers queries using Groq's LLM.

---

##Features
- Extracts YouTube video transcripts automatically  
- Splits text into chunks for better retrieval  
- Stores embeddings in FAISS vector database  
- Answers user queries using Groq's LLM (`deepseek-r1-distill-llama-70b`)  
- Handles cases where transcript is not available  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ilyashussain112/AskTube.git


2. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows


3. Install dependencies:
    pip install -r requirements.txt

4. Create a .env file and add your Groq API key:
    GROQ_API_KEY=your_api_key_here

Usage:
python main.py


Then:

    Enter a YouTube URL when prompted

    Enter your question about the video


Enter Youtube URL: https://www.youtube.com/watch?v=abcd1234
Enter your Query: What is the main topic of this video?



...Tech Stack...

LangChain
 for orchestration

Groq LLM
 for answering queries

FAISS
 for vector search

Sentence-Transformers
 for embeddings

YouTube Transcript API
 for captions


----Limitations----

Works only if the video has captions enabled

Accuracy depends on transcript quality

Needs internet connection for Groq API



License:
    MIT License
