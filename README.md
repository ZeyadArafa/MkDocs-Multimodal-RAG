🚀 MkDocs Multimodal RAG

AI-Powered Documentation Assistant with Text + Image Retrieval

This project is a Multimodal RAG (Retrieval Augmented Generation) system
built on top of MKDocs documentation.
It lets you ask natural language questions, and the system will:

✅ Search text pages <br> 
✅ Search images from the documentation using CLIP <br>
✅ Retrieve the most relevant snippet + image <br>
✅ Feed them to Gemini 2.5 Pro <br>
✅ Generate a clean, accurate answer <br>
✅ Display everything in a user-friendly Streamlit UI <br>

------------------------------------------------------------------------

✨ Features

🔍 Multimodal RAG (Text + Images)

-   Text embeddings: sentence-transformers/all-MiniLM-L6-v2
-   Image embeddings: clip-ViT-B-32
-   Vector database: ChromaDB
-   Supports .md, .png, .jpg, .jpeg, .gif

🤖 LLM-Powered Answers

-   Uses Gemini 2.5 Pro for reasoning, summarization, and context-aware
    responses.
-   LLM is restricted to retrieved context → no hallucinations.

🖼️ Image Understanding

-   The assistant can “see” images from your documentation.
-   Displays the most relevant documentation image for each query.

💡 Interactive UI (Streamlit)

-   Full chat interface
-   Image preview inside chat
-   Source context expansion
-   Persistent message history

------------------------------------------------------------------------

🧱 Project Structure

    .
    ├── app.py                 # Streamlit web interface
    ├── rag.py                 # Text + Image retrieval logic
    ├── ingest_images.py       # Script to ingest/encode images
    ├── chroma_db/             # Chroma vector database
    ├── mkdocs/
    │   └── docs/              # Your documentation folder (markdown + images)
    └── README.md

------------------------------------------------------------------------

⚙️ How It Works (Architecture Overview)

1️⃣ Ingest Documentation

-   Text pages go to mkdocs_collection
-   Images go to mkdocs_images
-   CLIP encodes each image → stored as vector + file path metadata

2️⃣ Query Phase

When user asks a question:

Text Retrieval

-   Convert question → embedding
-   Retrieve top 15 relevant text chunks
-   Merge into final context

Image Retrieval

-   Encode question using CLIP
-   Query Chroma for similar image embeddings
-   Distance < 0.35 = accept as relevant

LLM Answer

-   Combine:
    -   question
    -   text context
    -   best-matching image
-   Send to Gemini for answer generation

3️⃣ Display in Streamlit

-   Chat conversation
-   Found relevant image
-   Source snippets

------------------------------------------------------------------------

🛠️ Installation

Make sure you have Python 3.10+.

    git clone https://github.com/your-username/your-repo.git
    cd your-repo

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

📷 Ingest Images (Required First Step)

Make sure your MKDocs images are inside:

    mkdocs/docs/

Then run:

    python ingest_images.py

This script will:

-   Load CLIP
-   Index all images
-   Build/update the mkdocs_images Chroma collection

------------------------------------------------------------------------

▶️ Run the App

    streamlit run app.py

Then open:

    http://localhost:8501

------------------------------------------------------------------------

🔑 Environment Variables

Create .env or export:

    GOOGLE_API_KEY=your_api_key

------------------------------------------------------------------------

📌 Example Queries

Try asking:

    How do I configure dark mode in MkDocs?
    Where are the deployment settings?
    Show me the theme configuration.
    How do I enable navigation tabs?
    Where is the search bar image used?

------------------------------------------------------------------------

📁 Requirements

    streamlit
    chromadb
    sentence-transformers
    langchain
    langchain-huggingface
    langchain-community
    langchain-google-genai
    Pillow
    python-dotenv

------------------------------------------------------------------------

🤝 Contributing

Pull requests are welcome!

------------------------------------------------------------------------

📜 License

MIT License
