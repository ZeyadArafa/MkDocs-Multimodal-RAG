import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer 

# --- CONFIGURATION ---
GOOGLE_API_KEY = "Your API Key Here"

# Setup Environment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
DB_PATH = "./chroma_db"

def load_rag_resources():
    print("--- Loading Resources ---")
    print("1. Text Embeddings...")
    text_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("2. Vector DB...")
    text_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=text_embedding,
        collection_name="mkdocs_collection"
    )
    
    print("3. Gemini (1.5 Pro)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    print("4. CLIP Image Model...")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        image_collection = chroma_client.get_collection("mkdocs_images")
        print("✅ Image Collection Found.")
    except:
        image_collection = None
        print("⚠️ Warning: Image collection NOT found.")
    
    return {
        "text_db": text_db,
        "llm": llm,
        "clip_model": clip_model,
        "image_collection": image_collection
    }

def ask_gemini(question, resources):
    best_image = None
    
    text_db = resources["text_db"]
    llm = resources["llm"]
    clip_model = resources["clip_model"]
    image_collection = resources["image_collection"]
    
    # --- TEXT RAG ---
    retriever = text_db.as_retriever(search_kwargs={"k": 15})
    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # --- IMAGE RAG (Fixed for your Scale) ---
    if image_collection:
        try:
            print(f"\n🔍 Searching images for: '{question}'")
            query_embedding = clip_model.encode(question).tolist()
            
            image_results = image_collection.query(
                query_embeddings=[query_embedding],
                n_results=1 
            )
            
            if image_results['ids']:
                distance = image_results['distances'][0][0]
                found_path = image_results['metadatas'][0][0]['path']
                print(f"   -> Best Match: {os.path.basename(found_path)}")
                print(f"   -> Distance Score: {distance}")
                
                if distance < 200.0: 
                    best_image = found_path
                    print("   ✅ MATCH ACCEPTED")
                else:
                    best_image = None 
                    print("   ❌ MATCH REJECTED (Distance too high)")
            else:
                print("   ⚠️ No images in DB")
                
        except Exception as e:
            print(f"   Image search error: {e}")

    # --- GENERATE ANSWER ---
    template = """
    You are a friendly and expert technical assistant for MkDocs. 
    
    Guidelines:
    1. Answer the question using ONLY the provided Context.
    2. If the answer requires code, strictly use the code examples from the context.
    3. If the context contains multiple related pieces of info, synthesize them.
    4. Be concise and professional.
    
    Context:
    {context}
    
    Question: 
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    # --- TEXT CLEANER ---
    final_answer = response.content
    if isinstance(final_answer, list):
        final_answer = " ".join(final_answer)
    
    return final_answer, docs, best_image