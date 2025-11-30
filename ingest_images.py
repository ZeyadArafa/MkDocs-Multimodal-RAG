import os
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- CONFIGURATION ---
DOCS_PATH = "./mkdocs/docs"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "mkdocs_images"

def ingest_images():
    print("📷 Starting Image Ingestion...")
    
    # 1. Initialize the CLIP model (It understands images AND text)
    # This might download about 600MB the first time
    model = SentenceTransformer('clip-ViT-B-32')
    
    # 2. Connect to the existing ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete old image collection if it exists (to start fresh)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old image collection.")
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    
    # 3. Find all images
    image_paths = []
    image_ids = []
    
    # Walk through folders to find .png and .jpg
    for root, dirs, files in os.walk(DOCS_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Create full path
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                # Use filename as ID
                image_ids.append(file)
    
    print(f"Found {len(image_paths)} images.")
    
    if not image_paths:
        print("⚠️ No images found! Check your DOCS_PATH.")
        return

    # 4. Embed and Store
    print("Embedding images (this may take a moment)...")
    
    # CLIP can encode images directly into vectors
    embeddings = model.encode(
        [Image.open(p) for p in image_paths],
        batch_size=32,
        convert_to_tensor=False,
        show_progress_bar=True
    )
    
    # Add to Database
    # We store the 'path' in metadata so we can display it later
    metadatas = [{"path": p} for p in image_paths]
    
    collection.add(
        embeddings=embeddings,
        ids=image_ids,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully stored {len(image_paths)} images in '{COLLECTION_NAME}'")

if __name__ == "__main__":
    ingest_images()