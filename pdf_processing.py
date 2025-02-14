import os
import uuid
from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Initialize ChromaDB client and collection
client = chromadb.Client()
collection_name = "knowledge_base"
collection = client.get_or_create_collection(name=collection_name)

# Initialize the embedding model from sentence-transformers (all-MiniLM-L6-v2)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embeddings(text):
    """Generate embeddings for the given text using the all-MiniLM-L6-v2 model."""
    try:
        embeddings = embedding_model.encode(text)
        return embeddings.tolist()  # Ensure embeddings are converted to list
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        return None

def process_pdf(file_path):
    """Extract text from a PDF file, generate embeddings, and add it to the ChromaDB vector store."""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip() == "":
            raise ValueError("No text extracted from the PDF.")

        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Generate embeddings for the extracted text
        embeddings = generate_embeddings(text)

        if embeddings is not None:
            # Add extracted text and its embeddings to ChromaDB
            collection.add(
                documents=[text],
                embeddings=[embeddings],  # Store the embeddings
                ids=[doc_id]
            )
            print(f"Processed and added {file_path} to the knowledge base with ID {doc_id}.")
        else:
            print(f"Failed to process {file_path}: Embeddings generation failed.")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def update_knowledge_base(directory):
    """Process all PDF files in the given directory and update the ChromaDB vector store."""
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                process_pdf(file_path)
    except Exception as e:
        print(f"Failed to update knowledge base: {e}")
