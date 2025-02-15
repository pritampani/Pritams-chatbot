
import os
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/lib"
import pysqlite3  # Force using correct SQLite
import sys
sys.modules["sqlite3"] = pysqlite3
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db")

# Ensure the collection exists before using it
collection_name = "knowledge_base"

try:
    collection = client.get_collection(name=collection_name)
except:
    print("Collection not found. Creating a new one...")
    collection = client.create_collection(name=collection_name)

# Initialize the embedding model (all-MiniLM-L6-v2)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_query_embedding(query):
    """Generate embeddings for the query using the all-MiniLM-L6-v2 model."""
    try:
        # Generate the embedding using the sentence-transformers model
        embedding = embedding_model.encode(query)
        return embedding.tolist()  # Convert embedding to list
    except Exception as e:
        print(f"Failed to generate query embedding: {e}")
        return None

def retrieve_context(query):
    """Retrieve context from ChromaDB based on the query."""
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # ✅ Ensure collection exists
        collections = client.list_collections()
        if collection_name not in [col.name for col in collections]:
            print(f"Collection '{collection_name}' does not exist.")
            return "No relevant context found."

        if query_embedding is not None:
            # Query ChromaDB using the embedding
            results = collection.query(
                query_embeddings=[query_embedding],  # The query embedding must be a list of embeddings
                n_results=3  # Number of relevant documents to retrieve
            )

            # Check if documents were returned
            if 'documents' in results and results['documents']:
                # Flatten the list of lists to a single list of strings
                documents = [doc for sublist in results['documents'] for doc in sublist]

                # Combine the results into a single string
                context = " ".join(documents)
                return context

        return "No relevant context found."
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error occurred while retrieving context."
