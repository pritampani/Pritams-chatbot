import streamlit as st
import chromadb
import google.generativeai as genai
import json
import os
import logging
from vector_store import retrieve_context
from pdf_processing import update_knowledge_base
value=st.secrets["value"]
genai.configure(api_key=value)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


genai.configure(api_key=value)

generation_config = {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": 5,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
)

# Function to process and update knowledge base from PDFs
def process_pdfs(directory="uploads"):
    """Check and update the knowledge base from PDFs at startup."""
    if not os.path.exists(directory):
        logging.error(f"PDF directory '{directory}' does not exist.")
        print(f"PDF directory '{directory}' does not exist.")
        return
    
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logging.warning("No PDF files found in the directory.")
        print("No PDF files found in the directory.")
        return
    
    try:
        print(f"Processing PDFs: {pdf_files}")
        update_knowledge_base(directory)
        print("PDFs successfully processed and added to knowledge base.")
    except Exception as e:
        logging.error(f"Error processing PDFs: {e}")
        print(f"Error processing PDFs: {e}")

# Process PDFs at startup
process_pdfs()

def chat(query):
    """Handles the chat function with retrieval-augmented generation (RAG)."""
    try:
        context = retrieve_context(query)
    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        context = None

    if context:
        full_query = f"Context: {context}\nQuery: {query}"
    else:
        full_query = f"Query: {query}. I don't have specific context for this query."

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content([full_query])
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Sorry, I'm unable to assist at the moment."
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}: Error in chat: {e}")
            if attempt == 2:
                logging.error(f"Failed after 3 attempts: {e}")
    return "Sorry, I'm unable to assist at the moment."

def save_conversation(conversation, file_path='predefined_options.json'):
    """Saves user conversations for future reference."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []

        conversations.append(conversation)

        with open(file_path, 'w') as f:
            json.dump(conversations, f, indent=4)

        collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection(name="conversation_context")
        collection.add(
            documents=[conversation['User'] + ": " + conversation['Luna']],
            ids=[str(len(conversations))],
            metadatas=[{"conversation_id": str(len(conversations))}]
        )
    except Exception as e:
        logging.error(f"Failed to save conversation: {e}")

# Streamlit UI
st.title("Pritam's - AI Chatbot")
st.write("Ask me anything!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Type your message here...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    response = chat(query)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
    
    conversation = {"User": query, "Luna": response}
    save_conversation(conversation)

