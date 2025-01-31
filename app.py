import os
import PyPDF2
import faiss
import numpy as np
import pickle
import gdown
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# Set up Groq client
client = Groq(api_key=os.environ.get("Groq_Api_Key"))

# Paths for storing processed data and conversation history
DATA_FOLDER = "./pdfs"
PROCESSED_DATA_FILE = "./processed_data.pkl"
CONVERSATION_HISTORY_FILE = "./conversations.pkl"

# Google Drive folder link
GOOGLE_DRIVE_FOLDER_LINKS = [
    "https://drive.google.com/drive/folders/1e5rpHuCVNxutjrnkWhsw5Dxz6gS_jfCt?usp=sharing",
]

# Function to download and process PDFs
def download_and_process_pdfs():
    if os.path.exists(PROCESSED_DATA_FILE):
        st.success("Pre-processed data file already exists. Skipping processing.")
        return

    st.info("Downloading and processing PDFs...")
    pdf_texts = []
    doc_names = []

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    for folder_url in GOOGLE_DRIVE_FOLDER_LINKS:
        try:
            gdown.download_folder(folder_url, quiet=False, output=DATA_FOLDER)
        except Exception as e:
            st.error(f"Error downloading folder: {e}")
            continue

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            doc_names.append(filename)
            with open(os.path.join(DATA_FOLDER, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                pdf_texts.append(text)
                st.write(f"Processed: {filename}, Text Length: {len(text)}")

    chunks = []
    for document in pdf_texts:
        chunks.extend(chunk_text(document))

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)

    faiss_index = faiss.IndexFlatL2(X.shape[1])
    faiss_index.add(X.toarray().astype(np.float32))

    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "faiss_index": faiss_index, "doc_names": doc_names}, f)

    st.success("Pre-processing completed. Data saved.")

# Function to chunk text
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, faiss_index, chunks, vectorizer, k=3):
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = faiss_index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# Function for agentic RAG response
def generate_response(query, context_chunks):
    prompt = " ".join(context_chunks) + "\n\nUser's question: " + query
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Load pre-processed data
def load_data():
    if not os.path.exists(PROCESSED_DATA_FILE):
        st.error("Pre-processed data file is missing. Please run 'Download and Process'.")
        return None

    with open(PROCESSED_DATA_FILE, "rb") as f:
        return pickle.load(f)

# Load conversation history
def load_conversation_history():
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

# Save conversation history
def save_conversation_history(history):
    with open(CONVERSATION_HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)

# Streamlit UI
def main():
    st.set_page_config(page_title="UEP Procedures Chatbot", layout="wide")
    st.title("Agentic Chatbot for UEP Procedures")

    # Sidebar for document names
    st.sidebar.title("Processed Documents")
    data = load_data()
    if data:
        doc_names = data["doc_names"]
        st.sidebar.write("\n".join(doc_names))

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Download and Process"):
            download_and_process_pdfs()

    st.subheader("Ask Your Question")
    user_query = st.text_input("Enter your question:")

    # Load conversation history
    conversation_history = load_conversation_history()

    if st.button("Submit"):
        if user_query:
            if data:
                chunks = data["chunks"]
                vectorizer = data["vectorizer"]
                faiss_index = data["faiss_index"]

                context_chunks = retrieve_relevant_chunks(user_query, faiss_index, chunks, vectorizer)
                response = generate_response(user_query, context_chunks)
                
                # Display response
                st.write(f"**Response:** {response}")

                # Save conversation to history
                conversation_history.append({"question": user_query, "response": response})
                save_conversation_history(conversation_history)
        else:
            st.warning("Please enter a question.")

    # Show conversation history
    st.subheader("Conversation History")
    if conversation_history:
        for entry in conversation_history:
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Response:** {entry['response']}")
            st.write("---")
    else:
        st.write("No conversation history yet.")

if __name__ == "__main__":
    main()
