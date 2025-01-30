import os
import PyPDF2
import pinecone
import numpy as np
import pickle
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from google.colab import drive
from pinecone import Pinecone
from pinecone import ServerlessSpec # import ServerlessSpec
from sklearn.decomposition import TruncatedSVD 

# Mount Google Drive for saving files
drive.mount('/content/drive')

# Set up Groq client using environment variable
from google.colab import userdata
GROQ_API_KEY=userdata.get('Groq_Api_Key')
if not GROQ_API_KEY:
    raise ValueError("Groq API Key not found in environment variables.")
client = Groq(api_key=GROQ_API_KEY)

# Pinecone setup using environment variable
from google.colab import userdata
PINECONE_API_KEY=userdata.get('pinecone')
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key not found in environment variables.")

PINECONE_ENV = "aped-4627-b74a"  # Updated environment
PINECONE_INDEX_NAME = "uepchatbot-ms3md4p"  # Updated index name

# Initialize Pinecone using Pinecone class and your existing API key and environment
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check if the index exists; connect to it if it does, otherwise create it
try:
    # Attempt to connect to the index
    index = pinecone.Index(PINECONE_INDEX_NAME)
    print(f"Connected to existing index: {PINECONE_INDEX_NAME}")
except pinecone.core.client.exceptions.PineconeApiException as e:
    if "Index does not exist" in str(e):  # Check if the error is due to the index not existing
        # Create index if it does not exist
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Use ServerlessSpec for serverless index
        )
        print(f"Created new index: {PINECONE_INDEX_NAME}")
        index = pinecone.Index(PINECONE_INDEX_NAME)  # Get the index object
    else:
        # Re-raise the exception if it's not related to the index not existing
        raise e

# Get the index object here
index = pinecone.Index(PINECONE_INDEX_NAME) # Define index here

# Paths for storing processed data in Google Drive
DATA_FOLDER = "/content/drive/MyDrive/pdfs"
PROCESSED_DATA_FILE = "/content/drive/MyDrive/processed_data.pkl"

# Google Drive folder links for downloading PDFs
GOOGLE_DRIVE_FOLDER_LINKS = [
    "https://drive.google.com/drive/folders/1e5rpHuCVNxutjrnkWhsw5Dxz6gS_jfCt?usp=sharing",
]

# Function to download and process PDFs
def download_and_process_pdfs():
    if os.path.exists(PROCESSED_DATA_FILE):
        print("Pre-processed data file already exists. Skipping processing.")
        return

    print("Downloading and processing PDFs...")
    pdf_texts = []
    doc_names = []

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    for folder_url in GOOGLE_DRIVE_FOLDER_LINKS:
        try:
            gdown.download_folder(folder_url, quiet=False, output=DATA_FOLDER)
        except Exception as e:
            print(f"Error downloading folder: {e}")
            continue

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            doc_names.append(filename)
            with open(os.path.join(DATA_FOLDER, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages)
                pdf_texts.append(text)
                print(f"Processed: {filename}, Text Length: {len(text)}")

    # Tokenize, chunk, and embed the data
    chunks = []
    for document in pdf_texts:
        chunks.extend(chunk_text(document))

 # Generate embeddings using TF-IDF and reduce dimensionality using TruncatedSVD
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)

    # Reduce dimensionality to 768 using TruncatedSVD
    svd = TruncatedSVD(n_components=768)  # Create TruncatedSVD object with desired dimensions
    X = svd.fit_transform(X)  # Apply dimensionality reduction

   # Upload embeddings to Pinecone
    print("Uploading embeddings to Pinecone...")
    for i, chunk in enumerate(chunks):
        vector = X[i].astype(np.float32).tolist() # Convert to list of floats 
        index.upsert([(str(i), vector, {"text": chunk})]) # Use the index object


    # Save processed data
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "doc_names": doc_names}, f)

    print("Pre-processing completed. Data saved for future use.")

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to retrieve relevant chunks using Pinecone
# Function to retrieve relevant chunks using Pinecone
def retrieve_relevant_chunks(query, vectorizer, svd, k=3): # add svd as an argument
    query_vector = vectorizer.transform([query]).toarray()
    # Apply dimensionality reduction to the query vector
    query_vector = svd.transform(query_vector).astype(np.float32).flatten().tolist() # transform the vector
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]


# Function to generate a response using retrieved chunks
def generate_response(query, context_chunks):
    prompt = " ".join(context_chunks) + "\n\nUser's question: " + query
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Run the pipeline
download_and_process_pdfs()

# Example usage (modify or integrate with Streamlit)
query = input("Enter your question: ")
with open(PROCESSED_DATA_FILE, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
vectorizer = data["vectorizer"]
svd = TruncatedSVD(n_components=768) # Instantiate TruncatedSVD
svd.fit(vectorizer.transform(chunks)) # Fit to chunks

context_chunks = retrieve_relevant_chunks(query, vectorizer, svd) # Pass svd to the function
response = generate_response(query, context_chunks)

print("\nResponse:\n", response)
