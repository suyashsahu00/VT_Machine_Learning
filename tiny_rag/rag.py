import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# =====================================================================
# 1. DOCUMENT LOADER & CHUNKER
# =====================================================================
class DocumentLoader:
    """
    Handles reading local text files and splitting them into smaller chunks.
    
    In industrial RAG systems:
    - This corresponds to LangChain's 'Document Loaders' and 'Text Splitters'.
    - You would use libraries like PyPDF, Unstructured, or BeautifulSoup.
    - Advanced chunking might use semantic splitting (splitting by sentence boundaries).
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target length of each chunk in characters.
            chunk_overlap: Number of overlapping characters between consecutive chunks
                           to ensure context isn't lost at boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, filepath: str) -> str:
        """Reads a text file and returns its content as a single string."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def split_text(self, text: str) -> list[str]:
        """
        Splits text into chunks using a sliding window.
        
        Example: 
        text = "Hello world of machine learning"
        chunk_size=15, chunk_overlap=5
        chunk 1: "Hello world of "
        chunk 2: "rld of machine "
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Slide the window forward by chunk_size - chunk_overlap
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks


# =====================================================================
# 2. LOCAL VECTOR STORE & EMBEDDER
# =====================================================================
class SimpleVectorStore:
    """
    A simple in-memory vector store that stores embeddings and performs
    Cosine Similarity search.
    
    In industrial RAG systems:
    - This corresponds to databases like Pinecone, ChromaDB, Milvus, or pgvector.
    - Instead of keeping vectors in RAM and doing a manual numpy search, 
      you would index them in a DB using algorithms like HNSW for sub-millisecond search.
    """
    def __init__(self, client: genai.Client):
        self.client = client
        self.chunks = []
        self.embeddings = []  # Will store list of lists/numpy arrays

    def add_documents(self, chunks: list[str]):
        """Generates embeddings for chunks and stores them."""
        if not chunks:
            return
            
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
        
        # Batch embed for efficiency by wrapping each chunk in a Content object
        contents = [types.Content(parts=[types.Part.from_text(text=chunk)]) for chunk in chunks]
        response = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents
        )
        
        # Extract embeddings and save them
        for i, chunk in enumerate(chunks):
            # Each embedding contains a list of floats (vector values)
            vector = response.embeddings[i].values
            self.chunks.append(chunk)
            self.embeddings.append(vector)
            
        print("[INFO] Embeddings generated and stored successfully.")

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        """
        Computes cosine similarity between query embedding and stored vectors.
        Returns top_k most similar chunks and their scores.
        """
        if not self.embeddings:
            print("[WARNING] Vector store is empty.")
            return []

        # 1. Get embedding for the user's query
        query_response = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=query
        )
        query_vector = np.array(query_response.embeddings[0].values)

        # 2. Convert stored embeddings to numpy array for fast math
        stored_vectors = np.array(self.embeddings)

        # 3. Calculate Cosine Similarity: Dot product of normalized vectors
        # Formula: A . B / (||A|| * ||B||)
        
        # Normalize the query vector to unit length
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # Normalize each stored vector along the columns
        stored_norms = stored_vectors / np.linalg.norm(stored_vectors, axis=1, keepdims=True)
        
        # Calculate dot product
        similarities = np.dot(stored_norms, query_norm)

        # 4. Get the indices of the top_k elements sorted in descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
            
        return results


# =====================================================================
# 3. RAG PIPELINE
# =====================================================================
class RAGPipeline:
    """
    Coordinates the Loader, Vector Store, Prompt Construction, and LLM call.
    
    In industrial RAG systems:
    - This corresponds to Orchestration frameworks like LangChain, LlamaIndex, or Haystack.
    """
    def __init__(self, client: genai.Client, chunk_size: int = 500, chunk_overlap: int = 100):
        self.client = client
        self.loader = DocumentLoader(chunk_size, chunk_overlap)
        self.vector_store = SimpleVectorStore(client)

    def ingest_document(self, filepath: str):
        """Loads and indexes a document in the vector store."""
        print(f"[INFO] Ingesting document: {filepath}")
        raw_text = self.loader.load(filepath)
        chunks = self.loader.split_text(raw_text)
        print(f"[INFO] Document split into {len(chunks)} chunks.")
        self.vector_store.add_documents(chunks)

    def answer_question(self, question: str, top_k: int = 2) -> dict:
        """
        Retrieves context, formats prompt, calls Gemini LLM, and returns the response.
        """
        print(f"[INFO] Searching vector store for: '{question}'")
        
        # 1. Retrieve the top_k context chunks
        retrieved_results = self.vector_store.search(question, top_k=top_k)
        
        # Combine retrieved chunks into a single context string
        context_parts = []
        for chunk, score in retrieved_results:
            context_parts.append(f"Context Chunk (Similarity Score: {score:.4f}):\n{chunk}\n")
        
        context_text = "\n---\n".join(context_parts)
        
        # 2. Build the structured Prompt (System prompt + Context + Question)
        prompt = f"""You are a helpful study assistant. Answer the User Question using only the provided Search Context.
If the Search Context does not contain the answer, say "I couldn't find the answer in the provided notes."
Do not make up information. Use a concise and structured tone.

---
Search Context:
{context_text}
---

User Question: {question}

Answer:"""

        print("[INFO] Calling Gemini LLM (gemini-2.5-flash)...")
        
        # 3. Ask Gemini to generate content
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        return {
            "answer": response.text,
            "sources": retrieved_results
        }


# =====================================================================
# 4. MAIN INTERACTION LOOP
# =====================================================================
def get_api_key() -> str:
    """Helper to fetch API key from env, falling back to user input if missing."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    # If the key is not set, or is still the template placeholder, prompt user
    if not api_key or api_key.strip() == "your_gemini_api_key_here":
        print("\n[!] Gemini API Key not detected in environment or .env file.")
        api_key = input("Please paste your GEMINI_API_KEY here: ").strip()
        if not api_key:
            print("[ERROR] API Key is required. Exiting.")
            exit(1)
            
    return api_key

def main():
    print("=" * 60)
    print("            TINY GENAI RAG SYSTEM (GEMINI EDITION)")
    print("=" * 60)

    # 1. Initialize Gemini Client
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    # 2. Define document path (use our newly created ml_notes.md as default)
    doc_path = "ml_notes.md"
    
    if not os.path.exists(doc_path):
        print(f"[ERROR] Sample file {doc_path} not found. Please verify directory.")
        return

    # 3. Initialize RAG pipeline and ingest the sample document
    # Using small chunks (350 chars) and 50 chars overlap for this short file
    rag = RAGPipeline(client, chunk_size=350, chunk_overlap=50)
    rag.ingest_document(doc_path)
    
    print("\n[INFO] RAG System ready! You can now ask questions about your ML notes.")
    print("Type 'exit' or 'quit' to stop.\n")

    # 4. Interactive loop
    while True:
        try:
            question = input("\nAsk a question: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                print("Exiting RAG system. Happy learning!")
                break
                
            result = rag.answer_question(question, top_k=2)
            
            print("\n" + "=" * 50)
            print("ANSWER:")
            print("=" * 50)
            print(result["answer"])
            print("=" * 50)
            
            print("\nSOURCES RETRIEVED:")
            for idx, (chunk, score) in enumerate(result["sources"], 1):
                # Clean up newlines for prettier source printing
                preview = chunk.replace('\n', ' ')[:100] + "..."
                print(f"{idx}. [Score: {score:.4f}] {preview}")
            print("=" * 50)

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
