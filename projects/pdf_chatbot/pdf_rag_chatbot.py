import os
import numpy as np
from pypdf import PdfReader
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. architectural initialization
# initializing embedding model locally that runs entirely
# on machine to convert text chunks into 384 dimensional dense vectors
print("Loading local embedding model...")
embedding_model= SentenceTransformer('all-MiniLM-L6-v2')

# initalize gemini client. it will automatically look for GEMINI API KEy
client= genai.Client()

#2. INGESTION ENGINE (pdf to semantic chunks)
def extract_and_chunk_pdf(pdf_path):
    """
    Reads a PDF file, extracts raw text, and structures it into paragraphs 
    to preserve context integrity.
    """
    print(f"\n📄 Parsing PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""

    # extract text page by page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    # NEW CHUNKING LOGIC STARTS HERE:
    # Clean up massive whitespace gaps and page break artifact lines
    clean_text = " ".join(full_text.split())
    
    # Split the clean text into individual sentences
    sentences = clean_text.split(". ")
    clean_chunks = []
    current_chunk = []

    window= 5 # sentence per chunk
    overlap= 2 # sentence overlap between chunks to preserve context
    
    for i in range(0, len(sentences), window - overlap):
        chunk = sentences[i:i + window]
        if chunk:
            clean_chunks.append(". ".join(chunk) + ".")

    print(f"Extracted {len(clean_chunks)} structured sentence text chunks.")
    return clean_chunks
   

# 3. VECTOR DATABASE SIMULATOR
def build_vector_database(chunks):
    """
    Converts text chunks into dense vector embeddings and stores them in a list.
    In a production system, this would be a vector database like Pinecone or FAISS.
    """
    print("\n Building vector database from text chunks...")

    # generating embeddings for all text blocks simultaneously in parallel matrix operation
    vectors = embedding_model.encode(chunks)

    # Store text and vectors together so we can retrieve the text later using the vector index
    vector_db = {
        "text_chunks": chunks,
        "vectors": vectors
    }
    print("Local Vector Database successfully hydrated.")
    return vector_db

# 4. SEMANTIC RETRIEVAL MECHANISM
def expand_query(query):
    # If query is too vague/short for Gemini to expand meaningfully, add manual variants
    generic_intent = ["purpose", "objective", "goal", "motive", "aim", "about", "overview"]
    if any(w in query.lower() for w in generic_intent):
        return [
            query,
            "What is this document about?",
            "What problem does this project solve?",
            "What are the aims and objectives?",
            "What is the scope of this report?"
        ]
    
    expansion_prompt = f"""Rephrase this question into 2-3 alternative phrasings 
that mean the same thing. Return only the rephrased questions, one per line, no numbering.
Question: {query}"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=expansion_prompt,
        )
        alternatives = response.text.strip().split("\n")
        return [query] + [a.strip() for a in alternatives if a.strip()]
    except:
        return [query]
    
def retrieve_relevant_context(query, vector_db, top_k=5, min_score=0.35):
    """
    Retrieves the Top K most relevant matching text chunks instead of just one,
    preventing context fragmentation.
    """
    queries= expand_query(query)
    query_vectors = embedding_model.encode(queries)
    avg_vector = np.mean(query_vectors, axis=0, keepdims=True)  # average all phrasings

    similarity_scores = cosine_similarity(avg_vector, vector_db["vectors"])[0]
    
    # Get the indices of the top K highest scores
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    # Filter out chunks below the minimum relevance threshold
    filtered = [(idx, similarity_scores[idx]) for idx in top_indices 
                if similarity_scores[idx] >= min_score]
    
    if not filtered:
        return None, 0.0
    
    # Extract and combine the text from all top matching chunks
    retrieved_chunks = [vector_db["text_chunks"][idx] for idx in top_indices]
    combined_context = "\n\n--- Source Context Layer ---\n".join(retrieved_chunks)
    
    return combined_context, filtered[0][1]  # return the highest score among the top matches for reference

# 5. PROMPT ENGINEERING AND GENERATION LOOP
def run_chatbot(vector_db):
    """
    Launches the interactive chat terminal, bridges retrieval to Gemini,
    and applies rigorous prompt engineering system rules.
    """
    print("\n PDF Chatbot initialized! Type 'exit' to quit.")
    print("=" * 60)
    
    last_answer = ""  # track previous answer

    while True:
        query = input("\n Ask a question about your document: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
            
        if not query.strip():
            continue

        # If query is a follow-up (short + no real content), inject last answer as context
        followup_triggers=["that", "it", "this", "short", "summarize", "brief", "again", "explain more"]
        is_followup = len(query.split()) <= 5 and any(w in query.lower() for w in followup_triggers)

        if is_followup and last_answer:
            # Don't retrieve — just ask Gemini to rework the last answer
            crafted_prompt = f"""
The user previously received this answer:
\"\"\"
{last_answer}
\"\"\"
Now they are asking: "{query}"
Rework the above answer accordingly. Do not add new information.
ANSWER:
"""
            print("🤖 Gemini Thinking...\n")
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=crafted_prompt,
                )
                last_answer = response.text
                print(last_answer)
            except Exception as e:
                print(f"❌ Error: {e}")
            print("=" * 60)
            continue  # skip normal retrieval flow    
        # Normal retrieval flow
        context, score = retrieve_relevant_context(query, vector_db, top_k=5)

        if context is None:
            print("⚠️ No relevant context found in document for that question.")
            print("=" * 60)
            continue

        print(f" [System: Retrieved Top 5 matching context layers. Highest score: {score:.4f}]")

        crafted_prompt = f"""
You are an expert AI academic assistant reading an uploaded document.
Answer the user's question using ONLY the factual information contained in the context block below.
If the context does not contain the answer, state clearly: "I cannot find that information in the provided document."

CONTEXT BLOCK:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

ANSWER:
"""
        print("🤖 Gemini Thinking...\n")
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=crafted_prompt,
            )
            last_answer = response.text  # 👈 save for follow-ups
            print(last_answer)
        except Exception as e:
            print(f"❌ Error communicating with Gemini API: {e}")
        print("=" * 60)

# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Specify the path to any PDF you want to upload (research paper, textbook, or notes)
    PDF_FILE_PATH = "pdf_chatbot/sample_paper.pdf"
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ Error: Please place a PDF named '{PDF_FILE_PATH}' in your script folder first!")
    else:
        # Step 1: Run Ingestion
        document_chunks = extract_and_chunk_pdf(PDF_FILE_PATH)
        
        # Step 2: Hydrate local Vector Database
        local_db = build_vector_database(document_chunks)
        
        # Step 3: Open the Interactive Chat Session
        run_chatbot(local_db)

