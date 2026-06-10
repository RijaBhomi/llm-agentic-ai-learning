import os
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder

# loading system environment keys
load_dotenv()

# 1. Architectural setup and system initialization
print("Initializing RAG v2 Engine...")
MODEL_NAME= 'gemini-2.5-flash'

client = genai.Client()
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Configuration constraints
VECTOR_DIM= 384  # dimension of the bi-encoder embeddings
TOKEN_BUDGET= 200 # max input token allocation for context + question

# 2. Raw knowledge base and structural chunking
raw_kb_manual = """
Domain: UAV Propulsion and Control
Paragraph 1: Unmanned Aerial Vehicles (UAVs) running reinforcement learning algorithms frequently experience high reward variance when sudden environmental wind shear disrupts expected telemetry loops.

Paragraph 2: To handle extreme reward variance during autonomous flight, flight controllers deploy Proximal Policy Optimization (PPO), which leverages clipped surrogate objective functions to smooth weight tracking updates.

Paragraph 3: Traditional fixed-wing aircraft stabilize roll, pitch, and yaw through standard aerodynamic control surfaces, specifically via ailerons, elevators, and main rudders connected to the flight controller deck.

Domain: Culinary Engineering
Paragraph 4: Authentic Neapolitan pizza dough composition requires a high water hydration metric, explicitly holding between 65 percent and 70 percent total liquid content relative to flour mass.

Paragraph 5: True wood-fired baking environments require localized ambient chamber configurations hovering consistently around 450 degrees Celsius to activate immediate yeast fermentation bubbles.
"""

# structural chunking mechanism: segment raw strings cleanly via paragraph markers
chunks = [p.strip() for p in raw_kb_manual.split("\n\n") if p.strip()]
print(f"Successfully processed and tokenized {len(chunks)} document chunks.")

# 3. Build the local FAISS index (stage 1 vector retrieval)
# generating high-dimensional dense embeddings
chunk_embeddings= bi_encoder.encode(chunks).astype('float32')

# initialize FAISS index and add our chunk embeddings
faiss_index= faiss.IndexFlatL2(VECTOR_DIM)
faiss_index.add(chunk_embeddings)  # ingesting embeddings directly into RAM
print(" Local FAISS Index built and hydrated successfully.")

# 4. Unified Retrieval and Inference Pipeline
def run_rag_v2_inference(query):
    print(f"\n [QUERY RECEIVED]: '{query}'")
    print("=" * 70)

    # pipeline step1 : FAISS vector space scan (recall phase)
    query_vector= bi_encoder.encode([query]).astype('float32')

    # pull top 4 candidates across index space
    top_k_recall= 4
    distances, indices = faiss_index.search(query_vector, top_k_recall)

    # extract the matched structural chunks based on retrieved indices
    candidate_chunks = [chunks[idx] for idx in indices[0] if idx != -1]
    print(f"[FAISS RECALL]: Retrieved {len(candidate_chunks)} candidates via Euclidean distance profiling.")

    # pipeline step2 : Cross-Encoder re-ranking (precision phase)
    print(f"\n[Cross-Encoder Re-ranking]: Evaluating candidate relevance...")
    pair_inputs = [[query, doc] for doc in candidate_chunks]
    rerank_scores= cross_encoder.predict(pair_inputs)

    # sorting candidate chunks dynamically by descending precision score
    sorted_indices = np.argsort(rerank_scores)[::-1]
    ranked_chunks = [candidate_chunks[idx] for idx in sorted_indices]

    # pipeline step3: Token budget management (context allocation phase)
    system_instruction = "You are a professional systems engineer. Synthesize an explicit answer utilizing the approved context fragments."
    base_prompt_skeleton = f"{system_instruction}\n\nQuestion: {query}"

    # compute baseline token cost using the gemini client layout
    base_token_weight = client.models.count_tokens(model=MODEL_NAME, contents=base_prompt_skeleton).total_tokens
    print(f"[TOKEN MANAGEMENT]: Baseline Cost: {base_token_weight} tokens. Allocating remaining budget...")
    
    current_token_accumulation = base_token_weight
    validated_context_pool = []

    for rank_idx, chunk in enumerate(ranked_chunks, 1):
        chunk_token_weight = client.models.count_tokens(model=MODEL_NAME, contents=chunk).total_tokens
        
        # Budget evaluation barrier
        if current_token_accumulation + chunk_token_weight > TOKEN_BUDGET:
            print(f" [BUDGET LIMIT HIT]: Dropped Candidate Rank #{rank_idx} ({chunk_token_weight} tokens) to avoid context overflow.")
            continue
            
        validated_context_pool.append(chunk)
        current_token_accumulation += chunk_token_weight
        print(f" [BUDGET ACCEPT]: Added Rank #{rank_idx} ({chunk_token_weight} tokens). Running total: {current_token_accumulation}/{TOKEN_BUDGET}")
    
    # pipeline step4: protected prompt assembly and generation
    context_data_block= "\n\n".join(validated_context_pool)

    final_engineered_prompt = f"""
    {system_instruction}
    
    CRITICAL EXCLUSIVE CONTEXT REFERENCE:
    \"\"\"
    {context_data_block}
    \"\"\"
    
    USER QUESTION: {query}
    """
    
    print("\n [GENERATION]: Dispatching secure payload to Gemini...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_engineered_prompt
        )
        print("\n [GEMINI RESPONSE]:")
        print("-" * 40)
        print(response.text.strip())
        print("-" * 40)
        print(f" [METADATA]: Prompt Input Tokens: {response.usage_metadata.prompt_token_count} | Response Output Tokens: {response.usage_metadata.candidates_token_count}")
        
    except Exception as e:
        print(f"Execution failed via API interface: {e}")

# Run the unified pipeline engine
if __name__ == "__main__":
    test_query = "What ML architecture resolves high variance updates during UAV wind turbulence?"
    run_rag_v2_inference(test_query)