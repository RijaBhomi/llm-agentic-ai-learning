import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Initializing dual models (Bi-encodder for search, cross-encoder for re-ranking)
print("Loading stage 1 Bi-Encoder (Dense Vector Matcher)...")
vector_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading stage 2 Cross-Encoder (Reranker)... ")
# this model evaluates sentence pairs directly to output a probability score from 0 to 1
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Sample data
documents = [
    "Drone navigation relies on GPS coordinates and internal IMU sensors.",
    "Reinforcement learning optimizes aircraft trajectories by balancing stable glide paths.",
    "Pizza dough requires high hydration levels, ideally around 65% to 70% total water mass.",
    "For UAV flight control, Proximal Policy Optimization (PPO) handles high variance rewards.",
    "Baking standard Neapolitan style pizza requires an ambient oven temperature of 450°C.",
    "Unmanned Aerial Vehicles (UAVs) can encounter high reward variance due to sudden wind shear."
]

# pre-compute vector embeddings for Stage 1
document_embeddings= vector_model.encode(documents)

# 3. Advanced two-stage retrieval pipeline
def advanced_retrieval(query, top_k_initial= 5, top_k_final=2):
    print (f"\nQuery: {query}")
    print("-" *60)

    # Stage 1: broad retrieval (getting candidate pool)
    # calculate vector distances (simulation our vector index fallback)
    query_emb = vector_model.encode([query])[0]
    scores = np.dot(document_embeddings, query_emb) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    # simple keyword/lexical check (simulation basic hubrid search)
    # Give a massive algorithmic bonus if the query words match exact words in documents
    for idx, doc in enumerate(documents):
        query_words = set(query.lower().replace("?", "").split())
        doc_words = set(doc.lower().split())
        match_count = len(query_words.intersection(doc_words))
        if match_count > 0:
            scores[idx] += (match_count * 0.1) # Boost score for exact keyword overlap

    # Fetch the top candidates (Stage 1 Recall Pool)
    top_candidate_indices = np.argsort(scores)[::-1][:top_k_initial]
    candidate_chunks = [documents[i] for i in top_candidate_indices]
    
    print(f" STAGE 1 RECALL: Gathered Top-{top_k_initial} Candidates:")
    for c in candidate_chunks:
        print(f"  - {c}")

    # Stage 2: Precision Re-ranking (deep evaluation)
    print(f"\n STAGE 2 PRECISION: Passing candidate pool into Cross-Encoder...")

    # make pairs: [[query, doc1], [query, doc2], ... ]
    pair_inputs = [[query, doc] for doc in candidate_chunks]

    # The Cross-Encoder calculates internal cross-attention scores across both text blocks
    rerank_scores = reranker_model.predict(pair_inputs)

    # Sort the initial chunks based on their updated deep attention scores
    ranked_indices = np.argsort(rerank_scores)[::-1]

    print("\n Cross-Encoder Scoring Re-ordering Breakdown:")
    final_output_chunks = []
    for rank, idx in enumerate(ranked_indices[:top_k_final], 1):
        actual_chunk = candidate_chunks[idx]
        print(f"  Rank #{rank}: Score: {rerank_scores[idx]:.4f} -> \"{actual_chunk}\"")
        final_output_chunks.append(actual_chunk)
    
    return final_output_chunks

# run the engine 
if __name__ == "__main__":
    # Query with a specific keyword edge case: "UAV" and "reward variance"
    user_query = "How do UAV systems navigate high reward variance?"
    
    final_context = advanced_retrieval(user_query, top_k_initial=4, top_k_final=2)
    print("\n Ready for LLM Context Window:")
    for idx, chunk in enumerate(final_context, 1):
        print(f" [{idx}] {chunk}")