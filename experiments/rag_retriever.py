import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Simulating a single long document
large_document = """
The flight controller is the brain of a fixed-wing UAV. It processes sensor data from the IMU, GPS, and airspeed sensors to adjust control surfaces like ailerons, elevators, and rudders dynamically. For stable autonomous navigation, waypoint-following algorithms calculate the heading error and adjust the roll angle to maintain lateral track alignment.

On a completely different note, pizza dough requires a long fermentation process to develop complex flavors. Using a high-hydration flour blend of around 70% allows the yeast to produce tiny carbon dioxide pockets, resulting in a light, airy, and crispy Neapolitan crust when baked in a wood-fired oven at high temperatures.

When training reinforcement learning agents for drone simulation environments, the reward function must penalize erratic movements. If the agent experiences high angular acceleration, a negative reward is applied. This forces the neural network to converge toward smooth control transitions, optimizing battery life and structural longevity.
"""

# implementing the chunking function
def chunk_text(text, chunk_size=200, chunk_overlap=40):
    chunks=[]
    start=0
    while start<len(text):
        end= start +chunk_size
        chunk= text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        #slide the window forward by chunk_size minus the overlap
        start += (chunk_size - chunk_overlap)
    return chunks

# structural chuker
def chunk_by_paragraph(text):
    # Splits the document wherever there is a double newline
    paragraphs = text.strip().split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

# Run the new structural chunker
text_chunks = chunk_by_paragraph(large_document)

print(f"Original document split into {len(text_chunks)} distinct paragraphs.\n")
for i, c in enumerate(text_chunks):
    print(f"--- Chunk {i+1} --- \n{c}\n")


# loading embedding api, vectorize chunks and write search query
# Initialize local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize all text chunks
chunk_vectors = model.encode(text_chunks)

# Your search query
query = "How do you make a drone fly smoothly using machine learning?"
query_vector = model.encode([query])

# Calculate Cosine Similarity
similarity_scores = cosine_similarity(query_vector, chunk_vectors)[0]

# Get the top matching chunk index
best_chunk_idx = np.argmax(similarity_scores)

print("=" * 60)
print(f"Query: '{query}'")
print(f"Top Match Confidence Score: {similarity_scores[best_chunk_idx]:.4f}")
print("=" * 60)
print(f"Retrieved Context:\n\n{text_chunks[best_chunk_idx]}")
print("=" * 60)