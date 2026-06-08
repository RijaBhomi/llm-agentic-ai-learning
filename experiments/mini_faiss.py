import numpy as np
from sentence_transformers import SentenceTransformer

# core embedding model initialization
print("Loading local embedding model...")
model= SentenceTransformer('all-MiniLM-L6-v2')

# Raw dataset simulating different domains 
documents = [
    "Reinforcement learning uses reward functions to optimize drone trajectories.",
    "Flour hydration levels around 70 percent produce the best Neapolitan pizza crust.",
    "Neural networks adjust weights using backpropagation to minimize loss functions.",
    "The flight controller manages the UAV control surfaces dynamically via the IMU.",
    "Sourdough fermentation requires wild yeast cultures for complex flavor profiles.",
    "Ailerons and elevators are critical control surfaces on fixed-wing aircraft."
]

# vectorize docs into dense matrix: Shape (6, 384)
# means 6 documents, each represented by 384 dimensional floating point array
document_embeddings = model.encode(documents)

# custom IVF FAISS index architecture
class MiniFAISSIndex:
    def __init__(self, n_clusters=2):
        # n_clusters= how many geometric cells we want to divide our data into
        self.n_clusters= n_clusters
        self.centroids= None   # cluster centers in embedding space
        self.cluster_cells= {} # maps cluster IDs to text/vector payloads
    
    def train_and_index(self, embeddings, texts):
        # phase 1: Training and Partitioning (building the cells)
        print(f"\n⚡ Training Index: Clustering {len(texts)} documents into {self.n_clusters} cells...")

        # step A: centroid selection
        # For a mini simulation, we pick random vectors from our dataset to act as cluster centers.
        np.random.seed(42) # for reproducibility
        random_indices= np.random.choice(embeddings.shape[0], self.n_clusters, replace=False)
        self.centroids= embeddings[random_indices]  # Shape: (2, 384)

        # initialize empty dictionary buckets for each cluster cell
        for i in range(self.n_clusters):
            self.cluster_cells[i] = []
        
        # step B: vector space partitioning
        # map every single vector to its closest centroid pole using euclidean distance
        for idx, vec in enumerate(embeddings):
            # calculate the distance from this vector to all cluster centroids
            # np.linalg.norm computes the straight-line (L2) geometric distance
            distances = np.linalg.norm(self.centroids - vec, axis=1)  # Shape: (2,)

            # finding index of the absoulte smallest distance
            closest_centroid_id= np.argmin(distances)

            # Hydrate that specific cell bucket with the raw data and vector payload
            self.cluster_cells[closest_centroid_id].append({
                "text": texts[idx],
                "vector": vec
            })
        
        # debugging step: looking how our data gets divided
        for cell_id, contents in self.cluster_cells.items():
            print(f"Cell {cell_id} has {len(contents)} documents.")

    def approximate_search(self, query, top_k=1):
        # phase 2: ANN inference (search bypass)

        # 1. Embeded the query using identical coordinate space
        query_vector= model.encode([query])[0]  # Shape: (384,)

        # 2. Bypass: find closest cluster centroid instead of scanning documents
        centroid_distances= np.linalg.norm(self.centroids - query_vector, axis=1)  # Shape: (2,)
        target_cell_id= np.argmin(centroid_distances)
        print(f" Query localized to Cluster Cell #{target_cell_id}. Skipping all other cells entirely!")

        # 3. pull candidates exclusively from localized cell
        candidates= self.cluster_cells[target_cell_id]
        if not candidates:
            return "No matching candidates in this region of vector space"
        
        # 4. Localized Exact Scan: Brute force ONLY the candidates inside this bucket
        candidate_vectors = np.array([c["vector"] for c in candidates])
        candidate_distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        best_match_local_idx = np.argmin(candidate_distances)

        return candidates[best_match_local_idx]["text"]
    
# 3. excecution flow
if __name__ == "__main__":
    # Instantiate our index with 2 distinct clusters
    faiss_mock = MiniFAISSIndex(n_clusters=2)
    
    # Run the training setup to partition the data space
    faiss_mock.train_and_index(document_embeddings, documents)
    
    # Execute an approximate search query
    query = "Tell me about flight controllers or stabilizers for drones."
    best_match = faiss_mock.approximate_search(query)
    
    print(f"Best ANN Match Result:\n => \"{best_match}\"")
