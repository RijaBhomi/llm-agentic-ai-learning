# defining data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# database of documents
documents= [
    "A golden retriever is running around in the park", 
    "The stock market is experiencing a significant downturn",
    "The new movie released last week has received bad reviews",
    "The weather in Bhaktapur, Nepal is currently sunny with a high of 25 degrees Celsius",
    "Felines love to curl up in tight cardboard boxes for afternoon naps."
]

# initialize the embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# convert database documents into embeddings
document_vectors= model.encode(documents)
print("Documents successfully vectorized")

# executing vector search
# define a query
query= "Are there any stories about a playful puppy?"

# turn the query into vector
query_vector= model.encode([query])

# calculate cosine similarity between the query vector and All documents
# cosine_similarity expects 2D arrays, returns a matrix of scores
similarity_scores=cosine_similarity(query_vector, document_vectors)[0]

# print result with their maths scores
print(f"\nQuery: '{query}'\n")
print("Results (Ranked by Cosine Similarity):")
print("-" * 50)

# Sort indices from highest score to lowest
ranked_indices = similarity_scores.argsort()[::-1]

for index in ranked_indices:
    score = similarity_scores[index]
    print(f"Score: {score:.4f} | Document: {documents[index]}")