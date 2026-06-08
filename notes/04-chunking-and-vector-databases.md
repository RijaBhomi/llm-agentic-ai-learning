### Chunking
- Suppose u have a pdf with 100 pages and a user asks a highly specific ques, then passing entire 100 page pdf into embedding model will create a single vector array compressed with 100 pages
- the specific question being asked gets completely washed out and lost as it's trying to represent everything at once
- So, what we do is break down a large text into smaller, meaningful pieces called chunks before turning them into vectors
 **Flow how Retrieval happens**
 ![[Pasted image 20260605145631.png]]
### Vector Data Base
#### FAISS
**The problem:**
- when running vector search using `cosine_similarity`, we perform **Exact Search (Flat index)**
- If we have *N* text chunks in database, script has to calculate mathematical distance between query vector and every single one of those *N* vectors.
	- **Time complexity:** *O(N)*
	- So if *N* = 100, it takes microseconds but if *N* = 10,000 like thousands of text books then system will freeze as it will take more minutes for matrix multiplication to finish
To fix this, Meta built FAISS (Facebook AI Similarity Search) for ANN (Approximate Nearest Neighbor) search.

**ANN & Vector Space Partitioning**
- Instead of scanning everything, ANN narrows the search area down ahead of time
- The core mechanism that FAISS uses to do this is called **IVF Index( Inverted file Index)**
- **HOW THIS WORKS?**
	1. **Clustering (Training phase):** Before running a query, FAISS runs clustering algo like K-Means across all the document vector and identifies high-density region and places called Centroids
	2. **Cell Partitioning:** Every single document vector is assigned to closest centroid that draws invisible geometric boundaries across vector space. This boundary divides the data into localized cells or buckets
	3. **Search bypass:** When we send a query, the system checks which cluster centroid is closest to that query?, if centroid A is closest then the search engine only searches vector assigned to Cell A and skips cells B C D E
- **TRADEOFF**
	- If a document vector sits right between cell A and cell B and query only checks Cell A, then system might miss the borderline vector in cell B
	- FIX: We can tune a parameter called `nprobe` to solve this