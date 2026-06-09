## Hybrid Search & Cross-Encoder Re-ranking
### The problem with pure Vector Search
- Vector embeddings `all-MiniLM-L6-v2` are good at matching concepts but they are bad at matching exact specific alphanumeric strings, serial codes, or distinct parameters

### The solution: Multi-Stage Retrieval
- So here comes **Hybrid Search**
	- **Dense Retrieval (Vector search):** captures semantic meaning
	- **Sparse Retrieval (keyword/ lexical search):** captures exact word matches, codes...

### Re-ranking
- When these two search methods are combined we can catch everything which is called maximizing **Recall**.
- But LLM faces "Lost in the Middle" effect if feed with too many chunks so high **Precision** is also needed
- This introduces the **Cross-Encoder Re-ranker:**
	- **Stage 1(Bi-encoder Vector search):** documents and queries are embedded independently in lightning-fast speed which is perfect for scanning millions of files to return a rough Top-20
	- **Stage 2 (Cross-Encoder Re-ranker):** Takes the query and document chunk, keep them together into a single neural network pass and calculates deep attention maps between the words. But this is too slow to run on 1 million documents. So, we can use them when scoring our rough Top-20 candidate and bring down the absolute best Top-2 chunks