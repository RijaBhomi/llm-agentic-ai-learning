### Lifecycle of RAG application
- Divided into two completely separate timelines: **ingestion and Inference**.
- It's like running a library: **Ingestion** is the process of buying books, categorizing them and putting them on shelves while **Inference** is the live moment when someone walks to the library, asks a question, and gets an answer.
1. **Ingestion** (Setup Phase)
	- This phase is a workflow that prepares the raw data so that a machine can actually search through it later
	- **THE INGESTION WORKFLOW:**
		- **Load:** Reading huge text out of PDFs, Markdown files or SQL databases.
		- **Chunks:** Slicing these huge texts into smaller, structural paragraph.
		- **Embed:** Passing those text chunks through embedding model for converting them into vectors like coordinate points.
		- **Index:** Storing those vectors into local DB like FAISS or cloud database like Pinecone
		
2. **Inference** (Live Chat Phase)
	- Live, real-time execution loop that happens in millisecond after a user types question into chat bar and hits enter
	- **THE INFERENCE WORKFLOW:**
		- **Vectorize Query:** Converting the question by users into a vector
		- **Retrieve:** FAISS index finding the Top-K chunks that is closest to the query coordinate.
		- **Re-rank and budget:** Passing those matches through Cross-Encoder to sort them by precision and count the tokens to make sure they fit the window.
		- **Generate:** Sending the optimized prompt to Gemini and model uses its trained neural network to predict/generate the final answer text.

---

### Implementation 

I built an isolated script to unify these concepts into a production-grade backend engine:

* **[View Experiment Code](../experiments/hybrid_rerank_playground.py)** – A unified RAG v2 backend engine implementing offline data ingestion (FAISS index matrix) and a live inference loop (FAISS sweeps + Cross-Encoder re-ranking + dynamic token budgeting) with automatic exponential backoff retry logic.
