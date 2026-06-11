# LLM & Agentic AI Learning Journey

Welcome to my 60-day challenge repository! This space serves as a structured, public portfolio documenting my deep dive into Large Language Models, Retrieval-Augmented Generation (RAG), and Agentic AI architectures.

---

## 📚 Study Notes

Click on any topic below to view the detailed conceptual breakdown and intuitions:

* **[01. LLM Fundamentals](notes/01-llm-fundamentals.md)** – Pre-training vs. Fine-tuning (SFT, RLHF), Base vs. Assistant models, and hyperparameters.
* **[02. Tokenization & Embeddings](notes/02-tokenization-and-embeddings.md)** – Next-token prediction mechanics and converting text to numerical vectors.
* **[03. Vector Search & Semantic Search](notes/03-vector-search-and-semantic-search.md)** – Cosine similarity mathematical intuition and meaning-based retrieval.
* **[04. Chunking & Vector Databases](notes/04-chunking-and-vector-databases.md)** – Text segmentation strategies and dense information retrieval using FAISS (IVF Index).
* **[05. From RNNs to Attention](notes/05-rnns-to-attention-evolution.md)** – The evolution from sequential dependencies (LSTMs, GRUs) to parallel processing.
* **[06. Self-Attention Mechanism](notes/06-self-attention-mechanism.md)** – Deep dive into Queries ($Q$), Keys ($K$), and Values ($V$) vectors.
* **[07. Transformer Architecture](notes/07-transformer-architecture.md)** – Detailed breakdown of the Encoder-Decoder pipeline, positional encoding, and cross-attention.
* **[08. Hybrid Search & Re-ranking](notes/08-hybrid-search-and-reranking.md)** – Solving the alphanumeric mismatch problem and fixing the "Lost in the Middle" context effect using Cross-Encoders.
* **[09. RAG Lifecycle: Ingestion vs. Inference](notes/09-rag-lifecycle-ingestion-vs-inference.md)** – Breaking down the batch preprocessing pipeline versus the real-time execution loop.
---

## 🛠️ Hands-On Projects

Explore the implementation folders containing source code, experiments, and dedicated project documentation:

### 🤖 Core Projects
* **[PDF Chatbot](projects/pdf-chatbot/)** – A complete implementation utilizing document chunking, FAISS vector storage, and an LLM generation loop.
* **[Research Assistant AI](projects/research-assistant/)** – Building localized memory states and advanced tool-calling capabilities.
* **[Travel Planner](projects/travel-planner/)** – An Agentic tool-calling system built with `gemini-2.5-flash` that fetches live global weather data and gives clothing/packing suggestions and calculate cross-currency metrics for budget planning.

### 🧪 Isolated Experiments
* **[Code Sandbox & Experiments](experiments/)** – Micro-scripts including `mini_faiss.py` and semantic search verification playgrounds.

---

## 📖 External Resources
* **[Useful Links & Papers](resources/useful-links.md)** 
