### Embedding API
- Embedding API like OpenAI's, Hugging Face's takes human language (string) and translates it into Vector
- When text is send to an embedding API, it maps that into a high dimensional mathematical space
 ![[Pasted image 20260602122859.png|407]]
- If two sentence mean similar like: "How does self attention work" and "Explain transformer attention", these embedding API with output vectors with very similar numbers

### Vector Search Intuition
- this means finding related meanings
- Embedding API converts string into coordinates (vectors), so words or sentences with similar meanings get placed in the same neighborhood on the map.
- Like: Puppy is playful" and "Golden retriever is energetic ", vector search will immediately recognize they belong right next to each other because their concepts match, although they have no identical words.

### Cosine Similarity
- Its a metric that is used to calculate how exactly close two vectors stored on the map is
- It looks at the angle between two vectors from the center of the map
	- Score 1: If both vector points on same direction = very similar
	- Score 0: If the vectors are perpendicular (90 degree) = completely unrelated
	- Score -1: They have opposite meaning

**Overall**
![[Pasted image 20260602124411.png]]
- `"Is Cooper Codes a programmer"` -> run through embedding APi -> get a query vector
- **Vector Search + Cosine Similarity:** Search your Vector Database to find the most mathematically similar text. The database returns: `"Cooper codes is a programmer"`
- Take the relevant result, bundle it with ques and give to ChatGPT as context and ChatGPT reads the costume data and accurately replies `"Copper codes is a programmer`

### Semantic Search
- searching by meaning, not words
-  for eg: imagine a user searches: "Chilly evening comfort food"
	- Keyword search: looks for articles containing exactly the same words so it might miss recipe like "Homemade Jhol Momo" as none of the search words match
	- Semantic search: Understands that "chilly evening" means winter/cold weather and "comfort food" means warm filling meals and then it calculates a high similarity score and successfully serves the recipe