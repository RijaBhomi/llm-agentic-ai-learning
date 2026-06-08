### Tokenization
- breaking the sentences into smaller pieces (tokens) and giving each of those piece a unique ID number is called tokenization
- for eg: "I love Nepal"  -> ['I', 'love', 'Nepal']  -> [05, 1045, 99]
- unique id is given because Neural networks work with numbers not text.
- for images or voices, tokens are small chunks of those image or voice
![[Pasted image 20260519121027.png]]
### Embedding
- in this process the tokens are converted into vector of numbers
- because random ID numbers have no meaning whatsoever
- ![[Pasted image 20260518150733.png]]![[Pasted image 20260518151012.png]]

- so basically, embedding takes a token and translate it into a long list of numbers (vector)
- this vector represents meaning
- like token `King` gets a specific coordinates on the map
- likewise `queen` gets something that is close to `king`
- while other words like `appple` gets coordinates far awayy
- now we can do maths on these concept
![[Pasted image 20260520214637.png]]
### Next-token Prediction
LLM is like an advanced autocomplete systems
- for eg: input: `The capital of Nepal is` 
-  then this model calculates the probabilities of the next token
- Kathmandu: 97%
- Pokhara: 1%
- Everest- 0.1 %
- then it picks: Kathmadu
- now sentence becomes: `The capital of Nepal is Kathmadu`
- then again it predicts next token probabilities like "." or "," or "which"
- SO, this continues token by token
	- ONE TOKEN AT A TIME
- If a model becomes EXTREMLY good at predicting text, it indirectly learns structures of human knowledge

#### Why Hallucinations happen
- Model is constantly predicting what text is statistically likely, so sometimes it generates believable, fluent, and wrong answers like fake research papers, fake url because "look like" valid text patterns
#### Temperature
- Sometimes model always picks highest probablity so, it gives safe, predictable, boring answers 
- So, temperature is a hyperparameter that controls the randomness and creativity of model's reponse
- high temperature= more creativity

## FLOW
User input
↓
Tokenization
↓
Tokens become embeddings
↓
Transformer analyzes relationships
↓
Model predicts probability distribution for next token
↓
One token selected
↓
Added to sequence
↓
Repeat
