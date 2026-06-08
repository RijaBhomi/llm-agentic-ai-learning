### Transformers
- encoder-decoder architecture powered entirely by attention layers
- RNN processed words one at a time, like reading a book where u have to finish one page before moving to the next
- but with TRANSFORMERS entire sentence can be fed to the model all at once and model learn the relationships between the words simultaneously, rather than waiting for each word to process individually
- Like processing entire paragraph or entire page of a book at once
- RESULT= no vanishing gradient problem and speed issue
-  Here comes **SELF ATTENTION**
#### Self Attention
- Imagine a sentence like "The cat sat on the mat", in RNN each word is processed in sequenced but with self attention each word creates a kind of representation of itself like: how does **“cat”** relate to **“mat”**? Or how does **“sat”** relate to **“on”**?
- This happens in parallel
- STEPS
	- **Input representation**: representing each word as embedding and creating vectors for  “The,” “cat,” “sat,” “on,” “the,” and “mat.”
	- **Key, Query, and value vectors:** For each word, create these three vectors that helps to determine how much attention a word should play to others.
		- KEY: set of instructions
		- QUERY: question
		- VALUE: info
	- **Calculating attention scores** : 
		- by taking dot products of query sector of current word with key vector of all other words
		- then, these scalar products are scaled and transformed into probabilities using softmax function that shows similarity and correlation between current word and every other word in the sequence
		- The dot product captures how much attention the model should assign to each word based on their similarity in the feature space.
	- **Weighted sum of values:** That attention score is used to weigh the values of all words which becomes new representation of current word that shows relationship with other words in sequence
	- Then **repeat for each word**  Each word gets a new representation based on its relationships with others.

### Multi-Head Attention
- its like viewing multiple perspectives like in a movie there are so many cameras like one for focusing on actor's face, next on the bg, next on body language and when they are merged together by the director it creates more dynamic scene
- same with Multi-head attention where each head focus on different relationships like one on verbs, another subject-object connection which adds a unique perspective that makes data rich in patterns
