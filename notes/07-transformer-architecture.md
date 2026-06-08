### Transformers Architecture
#### **A. Encoder Block**
![[Pasted image 20260604115037.png]]
1. **Input Embeddings & Positional Encoding**
- **Input embeddings:** Converts the words into vector to capture meaning of the words
- **Positional encoding:** Processing the entire sentence is done in parallel so positional encoding provides sense of order, it is vector that gives context as per position of the word in sentence
*Word → Embedding → Positional Embedding → Final Vector, termed as Context.*

**2.Multi-head attention**
![[Pasted image 20260604115800.png|329]]
- **Self -attention mechanism:** Rather than focusing on one word at at time, self-attention mechanism evaluates how relevant each word is in relation to other words in sequence
- Multi-head attention: looking at one sentence through multiple perspectives.

3. **Feed-forward Neural Network**
![[Pasted image 20260604120403.png|283]]
- Its like a photographer adjusting focus after taking multiple shots
- this FNN in transformers acts like lens that sharpens imp details while discarding irrelevant info so that the image is more clearer and refined
- So the output from multi-head attention is passed there where each attention vector is processed independently
- consists of two linear transformations with ReLU activation function in between that introduces non-linearity to capture more complex patterns
- As each attention output is treated independently in parallel, transformers are highly scalable and faster to train.

**4.Add & Norm**
- this step is like merging every members report task into final report instead of discarding earlier version, we merge them so that no valuable insight is lost
- Here each layer includes **residual connections** that mean original input is added back to output to avoid losing info
- After that result is normalized using **Layer Normalization** which keeps the values at similar scale
Then all these steps are repeated multiple times.

#### **B. DECODER BLOCK**
1. **Input Embedding and Positional encoding (again)**
- Like we're training a translator to convert English sentences into French so the English sentence goes through encoder block and French translation goes through decoder block where first words are converted into vector and then info about their position is added 

2.**Masked Multi-Head Attention (Look=Ahead Attention)**
- this allows the decoder to focus on the input it's already seen while generating a sentence but prevents the model from cheating by looking ahead at a future words
- so mask ensures that when predicting next word the model only considers the words it has already generated up to that point
- after masking, decoders passed the info through self-attention block where attention vectors are generated for each word in French like in encoder but the future words remain hidden

**3.Encoder-Decoder Attention Block**
![[Pasted image 20260604125033.png]]
- After masked attention, next part of decoder block brings in the info from encoder block through an encoder-decoder attention block
- **Also called cross-attention**
- Self attention focuses on relationships between words in a single sequence while cross-attention allows decoder to look at encoder's output
- so, this means decoder can focus on both the sentence its generating and encoded representation of the input sentence at the same time
- **HOW THIS WORKS?**
	- Query comes from decoder (which focuses on partial output that is generated so far)
	- keys and values come from encoder (which contain info about entire input sentence)
	- This cross-attention allows the decoder to selectively focus on specific parts of the input while generating each word in the output.

4. **Feed-Forward Neural Network & Add and Norm**
![[Pasted image 20260604125307.png]]
- FNN refines the info coming from multi-head attention layers
- Similar to encoder block, decoder also uses residual connections and layer normalization where residual connection allow the model to add original input back to output of FNN and normalization ensures smooth training by scaling the output to consistent range

- After passing attention vectors through feed forward network, these vector is passed to linear layer. this layer adjusts the dimensions of the output to match the number of words in target value
- next the output passes through Softmax layer that converts the ouput into probability distribution where values are positive and sum up to 1.0 that represent the likelihood of each word being in the next sequence
- finally the word with highest probability in this distribution is selected as the next word generated translation and this process repeats until the full sentence is translated