### RNNs

![[Pasted image 20260603094036.png|547]]
RNNs are ANNs with loops which helps them in sequences and memory. RNN learn the similarity while training by introducing loops into their architecture that help them carry information across time steps in the data
- **Input layer:** When each input is passed through the network, the output not only depends only on the current input but also on what RNN has seen before. 
-  **The Hidden State ($a^{\langle t \rangle}$):**  Then in Hidden state, RNN updates this state based on the new input and previous hidden state that's how the model remembers. At any given time step $t$, the state updates dynamically based on the fresh sequence input ($x^{\langle t \rangle}$) blended with the historical hidden state from the previous step ($a^{\langle t-1 \rangle}$). 
- **The Looping Advantage:** Artificial Neural Networks (ANNs) pass information forward and treat inputs independently. RNNs introduce structural loops to handle sequence processing and memory tracking over time steps. 
- **Parameter Efficiency:** Rather than creating a unique, sprawling architecture with separate parameters for every step, an RNN loops through a single hidden layer sequentially. The exact same weights ($W_{aa}, W_{ax}, W_{ya}$) and biases ($b_a, b_y$) are systematically reused at every step.
$$a^{\langle t \rangle} = g_1(W_{aa}a^{\langle t-1 \rangle} + W_{ax}x^{\langle t \rangle} + b_a)$$ $$y^{\langle t \rangle} = g_2(W_{ya}a^{\langle t \rangle} + b_y)$$ *Where $g_1$ and $g_2$ act as activation functions to introduce non-linearity for complex pattern tracking.*
**Major Architectural flaw:**
1. **The Sequential Bottleneck:** Because step $t$ fundamentally requires the calculated output of step $t-1$, calculations must happen one word at a time. This makes it impossible to train them in parallel across modern, powerful multi-core GPUs. 
2. **Vanishing Gradients:** Over extended sequences or long paragraphs, early words are mathematically multiplied across the loop continuously. By the time the network reaches word 100, the foundational context from word 1 has entirely faded away.

### LSTM
- LSTM solves the vanishing gradient problem by introducing three gates:  **three gates**: the **forget gate**, the **input gate**, and the **output gate**. These gates allow the model to decide:
	- What information to **forget**,
	- What new information to **store**,
	- When to **output** information.
- **Forget gate:** first gate that helps remove irrelevant info from the memory to avoid overload
- **Input gate:** After unnecessary info is removed, then input gate controls how much new info flows into the memory, adds important stuffs only
- **Cell state:** This is like the memory of LSTM, carries necessary info forward and gates help decide what to add or remove from this memory
- **Output gate:** Finally after updating the memory, output gate decides what info to output as the next hidden state
- **Hidden state:** its like short-term memory used for immediate prediction while memory cell stores long term patterns. 
**LIMITATIONS:**
- Has multiple gates which is complex so training these models can be slow especially with large datasets or real-time applications
- as its complex, it needs more resources so not ideal where efficiency and speed are critical
- can lead to overfitting in small dataset as their parameters are high in number so poor generalization to new data

### GRU
- lightweight LSTMs
- have 2 gates only 
	- **Reset gate:** determines how much of previous hidden state should be combined with new input, if reset gate is zero= forgets every past info, this helps when to focus on recent inputs when to reset and start with new data
	- **Update gate:** decides how much of the previous hidden state to carry forward in next step, helps to pass on info and discard so that most relevant info gets transfered

### How is attention model different?
- in traditional seq-to-seq models, the encoder sends only final hidden state to the decoder which is like explaining a whole book in one sentence where we lose a lot of imp details along the way
- while in attention model, encoder provides a lot of info to decoder, it passes all the hidden states which is like giving each and every sentence or paragraph of a book not just summary
**How decoder works**
- decoder checks each hidden state it received from the encoder
- Then, it gives each hidden state a score like evaluation how imp each word is for that current translation step
- then decoder multiplies each hidden state by a softmax score that amplifies the imp words and lowers the influence of less imp words
- **Context Creation:** It multiplies the hidden states by these softmax weights to create a custom context vector tailored specifically for that exact moment in time.


![[Pasted image 20260603184748.png|386]]
If translating `"The black cat slept"` into French (`"Le chat noir dormait"`), when the decoder is ready to output `"chat"`, the attention scores will naturally spike heavily on the input hidden state for `"cat"`, allowing the model to focus its "eyes" exactly where it matters.
