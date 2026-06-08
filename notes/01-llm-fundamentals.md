### STAGE 1: Pre-Training
![[Pasted image 20260515092220.png]]
- Process: Internet bata massive chunk of data linxa and then 6000 GPUs are used to compress that 
- Result: as a result u get a **Base Model**
- Base Model: its like a document completer, it does know how to talk yet, like if i ask "What is the capital of Nepal", it might reply "What is the capital of India".
- This phase is kinda like document generator (more contents but low quality)
- Its like gathering tons and tons of knowledge.

### STAGE 2:Fine Tuning
Making that document generator into an **Assistant Model**
- in this phase, its more like changing the internet documents into a certain alignment like question and answer format so that it acts as an assistant model
1.  Writing labeling instructions where we tell the model how to behave sort of
2. then we hire people to create ideal high quality Q&A, and/or comparison 
3. finetune base model on this data, takes about 1 days
4. Then, we obtain assistant model
5. Run alot of evaluations
6. Deploy it
7. Monitor, and collect misbehaviors, go to step 1 (so like, when chatting with the model, if it gives wrong response, that response is given to the persons who write the Q&As and then they overwrite it with right answers and then when u again evaluate the model it gets better, so its an iterative process)

### STAGE 3: RLHF/ Preference Optimization
SO in the stage 3 of fine tuning
We can use a second kind of label: **Comparisons**
- like instead of writing answers, we compare the Answers given by the model to find the best answers
- These comparisons can be used to further finetune the model, this process is known as Reinforcement Learning from Human Feedback (RLHF)

Massive Internet Data
        ↓
Pretraining
        ↓
Base Model
        ↓
Supervised Fine-Tuning (SFT)
        ↓
Assistant Model
        ↓
RLHF / Preference Optimization
        ↓
Helpful & Safer Chat Assistant