import os
from dotenv import load_dotenv
from google import genai

# Load environment variables for the SDK configuration
load_dotenv()

# 1. INITIALIZE CLIENT

print(" Initializing Gemini Client...")
client = genai.Client()

MODEL_NAME = 'gemini-2.5-flash'


# 2. DEFINING THE SIMULATED CONTEXT SYSTEM
system_instruction = "You are an elite research assistant. Synthesize the provided context into an academic answer."
user_question = "Explain how reinforcement learning handles reward variance in UAV flight navigation."

retrieved_context_chunks = [
    "Layer 1: Reinforcement learning models optimize flight control parameters by processing temporal differences.",
    "Layer 2: High reward variance occurs when environmental factors like wind turbulence skew immediate feedback loops.",
    "Layer 3: Proximal Policy Optimization (PPO) mitigates this by clipping policy updates between 0.1 and 0.2.",
    "Layer 4: Telemetry logs from FlightGear simulations confirm that clipped surrogate objectives reduce crash vectors.",
    "Layer 5: Standardizing reward functions across episodic boundaries ensures consistent actor-critic weight updates."
]

# 3. THE PROMPT BUDGET ENGINEER
def run_prompt_budget_experiment(system_rules, question, context_layers, max_budget=150):
    """
    Simulates a production guardrail that continuously tracks token growth,
    ensuring the combined context layers do not breach a pre-allocated budget.
    """
    print(f"\n Strict Prompt Budget Target: {max_budget} tokens")
    print("=" * 60)
    
    # Track the foundational baseline cost (Rules + Question)
    base_text = f"{system_rules}\n\nQuestion: {question}"
    
    # MECHANISM: We call client.models.count_tokens to let the API compute 
    # the precise token count according to the model's active vocabulary.
    base_tokens = client.models.count_tokens(model=MODEL_NAME, contents=base_text).total_tokens
    print(f" Baseline Cost (Instructions + Query): {base_tokens} tokens")
    
    current_tokens = base_tokens
    allowed_context_layers = []
    
    print("\n Iterating through retrieved RAG layers...")
    for idx, chunk in enumerate(context_layers, 1):
        # Calculate the size of the incoming chunk
        chunk_tokens = client.models.count_tokens(model=MODEL_NAME, contents=chunk).total_tokens
        
        # WHY WE DO THIS: Check if adding this context block pushes us over our maximum limit
        if current_tokens + chunk_tokens > max_budget:
            print(f" [BUDGET BREACH] Skipping Chunk #{idx} ({chunk_tokens} tokens). Pushing it would hit {current_tokens + chunk_tokens} tokens!")
            continue # Skip this layer to protect our context window boundary
            
        # If it fits within the budget constraints, append it to our pipeline
        allowed_context_layers.append(chunk)
        current_tokens += chunk_tokens
        print(f"  Chunk #{idx} added safely (+{chunk_tokens} tokens). Running Total: {current_tokens}/{max_budget}")
        
    # Assemble the final optimized payload
    final_context_block = "\n".join(allowed_context_layers)
    final_prompt = f"""
    {system_rules}
    
    CONTEXT DATA:
    \"\"\"
    {final_context_block}
    \"\"\"
    
    USER QUESTION: {question}
    """
    
    final_token_count = client.models.count_tokens(model=MODEL_NAME, contents=final_prompt).total_tokens
    print("=" * 60)
    print(f" Optimized Prompt Ready! Final Structural Token Count: {final_token_count}")
    
    # Send the safely constrained prompt to Gemini
    print("\n Sending optimized prompt to Gemini...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_prompt
        )
        print("\n Response:")
        print(response.text.strip())
        
        # MECHANISM: Read the direct usage metadata returned in the API response headers
        print("\n API Usage Metadata:")
        print(f" - Prompt Input Tokens: {response.usage_metadata.prompt_token_count}")
        print(f" - Model Generated Output Tokens: {response.usage_metadata.candidates_token_count}")
        
    except Exception as e:
        print(f" API Communication failed: {e}")

# Run the execution loop
if __name__ == "__main__":
    run_prompt_budget_experiment(
        system_rules=system_instruction, 
        question=user_question, 
        context_layers=retrieved_context_chunks,
        max_budget=140 # Artificially low budget constraint to force a drop condition!
    )