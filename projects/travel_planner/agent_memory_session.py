import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()
MODEL_NAME = 'gemini-2.5-flash'

# setting a safety limit for learning purposes so we can watch trim happen live
TOKEN_SAFETY_CEILING = 600 

def run_stateful_agent():
    print(" [SYSTEM INITIALIZATION]: Activating Stateful Agent Session...")
    print(f" [GUARDRAIL ACTIVE]: Max memory threshold set to {TOKEN_SAFETY_CEILING} tokens.\n")
    print("Agent: Hello! I am your memory-managed companion. Let's chat! (Type 'exit' to quit)")
    print("=" * 70)

    # step 1: creating stateful chat object
    # client.chats.create()- automatically handles tracking user/model history
    chat= client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful, brief AI assistant. Keep responses under 3 sentences."
        )
    )

    # starting continuous terminal dialogue loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye! Ending session.")
            break

        if not user_input.strip():
            continue

        # send the user message to active chat session context
        response = chat.send_message(user_input)
        
        print(f"\n Agent: {response.text.strip()}")
        print("-" * 50)

        # step 2: live token accounting
        # fetching the exact history array directly from sdk chat state manager
        current_history= chat.get_history()

        # Call Google's token counting utility on our aggregated message history
        token_count_response = client.models.count_tokens(
            model=MODEL_NAME,
            contents=current_history
        )
        total_tokens = token_count_response.total_tokens
        print(f"[MEMORY METRIC]: Current Session Weight: {total_tokens} tokens")

        # step 3: memory trimmer (slide window guardrail)
        # If the total history tokens surpass our threshold, we must trim old data!
        if total_tokens > TOKEN_SAFETY_CEILING:
            print(f" [WARNING]: Memory weight ({total_tokens}) exceeded safety ceiling!")
            print(" [MEMORY AUTO-TRIM]: Chopping oldest back-and-forth exchange...")
            
            # A complete exchange contains 1 User part and 1 Model part. 
            # So we remove the first 2 entries from our history list to slide the window forward.
            while total_tokens > TOKEN_SAFETY_CEILING and len(chat._history) > 2:
                # Remove the oldest message exchange
                chat._history.pop(0) # Drops oldest User prompt
                chat._history.pop(0) # Drops oldest Model response
                
                # Recalculate token weight to see if we are back in the safe zone
                token_count_response = client.models.count_tokens(
                    model=MODEL_NAME,
                    contents=chat.get_history()
                )
                total_tokens = token_count_response.total_tokens
                
            print(f"[MEMORY STABILIZED]: New memory weight sitting comfortably at {total_tokens} tokens.")

if __name__ == "__main__":
    run_stateful_agent()



