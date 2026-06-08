from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

SYSTEM_PROMPT = """You are a research assistant. Answer clearly and concisely.
When asked follow-up questions, refer back to what was discussed."""

# This list IS the memory. Nothing else is.
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def chat(user_message: str) -> str:
    # Step 1: add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Step 2: send the ENTIRE history to the model
    response = client.chat.completions.create(
        model="openai/gpt-5-nano-2025-08-07",
        messages=conversation_history  # <-- full history every time
    )

    # Step 3: extract the reply
    reply = response.choices[0].message.content

    # Step 4: add the reply to history so next turn remembers it
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    # Step 5: show token usage so you can see memory growing
    usage = response.usage
    print(f"  [tokens used: {usage.prompt_tokens} in, {usage.completion_tokens} out]")

    return reply


def main():
    print("Research Assistant — type 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        reply = chat(user_input)
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    main()

# every single API call sends the full conversation_history list
# model has zero memory outside of that, so if its not in the list, it doesn't know about it
# this is how we create "memory" - by explicitly including past interactions in the messages we send to the model
# this is also how we create "context" - by including relevant information in the messages, we give the model the context it needs to generate appropriate responses
