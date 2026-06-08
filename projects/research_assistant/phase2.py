# System Prompt
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

SYSTEM_PROMPT = """You are a Socratic teacher. Never give direct answers. Only ask questions."""

response = client.chat.completions.create(
    model="openai/gpt-5-nano-2025-08-07",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Tell me about transformer architecture"}
    ]
)

print(response.choices[0].message.content)

# now the message array has two entries- a "System" role and "User" role
# model sees both as one long token sequence
# system message appears first so it heavily shapes what the model considers "appropriate" for next token
# its like steering its probablity distribution towards certain behavior before the conversion even starts
