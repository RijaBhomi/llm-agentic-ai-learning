from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="openai/gpt-5-nano-2025-08-07",
    messages=[
        {"role": "user", "content": "What is attention in transformers? One paragraph."}
    ]
)

print(response.choices[0].message.content)

# Bare API Call
# HTTP POST request is send to OpenRouter API's server
# message gets routed to gpt-5-nano
# next-token predicition is done in a loop, predicting the most probable next word given everything before it
# appending it and preditcing until it decides to stop
# response.choices[0].message.content is the collected result of all those predictions.

# rn, messages array is the entire convo that is a list of dictionaries with role and content
# there's only one message -the user's and modek has no memory of any previous runs
# every call starts completely fresh, no context of previous calls, no memory, no learning from previous calls. Each call is independent.