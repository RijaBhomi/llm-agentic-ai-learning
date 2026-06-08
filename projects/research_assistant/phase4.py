import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS  # Make sure you ran: pip install --upgrade duckduckgo-search
import urllib.request

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

SYSTEM_PROMPT = """You are a research assistant with access to a web search tool.

When given a topic:
1. Decide if you need to search for current information
2. If yes, call the search tool with a good search query
3. Read the results and synthesize a clear summary
4. Cite where information came from

Be concise but thorough."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current real-time weather for Bhaktapur, Nepal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (default: Bhaktapur)"
                    }
                },
                "required": []
            }
        }
    }
]

def web_search(query: str) -> str:
    """Perform a stable search using the updated DuckDuckGo package structural guidelines."""
    print(f"  [searching: '{query}']")
    results = []
    
    try:
        # The modern approach returns a distinct object structure.
        # We explicitly wrap the iterator into a clear list container to secure the payloads.
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=4))
            
            if search_results:
                for r in search_results:
                    # Using safe lookups (.get) prevents crashing if keys are missing or altered
                    title = r.get('title', 'No Title')
                    snippet = r.get('body', 'No Description')
                    url = r.get('href', '#')
                    
                    results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}")
            else:
                return "No matching search results found on the web."
                
    except Exception as e:
        # DEFENSIVE PROGRAMMING: Instead of crashing your pipeline, 
        # we hand the error context directly back to the model as a string response.
        return f"The search tool encountered a temporary connection issue: {str(e)}"
        
    return "\n\n".join(results)

def research_streaming(topic: str, conversation_history: list) -> str:
    conversation_history.append({"role": "user", "content": topic})
    search_attempts= 0
    MAX_SEARCH= 3

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-5-nano-2025-08-07",
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            search_attempts += 1
            # --- JSON METADATA TRACKER (OUR DEBUGGING EYE) ---
            print("\n🚨 DEBUG: EXECUTING MODEL TOOL REQUEST")
            print(json.dumps(message.to_dict(), indent=2)) 
            print("-" * 40 + "\n")
            # -------------------------------------------------
            
            conversation_history.append(message)
            for tool_call in message.tool_calls:
                func_name= tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if func_name == "web_search":
                    result = web_search(args["query"])
                elif func_name == "get_weather":
                    result= get_weather(args.get("city", "Bhaktapur"))
                else:
                    result = f"Unknown tool requested: {func_name}"

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        else:
            print("\nAssistant: ", end="", flush=True)
            full_text = ""

            stream = client.chat.completions.create(
                model="openai/gpt-5-nano-2025-08-07",
                messages=conversation_history,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    full_text += delta

            print()
            conversation_history.append({"role": "assistant", "content": full_text})
            return full_text

# adding weather tool using Open-Meteo API as an example of a second tool
def get_weather(city: str= "Bhaktapur") ->str:
    # bhaktapur coordinates
    lat, lon = 27.6722, 85.4298

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"precipitation,weathercode,windspeed_10m"
        f"&timezone=Asia/Kathmandu"
    )

    try:
        with urllib.request.urlopen(url) as resp:
            import json
            data = json.loads(resp.read())
            current = data["current"]
            
            code = current["weathercode"]
            descriptions = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
                3: "Overcast", 45: "Foggy", 51: "Light drizzle",
                61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
                80: "Rain showers", 95: "Thunderstorm"
            }
            condition = descriptions.get(code, f"Weather code {code}")
            
            return (
                f"Current weather in Bhaktapur:\n"
                f"Temperature: {current['temperature_2m']}°C\n"
                f"Humidity: {current['relative_humidity_2m']}%\n"
                f"Condition: {condition}\n"
                f"Wind: {current['windspeed_10m']} km/h\n"
                f"Precipitation: {current['precipitation']} mm"
            )
    except Exception as e:
        return f"Weather fetch failed: {e}"
        
def main():
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Research Assistant (with web search)")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        research_streaming(user_input, conversation_history)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()