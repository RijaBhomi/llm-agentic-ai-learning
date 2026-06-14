import os
import json
import urllib.request
from dotenv import load_dotenv
from google import genai
from google.genai import types

# loading env keys
load_dotenv()
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

# python functions (isolated tools) that the agents can call to get specific information or perform specific tasks.
def weather_tool(destination: str) -> str:
    # queries wttr. in for real-time global weather data
    city= destination.strip().replace(" ", "+")
    try:
        url= f"https://wttr.in/{city}?format=j1"
        req= urllib.request.Request(url, headers= {'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data= json.loads(response.read().decode())
            current= data['current_condition'][0]
            return json.dumps({"city": destination, "temp_c": current['temp_C'], "condition": current['weatherDesc'][0]['value']})
    except Exception:
        return json.dumps({"city": destination, "temp_c": "Unavailable", "condition": "Unavailable"})

def currency_tool(amount: float, source: str, target: str)-> str:
    # converts money using live financial exchange tracking tiers
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            rates = data.get("rates", {})
            converted = (amount / rates[source.upper()]) * rates[target.upper()]
            return json.dumps({"original": amount, "from": source, "to": target, "converted_amount": round(converted, 2)})
    except Exception:
        return json.dumps({"error": "Conversion failed"})

def budget_tool(total_amount: float, days: int) ->str:
    # splits a total budget into accomodation and activities pools
    return json.dumps({
        "hotel_funds": round(total_amount * 0.45, 2),
        "activity_funds": round(total_amount * 0.55, 2),
        "daily_average": round(total_amount/ days, 2)
    })

# 2. Specialist worker definition 
def run_weather_worker(user_intent: str)-> str:
    # agent that uses the weather tool to fetch weather data
    config= types.GenerateContentConfig(
        system_instruction= "You are a weather expert. Use the weather_tool to check the weather and give packing tips",
        tools=[weather_tool]
    )

    #one- shot execution loop to resolve the function call automatically
    response= client.models.generate_content(
        model= MODEL_NAME,
        contents= user_intent,
        config= config
    )

    # simple orchestration hand off: execute tool if model asks for it
    if response.function_calls:
        call= response.function_calls[0]
        result= weather_tool(**call.args)
        # feed the tool data back to worker for final formatting
        final= client.models.generate_content(
            model= MODEL_NAME,
            contents=[user_intent, response.candidates[0].content, types.Part.from_function_response(name=call.name, response={"result": result})]
        )
        return final.text
    return response.text

def run_currency_worker(user_intent: str) -> str:
    # agent that only converts money using its currency tool
    config = types.GenerateContentConfig(
        system_instruction="You are a financial exchange expert. Use currency_tool to convert funds.",
        tools=[currency_tool]
    )
    response = client.models.generate_content(model=MODEL_NAME, contents=user_intent, config=config)
    if response.function_calls:
        call = response.function_calls[0]
        result = currency_tool(**call.args)
        final = client.models.generate_content(
            model=MODEL_NAME,
            contents=[user_intent, response.candidates[0].content, types.Part.from_function_response(name=call.name, response={"result": result})]
        )
        return final.text
    return response.text

def run_budget_worker(user_intent: str) -> str:
    """Specialist Agent: Only breaks down finances using its budget tool."""
    config = types.GenerateContentConfig(
        system_instruction="You are a budget math expert. Use budget_tool to split trip funds.",
        tools=[budget_tool]
    )
    response = client.models.generate_content(model=MODEL_NAME, contents=user_intent, config=config)
    if response.function_calls:
        call = response.function_calls[0]
        result = budget_tool(**call.args)
        final = client.models.generate_content(
            model=MODEL_NAME,
            contents=[user_intent, response.candidates[0].content, types.Part.from_function_response(name=call.name, response={"result": result})]
        )
        return final.text
    return response.text

# supervisor orchestrator that manages the conversation and tool access for the travel planning agent
# 3. UPGRADED: THE STATEFUL TEAM ORCHESTRATION LOOP

def run_team_orchestrator():
    print(" [SUPERVISOR SYSTEM INITIALIZED]: Stateful Team Loop Active.")
    print(" Supervisor: Tell me your travel plans! I will remember our chat and coordinate specialists.")
    print("=" * 80)
    
    supervisor_instruction = """
    You are the Manager/Supervisor of a travel agency team with continuous chat memory.
    
    Your job is to look at the user's latest message and determine which specialists need to run right now.
    Output your decision strictly as a JSON object with True/False values.
    
    If the user asks a follow-up question about something already answered or if no tools are needed,
    set all keys to false.
    
    Format: {"need_weather": false, "need_currency": false, "need_budget": false}
    """
    
    # We turn the Supervisor into a persistent Chat Session!
    supervisor_chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=supervisor_instruction,
            response_mime_type="application/json"
        )
    )
    
    # We also keep a persistent background chat session for compiling the final answers
    synthesis_chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction="You are the lead travel manager. Use your conversational memory and worker data to guide the user naturally."
        )
    )

    while True:
        user_prompt = input("\n You: ")
        if user_prompt.lower().strip() == 'exit':
            print("\n Goodbye! Have an amazing trip!")
            break
            
        if not user_prompt.strip():
            continue

        print("\n [SUPERVISOR THINKING]: Analyzing request context...")
        
        # Pass the message to the supervisor chat to get the structural plan
        plan_response = supervisor_chat.send_message(user_prompt)
        
        try:
            task_matrix = json.loads(plan_response.text)
            print(f" [SUPERVISOR TASK MATRIX]: {task_matrix}\n")
        except Exception:
            # Fallback if JSON parsing glitches out
            task_matrix = {"need_weather": False, "need_currency": False, "need_budget": False}

        shared_team_notebook = {}
        
        # Delegate only if the supervisor says True
        if task_matrix.get("need_weather"):
            print(" [DELEGATION]: Sending task to Weather Specialist...")
            shared_team_notebook["weather_data"] = run_weather_worker(user_prompt)
            
        if task_matrix.get("need_currency"):
            print(" [DELEGATION]: Sending task to Currency Specialist...")
            shared_team_notebook["currency_data"] = run_currency_worker(user_prompt)
            
        if task_matrix.get("need_budget"):
            print(" [DELEGATION]: Sending task to Budget Specialist...")
            shared_team_notebook["budget_data"] = run_budget_worker(user_prompt)

        # Build the compilation prompt for this specific turn
        compilation_command = f"User said: {user_prompt} | New specialist worker data updates: {json.dumps(shared_team_notebook)}"
        
        # Send this directly into our continuous answer engine stream
        final_result = synthesis_chat.send_message(compilation_command)
        
        print("\n" + "="*40 + " FINAL OUTPUT " + "="*40)
        print(final_result.text.strip())
        print("=" * 94)

if __name__ == "__main__":
    run_team_orchestrator()