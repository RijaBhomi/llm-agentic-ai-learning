import os
import json
import urllib.request
from dotenv import load_dotenv
from google import genai
from google.genai import types

# loading env keys
load_dotenv()
client= genai.Client()
MODEL_NAME= 'gemini-2.5-flash'

# 1. Core core tools
def mock_weather_and_packing_api(destination: str)->str:
    # fetches real-time weather conditions for a city and returns packaging advices
    city= destination.strip().replace(" ", "+")
    try:
        url= f"http://wttr.in/{city}?format=j1"
        req= urllib.request.Request(url, headers= {'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data= json.loads(response.read().decode())
            current_condition = data['current_condition'][0] 
            temp_c = current_condition['temp_C']
            weather_desc = current_condition['weatherDesc'][0]['value']

            temp_int= int(temp_c)
            if temp_int < 12:
                packing = "Heavy winter coat, thermal layers, scarf, and warm boots."
            elif 12 <= temp_int <= 22:
                packing = "Light jacket, layered sweaters, long pants, and comfortable walking shoes."
            else:
                packing = "Breathable t-shirts, shorts, sunglasses, and a hat."

            return json.dumps({
                "weather": f"Currenty {weather_desc}, {temp_c}°C",
                "packing_suggestions": packing
            })
    except Exception:
        return json.dumps({"weather": "Unavailable", "packing_suggestions": "Pack smart casual clothes"})

def universal_currency_converter(amount: float, source_currency: str, target_currency: str) -> str:
    """Converts a budget amount from a source currency (like NPR) to a target currency (like JPY)."""
    src = source_currency.upper().strip()
    tgt = target_currency.upper().strip()
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            rates = data.get("rates", {})

            if src in rates and tgt in rates:
                amount_in_usd = amount / rates[src]
                final_converted_amount = amount_in_usd * rates[tgt]
                local_exchange_rate = rates[tgt] / rates[src]

                return json.dumps({
                    "status": "SUCCESS",
                    "original_amount": amount,
                    "source_currency": src,
                    "target_currency": tgt,
                    "calculated_exchange_rate": round(local_exchange_rate, 5),
                    "converted_amount": round(final_converted_amount, 2)
                })
            else:
                return json.dumps({"status": "ERROR", "message": "Invalid ISO currency codes."})
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

def budget_breakdown_calculator(total_amount: float, days: int) -> str:
    """Calculates a strict financial framework splitting money into hotel and activity pools."""
    daily_allowance = total_amount / days
    accommodation_pool = total_amount * 0.45
    food_and_leisure_pool = total_amount * 0.55
    return json.dumps({
        "total_budget": total_amount,
        "trip_duration_days": days,
        "avg_daily_allowance": round(daily_allowance, 2),
        "allocated_hotel_funds": round(accommodation_pool, 2),
        "allocated_activity_funds": round(food_and_leisure_pool, 2)
    })

# Dictionary map for runtime tool execution
AVAILABLE_TOOLS = {
    "mock_weather_and_packing_api": mock_weather_and_packing_api,
    "universal_currency_converter": universal_currency_converter,
    "budget_breakdown_calculator": budget_breakdown_calculator
}

# step 2: stateful chat console with automatic tool interception
def start_stateful_travel_agent():
    print(" [AGENT INITIALIZATION]: Starting Persistent Travel Session...")
    print(" Agent: Hello! I'm your continuous AI Concierge. Where are we heading? (Type 'exit' to quit)")
    print("=" * 70)

    system_rules = """
    You are an elite AI Travel Concierge with continuous conversational memory.
    
    CRITICAL PROTOCOL:
    1. Always use `mock_weather_and_packing_api` when the user mentions their destination city.
    2. When the user gives you a budget, look at past conversation history to identify their destination country, determine its local currency, and call `universal_currency_converter`.
    3. Use the converted target currency value and total days to execute `budget_breakdown_calculator`.
    
    Rely strictly on your background tool data to make financial calculations. Keep answers organized and friendly.
    """

    # binding the tools directly to the continuous chat state session
    chat= client.chats.create(
        model= MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_rules,
            tools=[mock_weather_and_packing_api, universal_currency_converter, budget_breakdown_calculator]
        )
    )

    while True:
        user_msg = input("\n You: ")
        if user_msg.lower().strip() == 'exit':
            print("\n Goodbye! Have a safe trip!")
            break
        if not user_msg.strip():
            continue

        # sending message into active, memory-aware stream
        response= chat.send_message(user_msg)

        # check if gemini is requesting a local tool run
        if response.function_calls:
            tool_responses= []

            for call in response.function_calls:
                print(f"\n [AUTONOMOUS ACTION]: Agent requested tool execution: {call.name}()")
                print(f" [ARGUMENTS EXTRACTED]: {call.args}")

                # running actual local python function
                tool_func = AVAILABLE_TOOLS[call.name]
                result_data = tool_func(**call.args)
                
                print(f" [DATA FETCHED]: {result_data}")
                
                # Save tool response in the correct structural format
                tool_responses.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result_data}
                    )
                )

                # sending the tool output back into the chat session history so the agent can read it
                print("\n [SYNTHESIZING]: Finalizing response using gathered tool metrics...")
                final_response = chat.send_message(tool_responses)
                print(f"\n Agent: {final_response.text.strip()}")
                print("-" * 60)
        
        else:
            print(f"\n Agent: {response.text.strip()}")
            print("-" * 60)

if __name__ == "__main__":
    start_stateful_travel_agent()