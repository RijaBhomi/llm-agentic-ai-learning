import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import urllib.request
import json

# Load environment variables from .env file
load_dotenv()
client= genai.Client() 
MODEL_NAME= 'gemini-2.5-flash'

# 1. Menubook: defining native python tools
# gemini reads these docstrings and typing constraints to understand the tools

def mock_weather_and_packing_api(destination: str) -> str:
    # fetches actual real-time weather condition for any city globally
    # using live weather api and returns adapative packing advice
    city= destination.strip().replace(" ", "+") # for API compatibility

    try:
        # querying a live, free JSON weather service
        url= f"https://wttr.in/{city}?format=j1"
        req= urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req) as response:
            data= json.loads(response.read().decode())

            # extract current temp and condition from live data
            current_condition= data['current_condition'][0] 
            temp_c= current_condition['temp_C']
            weather_desc= current_condition['weatherDesc'][0]['value']  

            # dynamically determine packing suggestions based on real-time temp
            temp_int = int(temp_c)
            if temp_int < 12:
                packing = "Heavy winter coat, thermal layers, scarf, and warm boots."
            elif 12 <= temp_int <= 22:
                packing = "Light jacket, layered sweaters, long pants, and comfortable walking shoes."
            else:
                packing = "Breathable t-shirts, shorts or light skirts, sunglasses, and a hat."
                
            return json.dumps({
                "weather": f"Currently {weather_desc}, temperature sitting at {temp_c}°C.",
                "packing_suggestions": packing
            })
        
    except Exception:
        # Fallback safeguard in case of network issues or typos
        return json.dumps({
            "weather": "Information temporarily unavailable.",
            "packing_suggestions": "Pack versatile, smart casual clothes suitable for general outdoor travel."
        })
    
def universal_currency_converter(amount: float, source_currency: str, target_currency: str) -> str:
    # converts baseline budget from any source currency to any target currency
    src= source_currency.upper().strip()
    tgt= target_currency.upper().strip()

    try:
        # querrying the live api
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            rates = data.get("rates", {})

            # guard rail checking if both currency keys exist in the global database
            if src in rates and tgt in rates:
                src_rate_to_usd= rates[src]
                tgt_rate_to_usd = rates[tgt]

                # converting source amt to baseline USD
                amount_in_usd = amount / src_rate_to_usd
                # converting baseline USD to target currency
                final_converted_amount = amount_in_usd * tgt_rate_to_usd

                # calculating the direct local exchange rate for user visibility
                local_exchange_rate = tgt_rate_to_usd / src_rate_to_usd

                return json.dumps({
                    "status": "SUCCESS",
                    "original_amount": amount,
                    "source_currency": src,
                    "target_currency": tgt,
                    "calculated_exchange_rate": round(local_exchange_rate, 5),
                    "converted_amount": round(final_converted_amount, 2)
                })
            else:
                return json.dumps({
                    "status": "ERROR",
                    "message": f"Currency code(s) not recognized. Please ensure both '{src}' and '{tgt}' are valid ISO currency codes."
                })
    except Exception as e:
        return json.dumps({
            "status": "ERROR",
            "message": f"Currency conversion service is currently unavailable. Please try again later. Error details: {str(e)}"
        })
    
def budget_breakdown_calculator(total_amount: float, days: int) -> str:
    # calculates strict financial framework for the trip allocating funds
    # across accomodation, dining and daily leisure spending
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

# This dictionary maps strings to our actual code functions for easy execution loops
AVAILABLE_TOOLS = {
    "mock_weather_and_packing_api": mock_weather_and_packing_api,
    "universal_currency_converter": universal_currency_converter,
    "budget_breakdown_calculator": budget_breakdown_calculator
}

# 2. Agent inference interactive loop
def generate_travel_plan(user_request: str):
    print(f"\n User Travel Request: \"{user_request}\"")
    print('='* 70)

    system_rules = """
    You are an elite, high-end AI Travel Concierge. 
    Your goal is to organize an incredible trip itinerary based on the user's input parameters.
    
    CRITICAL PROTOCOL:
    1. You MUST call `mock_weather_and_packing_api` to get real-time weather and packing advice for the destination city.
    2. Identify the user's starting currency (e.g., NPR) and the destination country's currency. You MUST call `universal_currency_converter` to convert their budget.
    3. Use the converted target currency amount and total days to call `budget_breakdown_calculator` to structure their financial framework.
    
    Do not guess, hallucinate, or calculate math yourself. Rely strictly on the tool outputs to build your final beautiful response.
    """

    # initial api inference pass telling Gemini what tools are sitting in our menu
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_request,
        config=types.GenerateContentConfig(
            system_instruction=system_rules,
            tools=[mock_weather_and_packing_api, universal_currency_converter, budget_breakdown_calculator]
        )
    )

    #agent tool execution router
    if response.function_calls:
        print("[AGENT DECISION]: Activating background data tools to process parameters...")
        tool_responses_content = []

        for call in response.function_calls:
            print(f"  Executing Tool: {call.name}({call.args})")
            
            # Match the function string name to our local python function
            tool_func = AVAILABLE_TOOLS[call.name]
            
            # Execute your local math/API code
            execution_result = tool_func(**call.args)
            print(f"  Tool Data Gathered: {execution_result}")
            
            # Package the results securely using the required Google parts array format
            tool_responses_content.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": execution_result}
                )
            )
        
        # Resubmit the data back to Gemini so it can read your code output
        print("\n Sending structural calculations back to Agent for final itinerary packaging...")
        final_itinerary = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=user_request)]),
                response.candidates[0].content,
                types.Content(role="tool", parts=tool_responses_content)
            ]
        )
        print("\n [YOUR CUSTOM TRAVEL ITINERARY]:")
        print("-" * 60)
        print(final_itinerary.text.strip())
        print("-" * 60)

    else:
        print("\n [DIRECT RESPONSE]:")
        print(response.text.strip())

# running the agent
if __name__== "__main__":
    sample_prompt = (
        "I want to go to Tokyo for 6 days. My total budget is 250,000 NPR. "
        "Please convert my budget to Japanese Yen, get the real weather/packing styles "
        "for Tokyo right now, and give me a clear breakdown of hotel vs activity funds."
    )
    generate_travel_plan(sample_prompt)