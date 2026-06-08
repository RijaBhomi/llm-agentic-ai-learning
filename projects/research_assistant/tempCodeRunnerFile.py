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
