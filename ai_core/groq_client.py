import os
import json
from groq import Groq
from dotenv import load_dotenv
from ai_core.prompts import SYSTEM_PROMPT
from ai_core.tools import AVAILABLE_TOOLS, TOOLS_SCHEMA

load_dotenv()

def get_secrets():
    """Get secrets from Streamlit Cloud or .env"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            return st.secrets
    except:
        pass
    return {}

def get_groq_client():
    secrets = get_secrets()
    api_key = secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def chat_with_tools(messages, model="llama-3.3-70b-versatile"):
    """
    Tiếp nhận list dictionary `messages` (role: user/assistant/tool, content: text...).
    Xử lý vòng lặp gọi Tool của LLM cho đến khi hoàn tất hoặc tối đa 10 vòng lặp.
    
    Lưu ý: messages đã bao gồm full chat history từ session state.
    """
    client = get_groq_client()
    if not client:
        return {"error": "GROQ_API_KEY chưa được cấu hình trong file .env."}
    
    tools_used = []
    max_loops = 10
    loops = 0
    
    try:
        while loops < max_loops:
            # Pass all tools - Groq can handle them
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,  # Use all 9 tools
                tool_choice="auto",
                max_tokens=4096,
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # Add assistant message
            messages.append(response_message.model_dump(exclude_none=True))
            
            # If no tool calls, return final answer
            if not tool_calls:
                return {"content": response_message.content, "tools_used": tools_used}
            
            # Process tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_args = {}
                
                tools_used.append({
                    "name": function_name,
                    "args": function_args
                })
                
                # Execute function
                if function_name in AVAILABLE_TOOLS:
                    function_to_call = AVAILABLE_TOOLS[function_name]
                    function_response = function_to_call(**function_args)
                else:
                    function_response = f"Lỗi: Function {function_name} không tồn tại!"
                
                # Add tool result
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })
            
            loops += 1
            
            # After first call, try without tools to get final response
            response2 = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            
            resp2_msg = response2.choices[0].message
            messages.append(resp2_msg.model_dump(exclude_none=True))
            
            if resp2_msg.content:
                return {"content": resp2_msg.content, "tools_used": tools_used}

        return {"content": "Hệ thống đã phân tích nhưng cần tạm ngừng vòng lặp công cụ.", "tools_used": tools_used}
        
    except Exception as e:
        return {"error": f"Lỗi liên kết Groq API: {str(e)}"}
