import os
import json
from dotenv import load_dotenv
from ai_core.tools import TOOLS_SCHEMA, AVAILABLE_TOOLS
from huggingface_hub import InferenceClient

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

def get_hf_api_key():
    secrets = get_secrets()
    return secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY")

def get_groq_api_key():
    secrets = get_secrets()
    return secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

def chat_with_hf(messages, model="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Tiếp nhận list dictionary `messages` (role: user/assistant/tool, content: text...).
    Xử lý vòng lặp gọi Tool của LLM cho đến khi hoàn tất hoặc tối đa 3 vòng lặp.
    Sử dụng HuggingFace InferenceClient với function calling support.
    """
    # Use HF token for HF inference provider
    api_key = get_hf_api_key()
    if not api_key:
        return {"error": "HF_API_KEY chưa được cấu hình trong file .env."}
    
    tools_used = []
    max_loops = 10
    loops = 0
    
    # Convert messages for the API (remove system prompt, keep only user/assistant)
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Include system content as first user message if present
            continue
        api_messages.append(msg)
    
    # Add system prompt at the beginning
    system_prompt = """Bạn là một trợ lý AI phân tích chứng khoán Việt Nam thông minh.

LƯU Ý QUAN TRỌNG:
- Model này KHÔNG hỗ trợ xử lý hình ảnh (image input)
- Chỉ xử lý văn bản (text-only)
- Nếu user gửi ảnh, hãy từ chối lịch sự và nói rằng bạn không thể xem ảnh

QUAN TRỌNG - Cách gọi function:
- KHI CẦN GỌI FUNCTION, HÃY SỬ DỤNG CƠ CHẾ TOOL_CALLS CỦA API (đừng viết JSON thủ công vào content)
- KHÔNG BAO GIỜ viết {"type": "function", ...} vào phần content trả về
- Chỉ cần gọi function qua tool_calls, AI sẽ tự động nhận kết quả và trả lời bằng văn bản tự nhiên

Quy tắc trả lời:
- Trả lời bằng tiếng Việt, dùng Markdown để format đẹp hơn
- Khi user hỏi về dữ liệu thị trường, gọi function thích hợp và tổng hợp kết quả
- Không bịa đặt dữ liệu"""
    
    # Prepare messages with system prompt
    final_messages = [{"role": "system", "content": system_prompt}] + api_messages
    
    try:
        # Use Cerebras provider with proper model format
        client = InferenceClient(
            provider="cerebras",
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key=api_key
        )
        
        while loops < max_loops:
            # Call API with tools - reduce tokens to avoid repetition issues
            response = client.chat_completion(
                model=model,
                messages=final_messages,
                tools=TOOLS_SCHEMA,
                max_tokens=2048,
                temperature=0.7
            )
            
            # Get the response message
            if not response.choices:
                return {"error": "No response from HuggingFace API"}
            
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            
            # Add assistant message to conversation
            assistant_msg = {"role": "assistant", "content": content or ""}
            if assistant_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            
            final_messages.append(assistant_msg)
            
            # Check if there are tool calls
            if assistant_message.tool_calls:
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    # Handle parsing errors for function arguments
                    try:
                        func_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    except (json.JSONDecodeError, TypeError):
                        # Try to parse as dict if it's a string representation
                        func_args = {}
                        print(f"Warning: Could not parse function arguments for {func_name}")
                    
                    # Log the tool usage
                    tools_used.append({
                        "name": func_name,
                        "args": str(func_args)
                    })
                    
                    if func_name in AVAILABLE_TOOLS:
                        try:
                            tool_result = AVAILABLE_TOOLS[func_name](**func_args)
                        except Exception as e:
                            tool_result = f"Lỗi khi gọi function {func_name}: {str(e)}"
                    else:
                        tool_result = f"Lỗi: Function {func_name} không tồn tại!"
                    
                    # Add tool result to messages
                    tool_result_str = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else str(tool_result)
                    
                    final_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_str
                    })
                
                # Continue loop to get final response after tool execution
                loops += 1
                continue
            elif content:
                # No tool calls but has content - return final response
                return {"content": content, "tools_used": tools_used}
            else:
                # No content and no tool calls - something went wrong
                return {"content": "Không nhận được phản hồi hợp lệ từ AI. Vui lòng thử lại.", "tools_used": tools_used}
        
        if tools_used:
            # Return what we have even if incomplete
            summary = f"Đã phân tích {len(tools_used)} công cụ nhưng chưa hoàn tất. "
            summary += f"Dưới đây là kết quả đã thu thập:\n\n"
            return {"content": summary + "\n\n".join([str(t) for t in tools_used]), "tools_used": tools_used}
        return {"content": "Hệ thống đã phân tích nhưng cần tạm ngừng vòng lặp công cụ.", "tools_used": tools_used}
        
    except Exception as e:
        error_msg = str(e).lower()
        if "image" in error_msg and "cannot read" in error_msg:
            return {"content": "Xin lỗi, model này không hỗ trợ xử lý hình ảnh. Tôi chỉ có thể phân tích văn bản. Bạn có thể mô tả nội dung bạn muốn hỏi bằng text không?"}
        return {"error": f"Lỗi liên kết HuggingFace API: {str(e)}"}