import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from ai_core.prompts import SYSTEM_PROMPT
from ai_core.tools import AVAILABLE_TOOLS, TOOLS_SCHEMA

load_dotenv()

def get_claude_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)

def chat_with_tools(messages, model="claude-sonnet-4-20250514"):
    """
    Tiếp nhận list dictionary `messages` (role: user/assistant/tool, content: text...).
    Xử lý vòng lặp gọi Tool của LLM cho đến khi hoàn tất hoặc tối đa 3 vòng lặp.
    """
    client = get_claude_client()
    if not client:
        return {"error": "ANTHROPIC_API_KEY chưa được cấu hình trong file .env."}
    
    # Kiểm tra Message đầu tiên: Nếu chưa có System Prompt thì thêm
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    tools_used = []
    max_loops = 3 # Tránh infinite loop
    loops = 0
    
    try:
        while loops < max_loops:
            # Chuyển đổi tools schema sang định dạng Claude
            claude_tools = []
            for tool in TOOLS_SCHEMA:
                claude_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
                tools=claude_tools,
            )
            
            # Ghi nhận tin nhắn asst trả lời (hoặc yêu cầu gọi tool)
            assistant_message = {"role": "assistant", "content": response.content[0].text if response.content else ""}
            messages.append(assistant_message)
            
            # Nếu LLM không gọi Tool nào -> return user final answer
            if not response.stop_reason == "tool_use":
                return {"content": assistant_message["content"], "tools_used": tools_used}
            
            # Nếu LLM có gọi Tools -> Resolve các tool calls
            for content_block in response.content:
                if content_block.type == "tool_use":
                    function_name = content_block.name
                    function_args = content_block.input
                    
                    # Lưu log UI
                    tools_used.append({
                        "name": function_name,
                        "args": function_args
                    })
                    
                    # Chạy python code cục bộ
                    if function_name in AVAILABLE_TOOLS:
                        function_to_call = AVAILABLE_TOOLS[function_name]
                        function_response = function_to_call(**function_args)
                    else:
                        function_response = f"Lỗi: Function {function_name} không tồn tại!"
                    
                    # Ghi nhận kết quả Tool
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(function_response),
                            "tool_use_id": content_block.id,
                        }
                    )
            
            # Tăng biến đếm và loop lại: Claude Model sẽ đọc nội dung trả về của Tools để phản hồi.
            loops += 1

        # NẾU max limit: return the exact thing (prevent timeout/rate)
        return {"content": "Hệ thống đã phân tích nhưng cần tạm ngừng vòng lặp công cụ. Xin thử lại câu khác hoặc chi tiết hơn.", "tools_used": tools_used}
        
    except Exception as e:
        return {"error": f"Lỗi liên kết Claude API: {str(e)}"}
