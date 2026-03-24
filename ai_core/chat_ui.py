import streamlit as st
from ai_core.groq_client import chat_with_tools

def init_chat_session():
    # Lưu tin nhắn để nạp vào API (cấu trúc Role/Tool/Call của LLM)
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []
    # Lưu tin nhắn rút gọn để hiển thị trên web
    if "ai_chat_display" not in st.session_state:
        st.session_state.ai_chat_display = []

@st.fragment
def render_ai_chat():
    init_chat_session()
    
    st.title("🤖 AI Trợ Lý Phân Tích")
    st.markdown("---")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Lỗi: Không tìm thấy `GROQ_API_KEY` trong file .env.")
        st.info("Hãy thêm dòng `GROQ_API_KEY=your_key` vào .env và tải lại.")
        return
    
    chat_container = st.container(height=600, border=True)
    
    with chat_container:
        if not st.session_state.ai_chat_display:
            st.info("👋 Chào bạn! Tôi là trợ lý ảo phân tích chứng khoán. Bạn có thể hỏi tôi về giá cổ phiếu, xu hướng thị trường (VD: Lịch sử giá mã HPG...).")
        
        # In các lịch sử có sẵn
        for msg in st.session_state.ai_chat_display:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tools_used"):
                    with st.expander("🔧 Tools đã dùng:"):
                        for tool in msg["tools_used"]:
                            st.code(f"{tool['name']}({tool['args']})")
    
    # Text input area
    prompt = st.chat_input("Hỏi AI về cổ phiếu...")
    if prompt:
        # Ghi và In user message
        st.session_state.ai_chat_display.append({"role": "user", "content": prompt})
        st.session_state.ai_chat_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Tạo Assistant stream spinner
            with st.chat_message("assistant"):
                with st.spinner("Đang tìm dữ liệu và phân tích..."):
                    # Pass toàn bộ memory context to the agent
                    result = chat_with_tools(st.session_state.ai_chat_messages)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown(result["content"])
                        if result.get("tools_used"):
                            with st.expander("🔧 Tools đã dùng:"):
                                for tool in result["tools_used"]:
                                    st.code(f"{tool['name']}({tool['args']})")
                        
                        # Save Display context
                        st.session_state.ai_chat_display.append({
                            "role": "assistant",
                            "content": result["content"],
                            "tools_used": result.get("tools_used", [])
                        })
