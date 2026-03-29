import streamlit as st
from ai_core.groq_client import chat_with_tools
from ai_core.hf_client import chat_with_hf
from ai_core.hybrid_client import chat_hybrid

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
    
    # Check for available API keys
    groq_key = os.getenv("GROQ_API_KEY")
    hf_key = os.getenv("HF_API_KEY")
    
    # Model selector - Hybrid is default (best for multiple stocks)
    ai_options = []
    default_index = 0  # Default to Hybrid if available
    
    if groq_key:
        ai_options.append("🔷 Hybrid (Tools + Groq) - Khuyến nghị")
        ai_options.append("Groq (Llama 3.3 - 70B)")
    if hf_key:
        ai_options.append("HuggingFace (Llama 3.1 - 8B)")
    
    if not ai_options:
        st.error("⚠️ Lỗi: Không tìm thấy API Key nào trong file .env.")
        st.info("Hãy thêm GROQ_API_KEY hoặc HF_API_KEY vào .env và tải lại.")
        return
    
    selected_ai = st.selectbox("AI Model", ai_options, index=default_index, label_visibility="collapsed")
    
    # Show model info
    if "Hybrid" in selected_ai:
        st.info(f"🔷 **Hybrid Approach** - Gọi tools trực tiếp + LLM tổng hợp. TỐT NHẤT cho phân tích nhiều cổ phiếu, tránh loop/repetition.")
    elif "Groq" in selected_ai:
        st.info(f"✅ **Groq (Llama 3.3 70B)** - Miễn phí không giới hạn, context 128K tokens")
    elif "HuggingFace" in selected_ai:
        st.info(f"⚠️ **HuggingFace (Llama 3.1 8B)** - Miễn phí 500 requests/ngày, context nhỏ ~8K (hạn chế)")
    
    st.markdown("---")
    
    chat_container = st.container(height=500, border=True)
    
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
                            if isinstance(tool, dict):
                                st.code(f"{tool.get('name', 'unknown')}({tool.get('args', {})})")
                            else:
                                st.code(tool)
    
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
                    # Use selected AI
                    if "Hybrid" in selected_ai:
                        result = chat_hybrid(prompt, st.session_state.ai_chat_messages)
                    elif "HuggingFace" in selected_ai:
                        result = chat_with_hf(st.session_state.ai_chat_messages)
                    else:
                        result = chat_with_tools(st.session_state.ai_chat_messages)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown(result["content"])
                        if result.get("tools_used"):
                            with st.expander("🔧 Tools đã dùng:"):
                                if isinstance(result["tools_used"][0], dict):
                                    for tool in result["tools_used"]:
                                        st.code(f"{tool.get('name', 'unknown')}({tool.get('args', {})})")
                                else:
                                    for tool in result["tools_used"]:
                                        st.code(tool)
                        
                        # Save Display context
                        st.session_state.ai_chat_display.append({
                            "role": "assistant",
                            "content": result["content"],
                            "tools_used": result.get("tools_used", [])
                        })
