"""
Hybrid AI Client - Kết hợp gọi tools trực tiếp + LLM tổng hợp
=============================================================

Luồng hoạt động:
1. Parse user query để xác định cần gọi tools nào
2. Gọi tools trực tiếp (không qua LLM function calling)
3. Gửi kết quả tools cho LLM để tổng hợp + phân tích + thêm kiến thức
4. Trả về response hoàn chỉnh

Ưu điểm:
- Không bị loop/repetition
- Gọi nhiều tools cùng lúc
- LLM chỉ tổng hợp, không cần function calling mạnh
- Có thể thêm kiến thức bên ngoài vào response
"""

import os
import json
import re
from dotenv import load_dotenv
from ai_core.tools import AVAILABLE_TOOLS

# Load .env for local development
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
    """Get Groq client - recommended for synthesis."""
    try:
        from groq import Groq
        
        # Try Streamlit secrets first (for Cloud), then .env
        secrets = get_secrets()
        api_key = secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
        if api_key:
            return Groq(api_key=api_key)
    except:
        pass
    return None

def get_hf_client():
    """Get HuggingFace client as backup."""
    try:
        from huggingface_hub import InferenceClient
        
        secrets = get_secrets()
        api_key = secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY")
        
        if api_key:
            return InferenceClient(provider="cerebras", model="meta-llama/Llama-3.1-8B-Instruct", api_key=api_key)
    except:
        pass
    return None

def get_all_stock_symbols() -> set:
    """Get all available stock symbols from all exchanges."""
    try:
        from stock_data.stock_data import get_stock_symbols
        
        all_symbols = set()
        
        # Get symbols from all exchanges
        for exchange in ["HOSE", "HNX", "UPCOM"]:
            try:
                symbols = get_stock_symbols(exchange=exchange)
                if symbols:
                    all_symbols.update(symbols)
            except:
                pass
        
        return all_symbols
    except:
        return set()

def extract_symbols_from_query(query: str) -> list:
    """Extract stock symbols from user query."""
    query_upper = query.upper()
    query_lower = query.lower()
    
    # Get all available stocks from real data
    all_stocks = get_all_stock_symbols()
    
    # Also include common stocks as backup
    common_stocks = {
        "FPT", "HPG", "VCB", "VNM", "BID", "VPB", "SSI", "MSN", "MWG", "PNJ",
        "ACB", "STB", "CTG", "TPB", "MBB", "ABB", "SSB", "EIB", "OCB", "VBB",
        "VIC", "VHM", "VIN", "ROS", "KDC", "SAB", "BMP", "MSN", "MCH", "QNS",
        "DPM", "PET", "PLX", "PVB", "PVD", "PVT", "GAS", "BSR", "KBT",
        "DXG", "KDH", "NVT", "KVR", "DIG", "FLC", "HAH", "BHH", "DRC",
        "GVR", "BCM", "BID", "CTG", "VPB", "TPB", "MBB", "EIB", "STB", "BAB",
        "NLG", "DGW", "WTC", "VRG", "SVC", "CII", "BCC", "BCG", "CEE", "CEN",
        "CGV", "CIC", "CLW", "CMG", "CNG", "CSM", "CTG", "DBC", "DHC", "DHT",
        "DIG", "DLG", "DPM", "DRC", "DSG", "DXG", "EIB", "ELC", "EVG", "FLC",
        "FPT", "GAS", "GEX", "GMD", "GTN", "HAG", "HAH", "HBC", "HDB", "HDG",
        "HPG", "HSG", "HT1", "HVN", "IBD", "ICG", "IDI", "KDC", "KHB", "KHL",
        "KSB", "LHG", "LSS", "MBB", "MCC", "MSN", "MWG", "NKD", "NLG", "NSC",
        "NT2", "OCB", "PAC", "PAN", "PC1", "PET", "PLX", "PNJ", "PVD", "PVT",
        "QCG", "RCL", "SAB", "SAF", "SCM", "SHB", "SJD", "SJS", "SKG", "SMG",
        "SSI", "STB", "TCH", "TCM", "TCO", "TCR", "TGG", "THG", "TKU", "TLH",
        "TPB", "TRA", "TTB", "TWA", "VCB", "VCI", "VDS", "VGC", "VGV", "VHM",
        "VIC", "VIX", "VND", "VNE", "VNM", "VPB", "VPD", "VPI", "VRE", "VSC",
        "VSH", "VTM", "WCS", "VNINDEX", "HNX", "UPCOM"
    }
    
    # Merge both sets
    known_stocks = all_stocks | common_stocks
    
    found_symbols = []
    for symbol in known_stocks:
        # Skip common words that aren't stock symbols
        skip_words = {"TIN", "TINH", "GIAM", "TANG", "MUA", "BAN", "CO", "KHONG", "VA", "VAI", "THI", "NHUNG", "MAU", "CAI", "NAY", "OI", "O", "U", "A", "AN"}
        if symbol in skip_words:
            continue
        if len(symbol) >= 2 and symbol in query_upper:
            found_symbols.append(symbol)
    
    # Fallback: if no symbols found but query mentions market
    if not found_symbols:
        market_keywords = ["vnindex", "thị trường", "market", "tổng quan", "chung", "hôm nay", "tình hình"]
        if any(kw in query_lower for kw in market_keywords):
            found_symbols = ["VNINDEX"]
    
    return found_symbols

def determine_tools_needed(query: str, symbols: list) -> list:
    """Determine which tools to call based on query and symbols."""
    query_lower = query.lower()
    tools_to_call = []
    
    has_market = any(word in query_lower for word in ["thị trường", "market", "tổng quan", "chung"])
    has_sentiment = any(word in query_lower for word in ["tâm lý", "sentiment", "xu hướng"])
    has_investor = any(word in query_lower for word in ["nhà đầu tư", "dòng tiền", "investor", "mua", "bán"])
    has_valuation = any(word in query_lower for word in ["định giá", "p/e", "p/b", "valuation"])
    has_stock = any(word in query_lower for word in ["cổ phiếu", "stock", "phân tích"]) and symbols
    has_recommend = any(word in query_lower for word in ["khuyến nghị", "recommend", "nên", "nhận định", "đánh giá"])
    has_price = any(word in query_lower for word in ["giá", "price"])
    
    if has_market or (not symbols and not has_sentiment and not has_investor and not has_valuation):
        tools_to_call.append(("menu_market_overview", {"days": 180}))
    
    if has_sentiment:
        tools_to_call.append(("tool_market_sentiment_full", {"days": 180}))
    
    if has_investor:
        for sym in symbols if symbols else ["VNINDEX"]:
            tools_to_call.append(("tool_investor_classification", {"symbol": sym, "frequency": "Daily"}))
    
    if has_valuation:
        for sym in symbols if symbols else ["VNINDEX"]:
            tools_to_call.append(("tool_valuation", {"symbol": sym, "days": 365}))
    
    if has_price and symbols:
        for sym in symbols:
            tools_to_call.append(("tool_stock_price", {"symbol": sym, "days": 30}))
    
    if has_stock and symbols:
        for sym in symbols:
            tools_to_call.append(("menu_stock_analysis", {"symbol": sym, "days": 180}))
    
    if has_recommend:
        for sym in symbols if symbols else ["VNINDEX"]:
            query_type = "stock" if sym != "VNINDEX" else "market"
            tools_to_call.append(("tool_recommendation", {"query_type": query_type, "symbol": sym, "days": 180}))
    
    if not tools_to_call:
        if symbols:
            for sym in symbols:
                tools_to_call.append(("menu_stock_analysis", {"symbol": sym, "days": 180}))
        else:
            tools_to_call.append(("menu_market_overview", {"days": 180}))
    
    return tools_to_call

def execute_tools(tools_to_call: list) -> str:
    """Execute tools directly and return combined results."""
    results = []
    
    for tool_name, tool_args in tools_to_call:
        if tool_name in AVAILABLE_TOOLS:
            try:
                result = AVAILABLE_TOOLS[tool_name](**tool_args)
                results.append(f"\n{'='*50}\n📌 {tool_name.replace('_', ' ').title()}\n{'='*50}\n{result}")
            except Exception as e:
                results.append(f"\n❌ Lỗi gọi {tool_name}: {str(e)}")
        else:
            results.append(f"\n❌ Tool {tool_name} không tồn tại")
    
    return "\n".join(results)

def synthesize_with_llm(query: str, tool_results: str) -> str:
    """Use LLM to synthesize tool results and add external knowledge."""
    
    client = get_groq_client() or get_hf_client()
    
    if not client:
        return tool_results + "\n\n⚠️ Không có LLM available để tổng hợp. Đây là kết quả trực tiếp từ tools."
    
    synthesis_prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam.

Dưới đây là KẾT QUẢ PHÂN TÍCH từ các công cụ dữ liệu:
---
{tool_results}
---

Câu hỏi của user: {query}

NHIỆM VỤ:
1. Đọc và hiểu kết quả từ các công cụ
2. Tổng hợp thành báo cáo phân tích mạch lạc, dễ đọc
3. Thêm KIẾN THỨC BÊN NGOÀI về:
   - Bối cảnh kinh tế vĩ mô hiện tại (lãi suất, tỷ giá, lạm phát)
   - Xu hướng ngành nếu có
   - Các yếu tố cần lưu ý
4. Đưa ra NHẬN ĐỊNH và khuyến nghị (nếu phù hợp)

Lưu ý quan trọng:
- KHÔNG bịa đặt số liệu - chỉ dùng data từ kết quả tools
- Trả lời bằng tiếng Việt, dùng Markdown để format
- Nếu thiếu dữ liệu, nói rõ "Không có đủ dữ liệu để phân tích"
- Kết quả phải TỰ NHIÊN như con người viết, KHÔNG phải dạng JSON/tool response"""

    try:
        if hasattr(client, 'chat'):  # Groq client
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=4096,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:  # HuggingFace client
            response = client.chat_completion(
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=4096,
                temperature=0.7
            )
            return response.choices[0].message.content or "Không có phản hồi từ LLM"
    except Exception as e:
        return tool_results + f"\n\n⚠️ Lỗi khi tổng hợp với LLM: {str(e)}\n\nĐây là kết quả trực tiếp từ tools."


def chat_hybrid(query: str) -> dict:
    """
    Main function for hybrid chat - directly call tools + use LLM to synthesize.
    
    Args:
        query: User's question/prompt
        
    Returns:
        dict with "content" and "tools_used" keys
    """
    tools_used = []
    
    try:
        # Step 1: Extract symbols from query
        symbols = extract_symbols_from_query(query)
        
        # Step 2: Determine which tools to call
        tools_to_call = determine_tools_needed(query, symbols)
        tools_used = [f"{tool}({args})" for tool, args in tools_to_call]
        
        # Step 3: Execute tools directly (not through LLM function calling)
        tool_results = execute_tools(tools_to_call)
        
        # Step 4: Use LLM to synthesize results + add external knowledge
        final_response = synthesize_with_llm(query, tool_results)
        
        return {
            "content": final_response,
            "tools_used": tools_used
        }
        
    except Exception as e:
        return {
            "content": f"❌ Lỗi: {str(e)}",
            "tools_used": tools_used
        }