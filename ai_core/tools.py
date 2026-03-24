"""
AI Tools cho Dashboard - Đồng nhất với cấu trúc menu Dashboard
===============================================================

CẤU TRÚC DASHBOARD:
- Trang chủ: Tổng quan thị trường
- Thị trường:
  - Tâm lý thị trường (Market Sentiment)
  - Phân loại nhà đầu tư (Investor Classification)  
  - Định giá (Valuation)
  - Phân loại giao dịch (Trading Classification)
- Cổ phiếu: Chi tiết từng mã cổ phiếu

TOOLS DESIGN:
- Mỗi topic-based tool gọi nhiều underlying functions cùng lúc
- Khi user hỏi về một chủ đề, AI sẽ tự động gọi tất cả tools liên quan
"""

import json
import pandas as pd
from stock_data.stock_data import get_stock_history, get_stock_symbols, investor_type
from market_sentiment.sentiment import (
    sentiment, volatility, high_low_index, 
    bpi, ma, market_breadth
)
from valuation.valuation import get_pb_pe
from datetime import datetime, timedelta
from typing import Optional

# ============================================================
# LOW-LEVEL FUNCTIONS (các hàm cơ bản, có thể gọi riêng)
# ============================================================

def tool_get_stock_data(symbol: str, count_back: int = 30):
    """
    Lấy dữ liệu giá OHLCV của một mã cổ phiếu.
    """
    try:
        df = get_stock_history(symbol=symbol.upper(), period='day', count_back=count_back)
        if df.empty:
            return f"Không tìm thấy dữ liệu giá cho mã {symbol} hoặc xảy ra lỗi API."
        
        latest = df.iloc[-1]
        earliest = df.iloc[0]
        summary = f"Dữ liệu giá mã {symbol} từ {earliest['time']} đến {latest['time']}:\n"
        summary += f"Tín hiệu mới nhất ({latest['time']}): Mở cửa: {latest['open']:,}, Cao: {latest['high']:,}, Thấp: {latest['low']:,}, Đóng: {latest['close']:,}, KL: {latest['volume']:,}\n"
        
        last_5 = df.tail(5)
        last_5_str = ""
        for _, row in last_5.iterrows():
            last_5_str += f"- Ngày {row['time']}: {row['close']} (KL: {row['volume']:,})\n"
            
        return f"{summary}\nChi tiết 5 phiên giao dịch gần nhất:\n{last_5_str}"
        
    except Exception as e:
        return f"Lỗi gọi tool get_stock_data: {str(e)}"

def tool_get_stock_symbols(exchange: str = 'HOSE'):
    """
    Lấy danh sách mã chứng khoán đang niêm yết trên sàn.
    """
    try:
        symbols = get_stock_symbols(exchange=exchange.upper())
        if not symbols:
            return f"Không tìm thấy mã nào niêm yết ở sàn {exchange}."
        
        shown_symbols = symbols[:80]
        total = len(symbols)
        return (f"Tổng cộng có {total} mã cổ phiếu trên sàn {exchange}. "
                f"Danh sách hiển thị một phần: {', '.join(shown_symbols)}...")
    except Exception as e:
        return f"Lỗi gọi tool get_stock_symbols: {str(e)}"

def tool_get_market_sentiment(days: int = 180):
    """
    Lấy dữ liệu Tâm lý thị trường (Market Sentiment).
    Bao gồm: short-term, long-term sentiment và VNINDEX.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = sentiment(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu tâm lý thị trường. Có thể cookie đã hết hạn."
        
        latest = df.iloc[-1]
        
        summary = f"📊 TÂM LÝ THỊ TRƯỜNG (Ngày: {latest['time']})\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• Short-term: {latest.get('short', 'N/A')}\n"
        summary += f"• Long-term: {latest.get('long', 'N/A')}\n"
        summary += f"• VNINDEX: {latest.get('close', 'N/A'):,.2f}\n"
        
        # Xu hướng
        if len(df) >= 5:
            avg_short = df.tail(5)['short'].mean()
            avg_long = df.tail(5)['long'].mean()
            summary += f"\n📈 Xu hướng 5 ngày:\n"
            summary += f"• Short-term trung bình: {avg_short:.2f}\n"
            summary += f"• Long-term trung bình: {avg_long:.2f}\n"
            
            if avg_short > 80:
                summary += "⚠️ Cảnh báo: EXTREME GREED\n"
            elif avg_short > 60:
                summary += "📈 Tâm lý bullish\n"
            elif avg_short < 20:
                summary += "⚠️ Cảnh báo: EXTREME FEAR\n"
            elif avg_short < 40:
                summary += "📉 Tâm lý bearish\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_volatility(symbol: str = 'VNINDEX', days: int = 252, forecast_days: int = 10):
    """
    Phân tích biến động (Volatility) sử dụng GJR-GARCH.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = volatility(symbol=symbol, end_date=end_date, countback=days, forecast_days=forecast_days)
        
        if df is None or df.empty:
            return f"Không có dữ liệu biến động cho {symbol}."
        
        latest = df.iloc[-1]
        
        summary = f"📈 BIẾN ĐỘNG THỊ TRƯỜNG (GJR-GARCH)\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• Giá đóng cửa: {latest.get('close', 'N/A'):,.2f}\n"
        summary += f"• Volatility hiện tại: {latest.get('volatility', 'N/A'):.4f}\n"
        
        vol = latest.get('volatility', 0)
        if vol > 0.03:
            summary += "\n🔴 BIẾN ĐỘNG MẠNH (>0.03): Rủi ro cao\n"
        elif vol > 0.02:
            summary += "\n🟡 BIẾN ĐỘNG TRUNG BÌNH (0.02-0.03)\n"
        else:
            summary += "\n🟢 BIẾN ĐỘNG THẤP: Ổn định\n"
        
        # Dự báo
        forecast_data = df[df['close'].isna()]
        if not forecast_data.empty:
            summary += f"\n📊 Dự báo {len(forecast_data)} ngày:\n"
            for _, row in forecast_data.tail(3).iterrows():
                summary += f"  {row['time']}: {row.get('volatility', 'N/A'):.4f}\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_high_low_index(days: int = 180):
    """
    High-Low Index - vùng quá mua/quá bán.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = high_low_index(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu High-Low Index."
        
        latest = df.iloc[-1]
        
        summary = f"📉 HIGH-LOW INDEX\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• HL Index: {latest.get('hl_index', 'N/A'):.2f}\n"
        summary += f"• Đỉnh cao mới: {latest.get('peak_count', 'N/A')} mã\n"
        summary += f"• Đáy thấp mới: {latest.get('trough_count', 'N/A')} mã\n"
        
        hl = latest.get('hl_index', 50)
        if hl > 70:
            summary += "\n🔴 OVERBOUGHT (>70): Quá mua\n"
        elif hl < 30:
            summary += "\n🟢 OVERSOLD (<30): Quá bán\n"
        else:
            summary += "\n🟡 Trung tính\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_bpi(days: int = 180):
    """
    Bullish Percent Index - xu hướng bullish/bearish.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = bpi(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu BPI."
        
        latest = df.iloc[-1]
        
        summary = f"📊 BULLISH PERCENT INDEX (BPI)\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• BPI: {latest.get('bpi', 'N/A'):.2f}%\n"
        summary += f"• Bullish: {latest.get('count', 'N/A')} / {latest.get('total', 'N/A')} mã\n"
        
        bpi_val = latest.get('bpi', 50)
        if bpi_val > 70:
            summary += "\n🔴 BULLISH MẠNH (>70%)\n"
        elif bpi_val > 60:
            summary += "\n📈 BULLISH (60-70%)\n"
        elif bpi_val < 30:
            summary += "\n🟢 BEARISH MẠNH (<30%)\n"
        elif bpi_val < 40:
            summary += "\n📉 BEARISH (30-40%)\n"
        else:
            summary += "\n➡️ Trung tính\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_moving_averages(days: int = 365):
    """
    Moving Averages MA50, MA200 - xu hướng dài hạn.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = ma(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu MA."
        
        latest = df.iloc[-1]
        
        summary = f"📊 MOVING AVERAGES (VNINDEX)\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• Giá: {latest.get('close', 'N/A'):,.2f}\n"
        summary += f"• MA50: {latest.get('ma50', 'N/A'):,.2f}\n"
        summary += f"• MA200: {latest.get('ma200', 'N/A'):,.2f}\n"
        
        close = latest.get('close', 0)
        ma50 = latest.get('ma50', 0)
        ma200 = latest.get('ma200', 0)
        
        if close > ma50 > ma200:
            summary += "\n📈 GOLDEN CROSS: Xu hướng tăng\n"
        elif close < ma50 < ma200:
            summary += "\n📉 DEATH CROSS: Xu hướng giảm\n"
        else:
            summary += "\n➡️ Xu hướng không rõ ràng\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_market_breadth(days: int = 180):
    """
    Market Breadth - độ rộng thị trường.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = market_breadth(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu Market Breadth."
        
        latest = df.iloc[-1]
        
        # Tìm các cột có trong dữ liệu
        above_col = 'above' if 'above' in df.columns else None
        below_col = 'below' if 'below' in df.columns else None
        
        summary = f"📊 MARKET BREADTH\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        if above_col and below_col:
            summary += f"• Trên EMA50: {latest.get(above_col, 'N/A')} mã\n"
            summary += f"• Dưới EMA50: {latest.get(below_col, 'N/A')} mã\n"
        
        if 'breadth' in latest:
            summary += f"• Breadth: {latest.get('breadth', 'N/A'):.2f}%\n"
        if 'vnindex' in latest:
            summary += f"• VNINDEX: {latest.get('vnindex', 'N/A'):,.2f}\n"
        
        # Đánh giá
        if 'breadth' in latest:
            breadth = latest.get('breadth', 50)
            if breadth > 70:
                summary += "\n🔴 Thị trường RỘNG MẠNH\n"
            elif breadth > 60:
                summary += "\n📈 Thị trường RỘNG\n"
            elif breadth < 30:
                summary += "\n🟢 Thị trường THU HẸP\n"
            elif breadth < 40:
                summary += "\n📉 Thị trường THU HẸP\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_investor_type(symbol: str = 'VNINDEX', frequency: str = 'Daily'):
    """
    Phân loại nhà đầu tư - dòng tiền.
    """
    try:
        df = investor_type(symbol=symbol, frequency=frequency)
        if df.empty:
            return f"Không có dữ liệu nhà đầu tư cho {symbol}."
        
        latest = df.iloc[-1]
        
        summary = f"👥 PHÂN LOẠI NHÀ ĐẦU TƯ ({symbol})\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Map các cột
        cols_map = {
            'foreignNetValue': 'Nước ngoài',
            'proprietaryNetValue': 'Tự doanh', 
            'localIndividualNetValue': 'Cá nhân trong nước',
            'localInstitutionalNetValue': 'Tổ chức trong nước',
            'foreignIndividualNetValue': 'Cá nhân NN',
            'foreignInstitutionalNetValue': 'Tổ chức NN'
        }
        
        total_net = 0
        for col, name in cols_map.items():
            if col in latest:
                val = latest.get(col, 0)
                total_net += val
                sign = "+" if val > 0 else ""
                summary += f"• {name}: {sign}{val:,.0f} tỷ\n"
        
        summary += f"\n💰 Tổng ròng: {'+' if total_net > 0 else ''}{total_net:,.0f} tỷ\n"
        
        if total_net > 1000:
            summary += "📈 Dòng tiền vào mạnh\n"
        elif total_net < -1000:
            summary += "📉 Dòng tiền ra mạnh\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"

def tool_get_valuation(symbol: str, days: int = 365):
    """
    Định giá P/E và P/B.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = get_pb_pe(symbol=symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"Không có dữ liệu định giá cho {symbol}."
        
        latest = df.iloc[-1]
        
        summary = f"💰 ĐỊNH GIÁ {symbol}\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"• Giá: {latest.get('price', 'N/A'):,.2f}\n"
        
        pe = latest.get('pe', 0)
        pb = latest.get('pb', 0)
        
        if pe and pe > 0:
            summary += f"• P/E: {pe:.2f}\n"
            if pe < 10:
                summary += "  🟢 Có vẻ undervalued\n"
            elif pe > 30:
                summary += "  🔴 Có vẻ overvalued\n"
        
        if pb and pb > 0:
            summary += f"• P/B: {pb:.2f}\n"
            if pb < 1:
                summary += "  🟢 Có vẻ undervalued\n"
            elif pb > 3:
                summary += "  🔴 Có vẻ overvalued\n"
        
        return summary
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# TOPIC-BASED FUNCTIONS (gọi nhiều tools cùng lúc)
# ============================================================

def analyze_market_sentiment(days: int = 180):
    """
    PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG
    =============================
    Gọi các tools:
    - tool_get_market_sentiment (tâm lý ngắn/dài hạn)
    - tool_get_volatility (biến động)
    - tool_get_high_low_index (vùng quá mua/bán)
    - tool_get_bpi (xu hướng bullish)
    - tool_get_moving_averages (xu hướng dài hạn)
    - tool_get_market_breadth (độ rộng)
    """
    results = []
    
    results.append("=" * 50)
    results.append("📊 PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG")
    results.append("=" * 50)
    
    # 1. Tâm lý thị trường
    results.append("\n" + tool_get_market_sentiment(days))
    
    # 2. Biến động
    results.append("\n" + tool_get_volatility(days=252))
    
    # 3. High-Low Index
    results.append("\n" + tool_get_high_low_index(days))
    
    # 4. BPI
    results.append("\n" + tool_get_bpi(days))
    
    # 5. Moving Averages
    results.append("\n" + tool_get_moving_averages(days))
    
    # 6. Market Breadth
    results.append("\n" + tool_get_market_breadth(days))
    
    return "\n".join(results)


def analyze_investor_type(symbol: str = 'VNINDEX', frequency: str = 'Daily'):
    """
    PHÂN TÍCH PHÂN LOẠI NHÀ ĐẦU TƯ
    ==============================
    Gọi các tools:
    - tool_get_investor_type (dòng tiền theo loại)
    """
    results = []
    
    results.append("=" * 50)
    results.append("👥 PHÂN TÍCH PHÂN LOẠI NHÀ ĐẦU TƯ")
    results.append("=" * 50)
    
    results.append("\n" + tool_get_investor_type(symbol, frequency))
    
    return "\n".join(results)


def analyze_valuation(symbol: str, days: int = 365):
    """
    PHÂN TÍCH ĐỊNH GIÁ
    ===================
    Gọi các tools:
    - tool_get_valuation (P/E, P/B)
    """
    results = []
    
    results.append("=" * 50)
    results.append(f"💰 PHÂN TÍCH ĐỊNH GIÁ {symbol}")
    results.append("=" * 50)
    
    results.append("\n" + tool_get_valuation(symbol, days))
    
    return "\n".join(results)


def analyze_stock(symbol: str, price_days: int = 30, valuation_days: int = 365):
    """
    PHÂN TÍCH CỔ PHIẾU
    ===================
    Gọi các tools:
    - tool_get_stock_data (giá OHLCV)
    - tool_get_valuation (P/E, P/B)
    """
    results = []
    
    results.append("=" * 50)
    results.append(f"📈 PHÂN TÍCH CỔ PHIẾU {symbol}")
    results.append("=" * 50)
    
    # Giá cổ phiếu
    results.append("\n" + tool_get_stock_data(symbol, price_days))
    
    # Định giá
    results.append("\n" + tool_get_valuation(symbol, valuation_days))
    
    return "\n".join(results)


def analyze_market_overview(days: int = 180):
    """
    TỔNG QUAN THỊ TRƯỜNG
    ====================
    Gọi TẤT CẢ các tools từ menu "Thị trường":
    - Tâm lý thị trường (sentiment + volatility + HL Index + BPI + MA + Breadth)
    - Phân loại nhà đầu tư (investor_type)
    - Định giá (P/E, P/B)
    """
    results = []
    
    results.append("=" * 60)
    results.append("🌐 TỔNG QUAN THỊ TRƯỜNG VIỆT NAM")
    results.append("=" * 60)
    
    # 1. Tâm lý thị trường
    results.append("\n" + analyze_market_sentiment(days))
    
    # 2. Phân loại nhà đầu tư
    results.append("\n" + analyze_investor_type())
    
    # 3. Định giá (VNINDEX)
    results.append("\n" + analyze_valuation("VNINDEX", days))
    
    results.append("\n" + "=" * 60)
    results.append("✅ Phân tích hoàn tất")
    results.append("=" * 60)
    
    return "\n".join(results)


# ============================================================
# MAPPING TOOLS + SCHEMA
# ============================================================

AVAILABLE_TOOLS = {
    # Low-level functions
    "get_stock_data": tool_get_stock_data,
    "get_stock_symbols": tool_get_stock_symbols,
    "get_market_sentiment": tool_get_market_sentiment,
    "get_volatility": tool_get_volatility,
    "get_high_low_index": tool_get_high_low_index,
    "get_bpi": tool_get_bpi,
    "get_moving_averages": tool_get_moving_averages,
    "get_market_breadth": tool_get_market_breadth,
    "get_investor_type": tool_get_investor_type,
    "get_valuation": tool_get_valuation,
    
    # Topic-based functions (gọi nhiều tools cùng lúc)
    "analyze_market_sentiment": analyze_market_sentiment,
    "analyze_investor_type": analyze_investor_type,
    "analyze_valuation": analyze_valuation,
    "analyze_stock": analyze_stock,
    "analyze_market_overview": analyze_market_overview
}


TOOLS_SCHEMA = [
    # ========== LOW-LEVEL FUNCTIONS ==========
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Lấy dữ liệu giá OHLCV của một mã cổ phiếu. Dùng khi user hỏi về giá của một cổ phiếu cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (VD: FPT, HPG, VCB, VNINDEX)"},
                    "count_back": {"type": "integer", "description": "Số ngày lấy dữ liệu. Mặc định 30 ngày."}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_symbols",
            "description": "Lấy danh sách mã chứng khoán trên sàn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "description": "Sàn: HOSE, HNX, UPCOM. Mặc định HOSE."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Lấy dữ liệu Tâm lý thị trường (short/long-term sentiment + VNINDEX).",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 180 ngày."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility",
            "description": "Phân tích biến động (Volatility) sử dụng mô hình GJR-GARCH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã. Mặc định VNINDEX."},
                    "days": {"type": "integer", "description": "Số ngày tính volatility. Mặc định 252."},
                    "forecast_days": {"type": "integer", "description": "Số ngày dự báo. Mặc định 10."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_high_low_index",
            "description": "High-Low Index - vùng quá mua/quá bán.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 180."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bpi",
            "description": "Bullish Percent Index - xu hướng bullish/bearish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 180."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_moving_averages",
            "description": "Moving Averages MA50, MA200 - xu hướng dài hạn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 365."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_breadth",
            "description": "Market Breadth - độ rộng thị trường.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 180."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_investor_type",
            "description": "Phân loại nhà đầu tư - dòng tiền theo loại (Foreign, Proprietary, Individual, Institutional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã. Mặc định VNINDEX."},
                    "frequency": {"type": "string", "description": "Tần suất: Daily, Weekly, Monthly."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_valuation",
            "description": "Định giá P/E và P/B của một cổ phiếu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu"},
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 365."}
                },
                "required": ["symbol"]
            }
        }
    },
    
    # ========== TOPIC-BASED FUNCTIONS ==========
    {
        "type": "function",
        "function": {
            "name": "analyze_market_sentiment",
            "description": "📊 PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG - GỌI TẤT CẢ TOOLS LIÊN QUAN: sentiment, volatility, high_low_index, bpi, moving_averages, market_breadth. Dùng khi user hỏi về 'Tâm lý thị trường', 'Market sentiment', 'Xu hướng thị trường'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày phân tích. Mặc định 180."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_investor_type",
            "description": "👥 PHÂN TÍCH PHÂN LOẠI NHÀ ĐẦU TƯ - GỌI investor_type tool. Dùng khi user hỏi về 'Phân loại nhà đầu tư', 'Dòng tiền', 'Ai đang mua/bán'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã. Mặc định VNINDEX."},
                    "frequency": {"type": "string", "description": "Tần suất. Mặc định Daily."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_valuation",
            "description": "💰 PHÂN TÍCH ĐỊNH GIÁ - GỌI get_valuation tool. Dùng khi user hỏi về 'Định giá', 'P/E', 'P/B', ' valuation'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu"},
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 365."}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": "📈 PHÂN TÍCH CỔ PHIẾU - GỌI get_stock_data + get_valuation. Dùng khi user hỏi về một cổ phiếu cụ thể (giá + định giá).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu"},
                    "price_days": {"type": "integer", "description": "Số ngày lấy giá. Mặc định 30."},
                    "valuation_days": {"type": "integer", "description": "Số ngày lấy P/E P/B. Mặc định 365."}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market_overview",
            "description": "🌐 TỔNG QUAN THỊ TRƯỜNG - GỌI TẤT CẢ TOOLS TỪ MENU 'THỊ TRƯỜNG': Tâm lý + Biến động + HL Index + BPI + MA + Breadth + Nhà đầu tư + Định giá. Dùng khi user hỏi về 'Tổng quan thị trường', 'Thị trường hôm nay', 'Market overview'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày phân tích. Mặc định 180."}
                },
                "required": []
            }
        }
    }
]
