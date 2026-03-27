"""
AI Tools cho Dashboard - Đồng nhất với cấu trúc menu Dashboard
==============================================================

CẤU TRÚC DASHBOARD:
- Trang chủ: Tổng quan thị trường
- Thị trường:
  - Tâm lý thị trường (Market Sentiment)
  - Phân loại nhà đầu tư (Investor Classification)  
  - Phân loại giao dịch (Trading Classification - per stock)
- Cổ phiếu:
  - Định giá (Valuation)
  - Phân loại giao dịch

TOOLS DESIGN:
- Mỗi tool mô phỏng một menu/submenu của dashboard
- Tool cấp cao (menu) gọi nhiều tools cấp thấp (submenu) cùng lúc
- Tool cấp thấp (submenu) gọi các functions cơ bản
- AI tự động chọn tool phù hợp dựa trên câu hỏi user
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

DAYS_MAP = {
    "1m": 30,
    "3m": 90, 
    "6m": 180,
    "1y": 365,
    "2y": 730
}

def _parse_days(days_param):
    """Parse days from string or int."""
    if isinstance(days_param, int):
        return days_param
    if isinstance(days_param, str):
        days_param = days_param.lower()
        if days_param in DAYS_MAP:
            return DAYS_MAP[days_param]
        try:
            return int(days_param)
        except:
            pass
    return 180

# ============================================================
# SUBMENU: Tâm lý thị trường (Market Sentiment)
# Gồm: sentiment, volatility, HL index, BPI, MA, market breadth
# ============================================================

def tool_market_sentiment_full(days: int = 180):
    """
    Tâm lý thị trường đầy đủ - gọi TẤT CẢ các chỉ báo:
    - Sentiment (short/long term)
    - Volatility (GJR-GARCH)
    - High-Low Index
    - BPI (Bullish Percent Index)
    - Moving Averages (MA50, MA200)
    - Market Breadth
    """
    try:
        days = _parse_days(days)
        results = []
        
        results.append("=" * 55)
        results.append("📊 TÂM LÝ THỊ TRƯỜNG")
        results.append("=" * 55)
        
        # 1. Sentiment
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df_sentiment = sentiment(start_date, end_date)
        if not df_sentiment.empty:
            latest = df_sentiment.iloc[-1]
            results.append(f"\n🎯 SENTIMENT (Ngày: {latest.get('time', 'N/A')})")
            results.append("-" * 40)
            results.append(f"  Short-term: {latest.get('short', 'N/A')}")
            results.append(f"  Long-term: {latest.get('long', 'N/A')}")
            results.append(f"  VNINDEX: {latest.get('close', 'N/A'):,.2f}")
            
            if len(df_sentiment) >= 5:
                avg_short = df_sentiment.tail(5)['short'].mean()
                if avg_short > 80:
                    results.append("  ⚠️ EXTREME GREED")
                elif avg_short > 60:
                    results.append("  📈 Tâm lý bullish")
                elif avg_short < 20:
                    results.append("  ⚠️ EXTREME FEAR")
                elif avg_short < 40:
                    results.append("  📉 Tâm lý bearish")
        
        # 2. Volatility
        df_vol = volatility(symbol='VNINDEX', end_date=end_date, countback=min(days, 252), forecast_days=5)
        if df_vol is not None and not df_vol.empty:
            latest = df_vol.iloc[0] if df_vol['close'].notna().iloc[0] else df_vol.iloc[-1]
            vol = latest.get('volatility', 0)
            results.append(f"\n📉 BIẾN ĐỘNG (GJR-GARCH)")
            results.append("-" * 40)
            results.append(f"  Volatility: {vol:.4f}")
            if vol > 0.03:
                results.append("  🔴 BIẾN ĐỘNG MẠNH - Rủi ro cao")
            elif vol > 0.02:
                results.append("  🟡 BIẾN ĐỘNG TRUNG BÌNH")
            else:
                results.append("  🟢 BIẾN ĐỘNG THẤP - Ổn định")
        
        # 3. High-Low Index
        df_hl = high_low_index(start_date, end_date)
        if not df_hl.empty:
            latest = df_hl.iloc[-1]
            hl = latest.get('hl_index', 50)
            results.append(f"\n📊 HIGH-LOW INDEX")
            results.append("-" * 40)
            results.append(f"  HL Index: {hl:.2f}")
            results.append(f"  Đỉnh cao mới: {latest.get('peak_count', 'N/A')} mã")
            results.append(f"  Đáy thấp mới: {latest.get('trough_count', 'N/A')} mã")
            if hl > 70:
                results.append("  🔴 OVERBOUGHT - Quá mua")
            elif hl < 30:
                results.append("  🟢 OVERSOLD - Quá bán")
            else:
                results.append("  🟡 Trung tính")
        
        # 4. BPI
        df_bpi = bpi(start_date, end_date)
        if not df_bpi.empty:
            latest = df_bpi.iloc[-1]
            bpi_val = latest.get('bpi', 50)
            results.append(f"\n📈 BPI (Bullish Percent Index)")
            results.append("-" * 40)
            results.append(f"  BPI: {bpi_val:.2f}%")
            results.append(f"  Bullish: {latest.get('count', 'N/A')}/{latest.get('total', 'N/A')} mã")
            if bpi_val > 70:
                results.append("  🔴 BULLISH MẠNH")
            elif bpi_val > 60:
                results.append("  📈 BULLISH")
            elif bpi_val < 30:
                results.append("  🟢 BEARISH MẠNH")
            elif bpi_val < 40:
                results.append("  📉 BEARISH")
            else:
                results.append("  ➡️ Trung tính")
        
        # 5. MA
        df_ma = ma(start_date, end_date)
        if not df_ma.empty:
            latest = df_ma.iloc[-1]
            close = latest.get('close', 0)
            ma50 = latest.get('ma50', 0)
            ma200 = latest.get('ma200', 0)
            results.append(f"\n📊 MOVING AVERAGES")
            results.append("-" * 40)
            results.append(f"  Giá: {close:,.2f}")
            results.append(f"  MA50: {ma50:,.2f}")
            results.append(f"  MA200: {ma200:,.2f}")
            if close > ma50 > ma200:
                results.append("  📈 GOLDEN CROSS - Xu hướng tăng")
            elif close < ma50 < ma200:
                results.append("  📉 DEATH CROSS - Xu hướng giảm")
            else:
                results.append("  ➡️ Xu hướng không rõ ràng")
        
        # 6. Market Breadth
        df_breadth = market_breadth(start_date, end_date)
        if not df_breadth.empty:
            latest = df_breadth.iloc[-1]
            breadth = latest.get('breadth', 50)
            results.append(f"\n🌐 MARKET BREADTH")
            results.append("-" * 40)
            if 'above' in latest:
                results.append(f"  Trên EMA50: {latest.get('above', 'N/A')} mã")
            if 'below' in latest:
                results.append(f"  Dưới EMA50: {latest.get('below', 'N/A')} mã")
            results.append(f"  Breadth: {breadth:.2f}%")
            if breadth > 70:
                results.append("  🔴 Thị trường RỘNG MẠNH")
            elif breadth > 60:
                results.append("  📈 Thị trường RỘNG")
            elif breadth < 30:
                results.append("  🟢 Thị trường THU HẸP")
            elif breadth < 40:
                results.append("  📉 Thị trường THU HẸP")
        
        results.append("\n" + "=" * 55)
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi khi lấy dữ liệu tâm lý thị trường: {str(e)}"


def tool_market_sentiment_simple(days: int = 180):
    """
    Tâm lý thị trường đơn giản - chỉ sentiment cơ bản.
    Dùng khi user hỏi đơn giản về tâm lý thị trường.
    """
    try:
        days = _parse_days(days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = sentiment(start_date, end_date)
        if df.empty:
            return "Không có dữ liệu tâm lý thị trường."
        
        latest = df.iloc[-1]
        
        short = latest.get('short', 0)
        long = latest.get('long', 0)
        close = latest.get('close', 0)
        
        result = f"📊 TÂM LÝ THỊ TRƯỜNG ({latest.get('time', '')})"
        result += f"\nVNINDEX: {close:,.2f}"
        result += f"\nShort-term: {short}"
        result += f"\nLong-term: {long}"
        
        if short > 70:
            result += "\n→ Tâm lý bullish, thị trường đang hưng phấn"
        elif short < 30:
            result += "\n→ Tâm lý bearish, thị trường đang hoảng loạn"
        else:
            result += "\n→ Tâm lý trung tính"
        
        return result
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# SUBMENU: Phân loại nhà đầu tư (Investor Classification)
# Dòng tiền: Foreign, Proprietary, Individual, Institutional
# ============================================================

def tool_investor_classification(symbol: str = "VNINDEX", frequency: str = "Daily"):
    """
    Phân loại nhà đầu tư - dòng tiền theo loại:
    - Nước ngoài (Foreign)
    - Tự doanh (Proprietary)
    - Cá nhân trong nước (Local Individual)
    - Tổ chức trong nước (Local Institutional)
    
    Dùng cho VNINDEX hoặc cổ phiếu cụ thể.
    """
    try:
        df = investor_type(symbol=symbol.upper(), frequency=frequency)
        if df.empty:
            return f"Không có dữ liệu phân loại nhà đầu tư cho {symbol}."
        
        latest = df.iloc[-1]
        
        results = []
        results.append(f"👥 PHÂN LOẠI NHÀ ĐẦU TƯ - {symbol}")
        results.append("=" * 45)
        
        cols_map = {
            'foreignNetValue': ('Nước ngoài', '🌍'),
            'proprietaryNetValue': ('Tự doanh', '🏢'),
            'localIndividualNetValue': ('Cá nhân trong nước', '👤'),
            'localInstitutionalNetValue': ('Tổ chức trong nước', '🏛️'),
            'foreignIndividualNetValue': ('Cá nhân NN', '🌏'),
            'foreignInstitutionalNetValue': ('Tổ chức NN', '🌐')
        }
        
        total_net = 0
        for col, (name, icon) in cols_map.items():
            if col in latest:
                val = latest.get(col, 0)
                total_net += val
                sign = "+" if val > 0 else ""
                results.append(f"{icon} {name}: {sign}{val:,.0f} tỷ")
        
        results.append("-" * 45)
        results.append(f"💰 Tổng ròng: {'+' if total_net > 0 else ''}{total_net:,.0f} tỷ")
        
        if total_net > 1000:
            results.append("📈 Dòng tiền vào mạnh")
        elif total_net < -1000:
            results.append("📉 Dòng tiền ra mạnh")
        else:
            results.append("➡️ Dòng tiền ổn định")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# SUBMENU: Định giá (Valuation)
# P/E, P/B của VNINDEX hoặc cổ phiếu cụ thể
# ============================================================

def tool_valuation(symbol: str = "VNINDEX", days: int = 365):
    """
    Định giá P/E và P/B.
    Dùng khi user hỏi về định giá của VNINDEX hoặc cổ phiếu cụ thể.
    """
    try:
        days = _parse_days(days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = get_pb_pe(symbol=symbol.upper(), start_date=start_date, end_date=end_date)
        if df.empty:
            return f"Không có dữ liệu định giá cho {symbol}."
        
        latest = df.iloc[-1]
        
        results = []
        results.append(f"💰 ĐỊNH GIÁ {symbol.upper()}")
        results.append("=" * 45)
        
        price = latest.get('price', 0)
        pe = latest.get('pe', 0)
        pb = latest.get('pb', 0)
        
        results.append(f"Giá: {price:,.2f}")
        
        if pe and pe > 0:
            results.append(f"P/E: {pe:.2f}")
            if pe < 10:
                results.append("  🟢 Có vẻ undervalued")
            elif pe > 30:
                results.append("  🔴 Có vẻ overvalued")
        
        if pb and pb > 0:
            results.append(f"P/B: {pb:.2f}")
            if pb < 1:
                results.append("  🟢 Có vẻ undervalued (P/B < 1)")
            elif pb > 3:
                results.append("  🔴 Có vẻ overvalued")
        
        if symbol.upper() != "VNINDEX":
            if pe and pe > 0 and pb and pb > 0:
                results.append("-" * 45)
                if pe < 10 and pb < 1:
                    results.append("→ Có vẻ đang bị định giá thấp")
                elif pe > 30 or pb > 3:
                    results.append("→ Có vẻ đang bị định giá cao")
                else:
                    results.append("→ Định giá hợp lý")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# SUBMENU: Giá cổ phiếu (Stock Price)
# OHLCV data
# ============================================================

def tool_stock_price(symbol: str, days: int = 30):
    """
    Lấy dữ liệu giá OHLCV của một mã cổ phiếu.
    """
    try:
        days = _parse_days(days)
        df = get_stock_history(symbol=symbol.upper(), period='day', count_back=days)
        if df.empty:
            return f"Không tìm thấy dữ liệu giá cho {symbol}."
        
        latest = df.iloc[-1]
        
        results = []
        results.append(f"📈 GIÁ {symbol.upper()}")
        results.append("=" * 45)
        results.append(f"Ngày: {latest.get('time', 'N/A')}")
        results.append(f"Giá đóng cửa: {latest.get('close', 'N/A'):,.2f}")
        results.append(f"Giá mở cửa: {latest.get('open', 'N/A'):,.2f}")
        results.append(f"Giá cao nhất: {latest.get('high', 'N/A'):,.2f}")
        results.append(f"Giá thấp nhất: {latest.get('low', 'N/A'):,.2f}")
        results.append(f"Khối lượng: {latest.get('volume', 'N/A'):,.0f}")
        
        if len(df) >= 2:
            prev_close = df.iloc[-2].get('close', 0)
            change = latest.get('close', 0) - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0
            sign = "+" if change > 0 else ""
            results.append("-" * 45)
            results.append(f"Thay đổi: {sign}{change:,.2f} ({sign}{change_pct:.2f}%)")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# SUBMENU: Danh sách mã chứng khoán
# ============================================================

def tool_stock_symbols(exchange: str = "HOSE"):
    """
    Lấy danh sách mã chứng khoán trên sàn.
    """
    try:
        symbols = get_stock_symbols(exchange=exchange.upper())
        if not symbols:
            return f"Không tìm thấy mã nào trên sàn {exchange}."
        
        total = len(symbols)
        shown = symbols[:50]
        
        return f"Danh sách mã trên sàn {exchange} ({total} mã):\n{', '.join(shown)}..."
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# MENU: Thị trường (Market Overview)
# Gọi tất cả: Tâm lý + Phân loại nhà đầu tư + Định giá VNINDEX
# ============================================================

def menu_market_overview(days: int = 180):
    """
    Menu THỊ TRƯỜNG - Tổng quan toàn diện:
    Gọi TẤT CẢ các submenus:
    - Tâm lý thị trường (sentiment, volatility, HL index, BPI, MA, breadth)
    - Phân loại nhà đầu tư (dòng tiền VNINDEX)
    - Định giá VNINDEX (P/E, P/B)
    
    Dùng khi user hỏi chung về "thị trường hôm nay", "tổng quan thị trường", v.v.
    """
    try:
        days = _parse_days(days)
        results = []
        
        results.append("🌐 TỔNG QUAN THỊ TRƯỜNG VIỆT NAM")
        results.append("=" * 55)
        results.append(f"Ngày: {datetime.now().strftime('%d/%m/%Y')}")
        results.append("")
        
        # 1. Tâm lý thị trường
        results.append(tool_market_sentiment_full(days))
        results.append("")
        
        # 2. Phân loại nhà đầu tư
        results.append(tool_investor_classification("VNINDEX", "Daily"))
        results.append("")
        
        # 3. Định giá VNINDEX
        results.append(tool_valuation("VNINDEX", days))
        
        results.append("\n" + "=" * 55)
        results.append("✅ Phân tích hoàn tất")
        results.append("=" * 55)
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# MENU: Phân tích cổ phiếu (Stock Analysis)
# Gọi: Giá + Định giá + Phân loại nhà đầu tư (cho cổ phiếu đó)
# ============================================================

def menu_stock_analysis(symbol: str, days: int = 180):
    """
    Menu CỔ PHIẾU - Phân tích toàn diện một cổ phiếu:
    - Giá cổ phiếu (OHLCV)
    - Định giá (P/E, P/B)
    - Phân loại giao dịch (dòng tiền cho cổ phiếu đó)
    
    Dùng khi user hỏi về một cổ phiếu cụ thể.
    """
    try:
        days = _parse_days(days)
        results = []
        
        results.append(f"📈 PHÂN TÍCH CỔ PHIẾU {symbol.upper()}")
        results.append("=" * 55)
        results.append(f"Ngày: {datetime.now().strftime('%d/%m/%Y')}")
        results.append("")
        
        # 1. Giá
        results.append(tool_stock_price(symbol, 30))
        results.append("")
        
        # 2. Định giá
        results.append(tool_valuation(symbol, days))
        results.append("")
        
        # 3. Phân loại giao dịch
        results.append(tool_investor_classification(symbol, "Daily"))
        
        results.append("\n" + "=" * 55)
        results.append("✅ Phân tích hoàn tất")
        results.append("=" * 55)
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi: {str(e)}"


# ============================================================
# KHUYẾN NGHỊ (Recommendation)
# Tổng hợp dữ liệu + kiến thức chung để đưa ra khuyến nghị
# ============================================================

def tool_recommendation(query_type: str, symbol: str = "", days: int = 180):
    """
    Đưa ra khuyến nghị dựa trên dữ liệu + kiến thức chung về thị trường chứng khoán.
    
    Args:
        query_type: Loại câu hỏi - "market" (thị trường chung) hoặc "stock" (cổ phiếu cụ thể)
        symbol: Mã cổ phiếu (nếu query_type="stock")
        days: Số ngày phân tích
    
    Lưu ý: Tool này chỉ đưa ra NHẬN ĐỊNH dựa trên dữ liệu và kiến thức chung,
    KHÔNG phải lời khuyên đầu tư. User cần tự chịu trách nhiệm.
    """
    if not symbol:
        symbol = "VNINDEX"
    
    try:
        days = _parse_days(days)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        results = []
        results.append("📋 PHÂN TÍCH VÀ NHẬN ĐỊNH")
        results.append("=" * 55)
        results.append("⚠️ Lưu ý: Đây chỉ là nhận định dựa trên dữ liệu và kiến thức chung.")
        results.append("   KHÔNG phải lời khuyên đầu tư. Hãy tự nghiên cứu trước khi quyết định.")
        results.append("=" * 55)
        
        if query_type == "market":
            # Lấy dữ liệu thị trường
            sentiment_data = tool_market_sentiment_full(days)
            investor_data = tool_investor_classification("VNINDEX", "Daily")
            valuation_data = tool_valuation("VNINDEX", days)
            
            results.append("\n📊 DỮ LIỆU THỊ TRƯỜNG:")
            results.append("-" * 55)
            results.append(sentiment_data)
            results.append("")
            results.append(investor_data)
            results.append("")
            results.append(valuation_data)
            
            # Phân tích và nhận định
            results.append("\n📝 NHẬN ĐỊNH:")
            results.append("-" * 55)
            
            # Parse sentiment
            short_sentiment = 50
            vol_status = "normal"
            
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                df_sent = sentiment(start_date, end_date)
                if not df_sent.empty:
                    short_sentiment = df_sent.iloc[-1].get('short', 50)
            except:
                pass
            
            # Parse volatility
            try:
                df_vol = volatility('VNINDEX', end_date=end_date, countback=min(days, 252))
                if df_vol is not None and not df_vol.empty:
                    vol = df_vol.iloc[0].get('volatility', 0.02) if df_vol['close'].notna().iloc[0] else df_vol.iloc[-1].get('volatility', 0.02)
                    if vol > 0.03:
                        vol_status = "high"
                    elif vol < 0.015:
                        vol_status = "low"
            except:
                pass
            
            # Tổng hợp nhận định
            if short_sentiment > 70 and vol_status == "high":
                results.append("• Thị trường đang ở vùng QUÁ MUA - Cẩn trọng với việc mua vào")
            elif short_sentiment < 30:
                results.append("• Thị trường đang ở vùng QUÁ BÁN - Có thể là cơ hội cho nhà đầu tư dài hạn")
            elif short_sentiment > 55:
                results.append("• Tâm lý thị trường tích cực - Xu hướng tăng đang chiếm ưu thế")
            elif short_sentiment < 45:
                results.append("• Tâm lý thị trường tiêu cực - Cần theo dõi cẩn thận")
            else:
                results.append("• Thị trường đang trong giai đoạn cân bằng")
            
            if vol_status == "high":
                results.append("• Biến động cao - Rủi ro cao, nên thận trọng với vị thế lớn")
            elif vol_status == "low":
                results.append("• Biến động thấp - Thị trường ổn định")
            
            results.append("\n💡 Khuyến nghị chung:")
            results.append("• Nên Diversify (đa dạng hóa) danh mục")
            results.append("• Theo dõi các chỉ báo vĩ mô: lãi suất, tỷ giá, lạm phát")
            results.append("• Cân nhắc chiến lược Dollar Cost Averaging (DCA)")
            
        elif query_type == "stock" and symbol:
            # Lấy dữ liệu cổ phiếu
            price_data = tool_stock_price(symbol, 30)
            valuation_data = tool_valuation(symbol, days)
            investor_data = tool_investor_classification(symbol, "Daily")
            
            results.append(f"\n📊 DỮ LIỆU {symbol.upper()}:")
            results.append("-" * 55)
            results.append(price_data)
            results.append("")
            results.append(valuation_data)
            results.append("")
            results.append(investor_data)
            
            results.append("\n📝 NHẬN ĐỊNH:")
            results.append("-" * 55)
            
            # Parse valuation
            pe = 0
            pb = 0
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                df_val = get_pb_pe(symbol.upper(), start_date, end_date)
                if not df_val.empty:
                    pe = df_val.iloc[-1].get('pe', 0) or 0
                    pb = df_val.iloc[-1].get('pb', 0) or 0
            except:
                pass
            
            if pe > 0:
                if pe < 10:
                    results.append(f"• P/E = {pe:.2f} - Có vẻ undervalued (so với trung bình thị trường)")
                elif pe > 30:
                    results.append(f"• P/E = {pe:.2f} - Có vẻ overvalued")
                else:
                    results.append(f"• P/E = {pe:.2f} - Định giá trung bình")
            
            if pb > 0:
                if pb < 1:
                    results.append(f"• P/B = {pb:.2f} - Có vẻ undervalued (P/B < 1)")
                elif pb > 3:
                    results.append(f"• P/B = {pb:.2f} - Có vẻ overvalued")
            
            results.append("\n💡 Khuyến nghị chung:")
            results.append("• Nghiên cứu kỹ doanh nghiệp trước khi mua")
            results.append("• So sánh với các doanh nghiệp cùng ngành")
            results.append("• Theo dõi các yếu tố cơ bản: doanh thu, lợi nhuận, nợ vay")
            results.append("• Chú ý đến dòng tiền và xu hướng giao dịch")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Lỗi khi tạo khuyến nghị: {str(e)}"


# ============================================================
# MAPPING TOOLS + SCHEMA
# ============================================================

AVAILABLE_TOOLS = {
    # Menu-level tools
    "menu_market_overview": menu_market_overview,
    "menu_stock_analysis": menu_stock_analysis,
    
    # Submenu-level tools
    "tool_market_sentiment_full": tool_market_sentiment_full,
    "tool_market_sentiment_simple": tool_market_sentiment_simple,
    "tool_investor_classification": tool_investor_classification,
    "tool_valuation": tool_valuation,
    "tool_stock_price": tool_stock_price,
    "tool_stock_symbols": tool_stock_symbols,
    
    # Recommendation
    "tool_recommendation": tool_recommendation,
}


TOOLS_SCHEMA = [
    # ========== MENU: THỊ TRƯỜNG ==========
    {
        "type": "function",
        "function": {
            "name": "menu_market_overview",
            "description": "📊 TỔNG QUAN THỊ TRƯỜNG - Menu chính. GỌI TẤT CẢ: Tâm lý thị trường + Phân loại nhà đầu tư + Định giá VNINDEX. Dùng khi user hỏi: 'Thị trường hôm nay thế nào?', 'Tổng quan thị trường', 'Tình hình chứng khoán', 'Market overview', 'Thị trường VN'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày phân tích: 30 (1 tháng), 90 (3 tháng), 180 (6 tháng), 365 (1 năm). Mặc định 180."}
                },
                "required": []
            }
        }
    },
    
    # ========== SUBMENU: TÂM LÝ THỊ TRƯỜNG ==========
    {
        "type": "function",
        "function": {
            "name": "tool_market_sentiment_full",
            "description": "📊 TÂM LÝ THỊ TRƯỜNG ĐẦY ĐỦ - GỌI TẤT CẢ: Sentiment + Volatility + High-Low Index + BPI + MA + Market Breadth. Dùng khi user hỏi về 'tâm lý thị trường', 'xu hướng thị trường', 'market sentiment', 'thị trường bullish hay bearish'.",
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
            "name": "tool_market_sentiment_simple",
            "description": "🎯 TÂM LÝ THỊ TRƯỜNG ĐƠN GIẢN - Chỉ Sentiment cơ bản (short/long term). Dùng khi user hỏi đơn giản về tâm lý.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 180."}
                },
                "required": []
            }
        }
    },
    
    # ========== SUBMENU: PHÂN LOẠI NHÀ ĐẦU TƯ ==========
    {
        "type": "function",
        "function": {
            "name": "tool_investor_classification",
            "description": "👥 PHÂN LOẠI NHÀ ĐẦU TƯ - Dòng tiền theo loại: Nước ngoài, Tự doanh, Cá nhân, Tổ chức. Dùng khi user hỏi: 'Ai đang mua/bán?', 'Dòng tiền', 'Nhà đầu tư nào giao dịch nhiều?', 'foreign', 'proprietary', 'investor type'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (VD: VNINDEX, FPT, HPG). Mặc định VNINDEX."},
                    "frequency": {"type": "string", "description": "Tần suất: Daily, Weekly, Monthly. Mặc định Daily."}
                },
                "required": []
            }
        }
    },
    
    # ========== SUBMENU: ĐỊNH GIÁ ==========
    {
        "type": "function",
        "function": {
            "name": "tool_valuation",
            "description": "💰 ĐỊNH GIÁ - P/E và P/B. Dùng khi user hỏi: 'định giá', 'P/E', 'P/B', 'valuation', 'đắt hay rẻ' cho VNINDEX hoặc cổ phiếu cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (VD: VNINDEX, FPT, HPG). Mặc định VNINDEX."},
                    "days": {"type": "integer", "description": "Số ngày. Mặc định 365."}
                },
                "required": []
            }
        }
    },
    
    # ========== SUBMENU: GIÁ CỔ PHIẾU ==========
    {
        "type": "function",
        "function": {
            "name": "tool_stock_price",
            "description": "📈 GIÁ CỔ PHIẾU - Dữ liệu OHLCV (Giá mở, cao, thấp, đóng, khối lượng). Dùng khi user hỏi về giá của một cổ phiếu cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (VD: FPT, HPG, VCB)"},
                    "days": {"type": "integer", "description": "Số ngày dữ liệu. Mặc định 30."}
                },
                "required": ["symbol"]
            }
        }
    },
    
    # ========== SUBMENU: DANH SÁCH MÃ ==========
    {
        "type": "function",
        "function": {
            "name": "tool_stock_symbols",
            "description": "📋 DANH SÁCH MÃ CHỨNG KHOÁN - Lấy danh sách mã trên sàn. Dùng khi user hỏi: 'danh sách cổ phiếu', 'có những mã nào', 'list stocks'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "description": "Sàn: HOSE, HNX, UPCOM. Mặc định HOSE."}
                },
                "required": []
            }
        }
    },
    
    # ========== MENU: CỔ PHIẾU ==========
    {
        "type": "function",
        "function": {
            "name": "menu_stock_analysis",
            "description": "📈 PHÂN TÍCH CỔ PHIẾU - Menu Cổ phiếu. GỌI TẤT CẢ: Giá + Định giá + Phân loại giao dịch. Dùng khi user hỏi về một cổ phiếu cụ thể (VD: 'phân tích FPT', 'FPT thế nào', 'cổ phiếu VCB').",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (VD: FPT, HPG, VCB)"},
                    "days": {"type": "integer", "description": "Số ngày phân tích. Mặc định 180."}
                },
                "required": ["symbol"]
            }
        }
    },
    
    # ========== KHUYẾN NGHỊ ==========
    {
        "type": "function",
        "function": {
            "name": "tool_recommendation",
            "description": "💡 KHUYẾN NGHỊ - Đưa ra nhận định và khuyến nghị dựa trên dữ liệu + kiến thức chung về thị trường. GỌI TẤT CẢ dữ liệu liên quan và tổng hợp. Dùng khi user yêu cầu 'khuyến nghị', 'nhận định', 'đánh giá', 'recommendation', 'opinion', 'nên mua không', 'có nên đầu tư không'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {"type": "string", "description": "Loại phân tích: 'market' (thị trường chung) hoặc 'stock' (cổ phiếu cụ thể)."},
                    "symbol": {"type": "string", "description": "Mã cổ phiếu (nếu query_type='stock')."},
                    "days": {"type": "integer", "description": "Số ngày phân tích. Mặc định 180."}
                },
                "required": ["query_type"]
            }
        }
    },
]
