import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import sys
from collections import Counter
import pandas_ta as ta
import time
import numpy as np
from arch import arch_model
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stock_data import get_stock_symbols, get_stock_history

# Create a global session with connection pooling for better performance
_session = None

def _get_session():
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,  # Increased for high concurrency
            max_retries=3,
            pool_block=False
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


def _parse_date(d):
    """Accept str or datetime/date and return a date object."""
    if d is None:
        return None
    if isinstance(d, str):
        return datetime.strptime(d, '%Y-%m-%d').date()
    if isinstance(d, datetime):
        return d.date()
    return d

def sentiment (start_date, end_date=None):
    """
    Lấy dữ liệu tâm lý thị trường từ sstock.

    Args:
        start_date (str|datetime.date): YYYY-MM-DD hoặc datetime/date
        end_date (str|datetime.date): YYYY-MM-DD hoặc datetime/date (mặc định hôm nay)

    Returns:
        pd.DataFrame: cột ['time', 'short', 'long', 'close'] với 'time' là datetime
        short: tâm lý ngắn hạn
        long: tâm lý dài hạn
        close: giá đóng cửa vnindex
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    session = _get_session()
    
    # Define headers for both requests
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'origin': 'https://sstock.vn',
        'priority': 'u=1, i',
        'referer': 'https://sstock.vn/',
        'sec-ch-ua': '"Opera GX";v="125", "Not?A_Brand";v="8", "Chromium";v="141"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0 (Edition globalgames-sd)',
        'Cookie': 'sstock.current_company_full_info={%22code%22:%22SSI%22%2C%22label%22:%22C%C3%B4ng%20Ty%20C%E1%BB%95%20Ph%E1%BA%A7n%20Ch%E1%BB%A9ng%20Kho%C3%A1n%20SSI%22%2C%22value%22:%22SSI%22%2C%22sector%22:%22Ch%E1%BB%A9ng%20kho%C3%A1n%22%2C%22sectorId%22:%229%22}; __Secure-better-auth.session_token=mAHcV4RqH9wgiM6NHqpmmHCmeGd4EcIQ.c%2BzYenahhX2%2FiJwdB%2FuLNF7Rt3HpNqwxXBXNm0LUr18%3D; ph_phc_2O2eCgo6AOpwUykoQ5ufJGvaahcsg9cOPCMp4sZwSMh_posthog=%7B%22distinct_id%22%3A%22019b4ffe-1e86-76ec-a1b7-8c742b50bf6e%22%2C%22%24sesid%22%3A%5B1771509914669%2C%22019c7632-4c99-7221-9c69-2dc3d31260bb%22%2C1771509533818%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22https%3A%2F%2Fchat.zalo.me%2F%22%2C%22u%22%3A%22https%3A%2F%2Fsstock.vn%2Fthuc-chien%3Ftab%3Dbo-loc%22%7D%7D; __Secure-better-auth.session_data=eyJzZXNzaW9uIjp7InNlc3Npb24iOnsiZXhwaXJlc0F0IjoiMjAyNi0wMi0yNlQxNDowNDowOS42MzVaIiwidG9rZW4iOiJtQUhjVjRScUg5d2dpTTZOSHFwbW1IQ21lR2Q0RWNJUSIsImNyZWF0ZWRBdCI6IjIwMjYtMDItMTlUMTQ6MDQ6MDkuNjM1WiIsInVwZGF0ZWRBdCI6IjIwMjYtMDItMTlUMTQ6MDQ6MDkuNjM1WiIsImlwQWRkcmVzcyI6IjEwLjQyLjAuMzgiLCJ1c2VyQWdlbnQiOiJNb3ppbGxhLzUuMCAoV2luZG93cyBOVCAxMC4wOyBXaW42NDsgeDY0KSBBcHBsZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSBDaHJvbWUvMTQzLjAuMC4wIFNhZmFyaS81MzcuMzYgT1BSLzEyNy4wLjAuMCAoRWRpdGlvbiBnbG9iYWxnYW1lcy1zZCkiLCJ1c2VySWQiOiJ3Zk1ka1NpcjMwMTVWaDhJOEExMW5VVmlFYm44bFNneSIsImltcGVyc29uYXRlZEJ5IjpudWxsLCJhY3RpdmVPcmdhbml6YXRpb25JZCI6bnVsbCwiYWN0aXZlVGVhbUlkIjpudWxsLCJpZCI6IlNmMnFwYWtYbUhOdFBDZ3pVSElJbWtaOXBKYVJ1enFiIn0sInVzZXIiOnsibmFtZSI6Ik5oxqFuIE5ndXnhu4VuIEjhu691IiwiZW1haWwiOiJodXVuaG9uNTU5N0BnbWFpbC5jb20iLCJlbWFpbFZlcmlmaWVkIjp0cnVlLCJpbWFnZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0kxSDVuaUFoMFpaTlh0SUdlOGpXQURUQ0JPUjZINUdkbTc4MUd4T18wYkUwblJQeGdVPXM5Ni1jIiwiY3JlYXRlZEF0IjoiMjAyNS0xMi0yNFQxMDo1Mzo0NS4wMDBaIiwidXBkYXRlZEF0IjoiMjAyNS0xMi0yNFQxMDo1Mzo0NS4wMDBaIiwidXNlcm5hbWUiOm51bGwsImRpc3BsYXlVc2VybmFtZSI6bnVsbCwicm9sZSI6InVzZXIiLCJiYW5uZWQiOmZhbHNlLCJiYW5SZWFzb24iOm51bGwsImJhbkV4cGlyZXMiOm51bGwsInVzZXJUeXBlIjpudWxsLCJkaXNwbGF5UGhvbmVOdW1iZXIiOm51bGwsImlkIjoid2ZNZGtTaXIzMDE1Vmg4SThBMTFuVVZpRWJuOGxTZ3kifSwidXBkYXRlZEF0IjoxNzcxNTEwMTU4MjgyLCJ2ZXJzaW9uIjoiMSJ9LCJleHBpcmVzQXQiOjE3NzE1MTA0NTgyODIsInNpZ25hdHVyZSI6IlR2SUN3bTlrUDNNRTJUUEVLUnIwVENSeFBBZnJONThuTjM0bGtxVEJFZncifQ'
    }

    # Function to fetch sentiment data
    def fetch_sentiment_data():
        url = "https://api-feature.sstock.vn/api/v1/market/vn-index"
        response = session.get(url, headers=headers, timeout=15)
        return response.json()

    # Function to fetch VNINDEX data
    def fetch_vnindex_data():
        url = "https://api-feature.sstock.vn/api/v1/market/market-data?mack=VNINDEX"
        response = session.get(url, headers=headers, timeout=15)
        return response.json()

    # Execute both API calls in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_sentiment = executor.submit(fetch_sentiment_data)
        future_vnindex = executor.submit(fetch_vnindex_data)
        
        sentiment_response = future_sentiment.result()
        vnindex_response = future_vnindex.result()

    # Process sentiment data
    data1 = pd.DataFrame(sentiment_response['rs_market_breath_2w'])
    data1.columns = ['time', 'short']
    data2 = pd.DataFrame(sentiment_response['rs_market_breath_2m'])
    data2.columns = ['time', 'long']
    data = pd.merge(data1, data2, on='time', how='outer')
    data = data[data['short'] != 0]
    data['time'] = pd.to_datetime(data['time'], unit='ms')
    data['time'] = data['time'].dt.date

    # Process VNINDEX data
    vnindex = pd.DataFrame(vnindex_response['data'])
    vnindex = vnindex.rename(columns={'date': 'time'})
    vnindex = vnindex.iloc[::-1].reset_index(drop=True)
    
    # Ensure the 'time' columns are in a consistent datetime format
    data['time'] = pd.to_datetime(data['time'])
    vnindex['time'] = pd.to_datetime(vnindex['time'])

    # Merge the 'close' column from vnindex into the data DataFrame
    # A 'left' merge keeps all rows from the 'data' DataFrame
    data = data.merge(vnindex[['time', 'close']], on='time', how='left')
    data = data[data['time'] >= start_date]
    return data


def volatility(symbol='VNINDEX', end_date=None, countback=252, return_summary=False, forecast_days=10):
    """
    Hàm tính toán biến động (volatility) sử dụng mô hình GJR-GARCH
    
    Parameters:
    - symbol (str): Mã cổ phiếu, mặc định là 'VNINDEX'
    - end_date (str): Ngày kết thúc, mặc định là ngày hiện tại
    - countback (int): Số ngày quay lại để tính toán, mặc định là 252 ngày
    - return_summary (bool): Trả về thống kê mô hình nếu là True. Default là False.
    - forecast_days (int): Số ngày dự báo biến động trong tương lai, mặc định là 10 ngày
    
    Returns:
    - DataFrame: DataFrame chứa dữ liệu gốc và cột volatility tính toán, bao gồm cả dự báo tương lai
    - hoặc dict: chứa DataFrame và thống kê mô hình nếu return_summary=True
    """
    
    # Xử lý ngày kết thúc
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Tải dữ liệu lịch sử với buffer đủ lớn để tính toán GJR-GARCH
    try:
        df = get_stock_history(symbol, count_back=countback + 50)  # Thêm buffer 50 ngày
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")
        return None
    
    if df.empty:
        print(f"Không đủ dữ liệu cho {symbol} với countback = {countback}")
        return None
    
    # Chuyển đổi time sang datetime và sắp xếp
    if df['time'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Lọc dữ liệu theo countback (lấy countback ngày cuối cùng)
    df = df.tail(countback)
    
    # Tính Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()
    
    # Kiểm tra đủ dữ liệu để mô hình GJR-GARCH
    if len(df) < 50:  # Cần ít nhất 50 quan sát
        print(f"Không đủ dữ liệu để tính toán GJR-GARCH cho {symbol}")
        return None
    
    try:
        # Cấu hình mô hình GJR-GARCH (1,1,1)
        model = arch_model(df['log_return'], p=1, q=1, o=1, vol='GARCH', dist='Normal')
        # Khớp mô hình
        res = model.fit(disp='off')
        # Trích xuất biến động điều kiện
        df['volatility'] = res.conditional_volatility
        
        # Dự báo biến động tương lai
        if forecast_days > 0:
            # Dự báo biến động cho forecast_days ngày tiếp theo
            forecast = res.forecast(horizon=forecast_days)
            forecast_vol = np.sqrt(forecast.variance.values[-1, :])
            
            # Tạo các ngày giao dịch tiếp theo (bỏ qua cuối tuần)
            from datetime import timedelta, date
            current_date = pd.Timestamp.now().date()
            trading_days = []
            count = 0
            
            while count < forecast_days:
                if current_date.weekday() < 5:  # Monday=0, Friday=4
                    trading_days.append(current_date)
                    count += 1
                current_date += timedelta(days=1)
            
            # Tạo DataFrame cho dữ liệu dự báo
            forecast_data = []
            for i, trading_day in enumerate(trading_days):
                forecast_data.append({
                    'time': pd.to_datetime(trading_day),
                    'close': np.nan,  # Không có dữ liệu giá cho ngày dự báo
                    'volatility': forecast_vol[i] if i < len(forecast_vol) else np.nan,
                    'log_return': np.nan  # Không có dữ liệu return cho ngày dự báo
                })
            
            # Thêm dữ liệu dự báo vào DataFrame gốc
            # Keep the original 'time' column for historical data
            # Create forecast_df without setting 'time' as index
            forecast_df = pd.DataFrame(forecast_data)
            # Ensure 'time' column exists in both dataframes before concat
            if 'time' not in forecast_df.columns:
                forecast_df['time'] = pd.to_datetime(forecast_df['time'])
            # Concat directly - both dataframes have 'time' as regular column
            df = pd.concat([df, forecast_df], ignore_index=True)
        
        # Convert 'time' column to string format for consistency
        if 'time' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = df['time'].dt.strftime('%Y-%m-%d')
        
        if return_summary:
            summary = res.summary()
            return {'data': df, 'summary': str(summary)}
        else:
            return df
            
    except Exception as e:
        print(f"Lỗi khi mô hình GJR-GARCH cho {symbol}: {e}")
        return None



def high_low_index(start_date, end_date=None):
    """
    Tính High-Low Index dựa trên 252 phiên (1 năm giao dịch) cho mỗi mã trong danh sách.
    Đã được tối ưu hóa để tăng tốc độ xử lý.

    Args:
        start_date (str|datetime.date): ngày bắt đầu (YYYY-MM-DD hoặc datetime)
        end_date (str|datetime.date): ngày kết thúc (mặc định hôm nay)

    Returns:
        pd.DataFrame: ['time','peak_count','trough_count','record_high_percent','hl_index']
    """
    start = _parse_date(start_date) or datetime.now().date()
    end = _parse_date(end_date) or datetime.now().date()

    hose_list = get_stock_symbols()
    
    # Check if we got a valid symbol list
    if not hose_list:
        print("Warning: Could not fetch stock symbols for High-Low Index")
        return pd.DataFrame(columns=['time', 'peak_count', 'trough_count', 'record_high_percent', 'hl_index'])

    # Tối ưu: Xử lý song song các mã cổ phiếu với số lượng worker lớn hơn
    from functools import partial
    import threading
    
    # Thread-safe counters
    peak_counts = Counter()
    trough_counts = Counter()
    counts_lock = threading.Lock()

    def process_stock(symbol, start_date, end_date):
        """Process a single stock and return peak/trough dates."""
        local_peaks = []
        local_troughs = []
        try:
            # Lấy dữ liệu lịch sử với buffer đủ lớn để tính toán rolling window
            df = get_stock_history(symbol, count_back=252 * 3)
            if df.empty:
                return None
            
            # Chuyển đổi và sắp xếp dữ liệu một lần duy nhất
            df = df.sort_values('time')
            df['time'] = pd.to_datetime(df['time']).dt.date
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Lọc dữ liệu trong khoảng thời gian cần thiết
            window_df = df[(df['time'] >= start) & (df['time'] <= end)].copy()
            if window_df.empty:
                return None

            # Tối ưu: Tính toán rolling window một lần duy nhất
            rolling_max = df['close'].rolling(window=252, min_periods=252).max()
            rolling_min = df['close'].rolling(window=252, min_periods=252).min()

            # Lọc chỉ các ngày đủ dữ liệu (>= 252 ngày lịch sử)
            valid_dates = df[df['time'] <= end].tail(252).index[0]
            window_df = window_df[window_df.index >= valid_dates]

            # Xử lý từng ngày với dữ liệu đã tính toán sẵn
            for idx, row in window_df.iterrows():
                current_date = row['time']
                current_close = row['close']
                
                # Sử dụng dữ liệu rolling đã tính toán
                if current_close == rolling_max[idx]:
                    local_peaks.append(current_date)
                if current_close == rolling_min[idx]:
                    local_troughs.append(current_date)

            return (local_peaks, local_troughs)
        except Exception:
            return None
    
    # Sử dụng ThreadPoolExecutor với số lượng worker giảm để tránh rate limiting
    max_workers = min(10, len(hose_list))  # Giảm số worker xuống 10 để tránh rate limiting
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tất cả các task cùng lúc
        futures = {executor.submit(process_stock, symbol, start, end): symbol for symbol in hose_list}
        
        # Thu thập kết quả khi hoàn thành với timeout
        for future in as_completed(futures, timeout=120):  # 2 minute timeout
            try:
                result = future.result(timeout=30)  # 30 second timeout per stock
                if result:
                    local_peaks, local_troughs = result
                    # Thread-safe update of counters
                    with counts_lock:
                        for date in local_peaks:
                            peak_counts[date] += 1
                        for date in local_troughs:
                            trough_counts[date] += 1
            except Exception:
                continue  # Skip failed stocks

    # Tạo DataFrame kết quả
    all_dates = sorted(set(list(peak_counts.keys()) + list(trough_counts.keys())))
    if not all_dates:
        return pd.DataFrame(columns=['time', 'peak_count', 'trough_count', 'record_high_percent', 'hl_index'])
    
    hl_df = pd.DataFrame({'time': all_dates})
    hl_df['peak_count'] = hl_df['time'].map(lambda d: peak_counts.get(d, 0))
    hl_df['trough_count'] = hl_df['time'].map(lambda d: trough_counts.get(d, 0))

    # avoid division by zero
    tot = hl_df['peak_count'] + hl_df['trough_count']
    hl_df['record_high_percent'] = (hl_df['peak_count'] / tot.replace({0: pd.NA})) * 100
    hl_df['hl_index'] = ta.sma(hl_df['record_high_percent'].fillna(0), length=10)
    hl_df['time'] = pd.to_datetime(hl_df['time']).dt.strftime('%Y-%m-%d')

    return hl_df


def market_breadth(start_date, end_date):
    """
    Lấy dữ liệu Market Breadth từ VietCap và ghép với VN-Index.

    Args:
        start_date (str): Ngày bắt đầu (format: YYYY-MM-DD)
        end_date (str): Ngày kết thúc (format: YYYY-MM-DD)

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu market breadth đã ghép với vnindex
    """
    session = _get_session()
    
    # Breadth API
    url1 = (
        f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/market-watch/breadth?"
        f"condition=EMA50&exchange=HSX&fromDate={start_date}&toDate={end_date}"
    )

    headers1 = {
        'Accept': 'application/json',
        'Origin': 'https://trading.vietcap.com.vn',
        'Referer': 'https://trading.vietcap.com.vn/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response1 = session.get(url1, headers=headers1, timeout=10)
        response1.raise_for_status()
        j1 = response1.json()
    except Exception:
        # return empty DataFrame to let caller handle
        return pd.DataFrame()

    # Try multiple shapes for the returned JSON
    data_list = None
    if isinstance(j1, dict):
        if 'data' in j1:
            data_list = j1['data']
        elif 'result' in j1:
            data_list = j1['result']
    elif isinstance(j1, list):
        data_list = j1

    if data_list is None:
        try:
            data_list = json.loads(response1.text)
        except Exception:
            data_list = None

    data = pd.DataFrame(data_list) if data_list else pd.DataFrame()
    if data.empty:
        return pd.DataFrame()

    if 'tradingDate' in data.columns:
        data.rename(columns={'tradingDate': 'time'}, inplace=True)

    # normalize breadth time to date (avoid timezone/datetime mismatches)
    if 'time' in data.columns and not data['time'].empty:
        data['time'] = pd.to_datetime(data['time']).dt.date

    # Lấy dữ liệu VN-Index sử dụng get_stock_history (tương tự như hàm ma)
    # Tính số ngày cần lấy
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    days_to_fetch = (end_date_dt - start_date_dt).days + 10  # Thêm buffer 10 ngày
    
    try:
        vni_df = get_stock_history('VNINDEX', 'day', end_date, days_to_fetch)
        if vni_df.empty or 'close' not in vni_df.columns:
            vni_df = pd.DataFrame({'time': [], 'close': []})
        else:
            vni_df = vni_df[['time', 'close']].copy()
            vni_df['time'] = pd.to_datetime(vni_df['time'])
    except Exception:
        # If VN-Index data fails, we can still proceed with the breadth data
        # The 'vnindex' column will just be empty (NaN)
        vni_df = pd.DataFrame({'time': [], 'close': []})

    # Merge dữ liệu
    data['time'] = pd.to_datetime(data['time'])
    # Ensure vni_df time is also just date for merging
    vni_df['time'] = pd.to_datetime(vni_df['time']).dt.date
    data['time'] = data['time'].dt.date

    result = data.merge(vni_df, left_on='time', right_on='time', how='left').rename(columns={'close': 'vnindex'})

    # Ensure 'time' is a datetime for plotting
    if 'time' in result.columns and not result['time'].empty:
        result['time'] = pd.to_datetime(result['time'])

    return result


def bpi(start_date, end_date=None):
    """
    Calculates the Bullish Percent Index (BPI) for stocks on the HOSE exchange.
    The BPI is the percentage of stocks that have their 20-day EMA above their 50-day EMA.
    
    Đã được tối ưu hóa để tăng tốc độ xử lý và sửa lỗi logic:
    - Tăng count_back lên 200 để đảm bảo đủ dữ liệu cho EMA50
    - Tính total dựa trên số mã có EMA hợp lệ thay vì tổng mã giao dịch
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    hose_list = get_stock_symbols()
    
    # Check if we got a valid symbol list
    if not hose_list:
        print("Warning: Could not fetch stock symbols for BPI")
        return pd.DataFrame(columns=['time', 'count', 'total', 'bpi'])
    
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Thread-safe counter
    import threading
    bullish_counter = Counter()  # Đếm số mã có EMA20 > EMA50
    valid_ema_counter = Counter()  # Đếm số mã có EMA hợp lệ (đủ dữ liệu)
    counter_lock = threading.Lock()
    
    def process_stock_for_bpi(symbol, start_date, end_date):
        """
        Process a single stock and return:
        - dates where EMA20 > EMA50 (bullish)
        - dates where both EMA20 and EMA50 are valid (not NaN)
        """
        try:
            # Lấy dữ liệu lịch sử với buffer đủ lớn để tính toán EMA
            # Tăng count_back lên 200 để đảm bảo đủ dữ liệu cho EMA 50 ngày
            # EMA50 cần ít nhất 50 dữ liệu, nên cần buffer lớn hơn khoảng thời gian cần tính
            df = get_stock_history(symbol, count_back=200)
            if df.empty or 'close' not in df.columns or 'time' not in df.columns:
                return None, None

            # Tối ưu: Chỉ lấy các cột cần thiết và chuyển đổi dữ liệu một lần
            df = df[['time', 'close']].copy()
            df['time'] = pd.to_datetime(df['time'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # Tối ưu: Tính toán EMA một lần duy nhất
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            
            # Tối ưu: Lọc dữ liệu trước khi xử lý
            df_filtered = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
            if df_filtered.empty:
                return None, None
            
            # Tìm các ngày có EMA hợp lệ (cả EMA20 và EMA50 không phải NaN)
            valid_ema_mask = df_filtered['ema20'].notna() & df_filtered['ema50'].notna()
            valid_ema_dates = df_filtered.loc[valid_ema_mask, 'time'].tolist()
            
            # Tìm các ngày có EMA20 > EMA50 (trong số các ngày có EMA hợp lệ)
            bullish_mask = valid_ema_mask & (df_filtered['ema20'] > df_filtered['ema50'])
            bullish_dates = df_filtered.loc[bullish_mask, 'time'].tolist()
            
            return bullish_dates, valid_ema_dates
                
        except Exception:
            return None, None
    
    # Sử dụng ThreadPoolExecutor với số lượng worker giảm để tránh rate limiting
    max_workers = min(10, len(hose_list))  # Giảm số worker xuống 10 để tránh rate limiting
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tất cả các task cùng lúc
        futures = {executor.submit(process_stock_for_bpi, symbol, start_date_dt, end_date_dt): symbol for symbol in hose_list}
        
        # Xử lý kết quả khi các task hoàn thành với timeout
        for future in as_completed(futures, timeout=180):  # 3 minute timeout
            try:
                result = future.result(timeout=30)  # 30 second timeout per stock
                if result:
                    bullish_dates, valid_ema_dates = result
                    # Thread-safe update of counters
                    with counter_lock:
                        if bullish_dates:
                            for date in bullish_dates:
                                bullish_counter[date] += 1
                        if valid_ema_dates:
                            for date in valid_ema_dates:
                                valid_ema_counter[date] += 1
            except Exception:
                continue  # Skip failed stocks
    
    # Tạo DataFrame từ các Counter
    if not bullish_counter and not valid_ema_counter:
        return pd.DataFrame(columns=['time', 'count', 'total', 'bpi'])
    
    # Lấy tất cả các ngày từ cả hai counter
    all_dates = sorted(set(list(bullish_counter.keys()) + list(valid_ema_counter.keys())))
    
    bpi_df = pd.DataFrame({'time': all_dates})
    bpi_df['count'] = bpi_df['time'].map(lambda d: bullish_counter.get(d, 0))
    bpi_df['total'] = bpi_df['time'].map(lambda d: valid_ema_counter.get(d, 0))
    
    # Tính BPI: chỉ tính khi total > 0
    bpi_df['bpi'] = 0.0
    mask = bpi_df['total'] > 0
    bpi_df.loc[mask, 'bpi'] = (bpi_df.loc[mask, 'count'] / bpi_df.loc[mask, 'total']) * 100
    
    # Convert time to string format for consistency
    bpi_df['time'] = pd.to_datetime(bpi_df['time']).dt.strftime('%Y-%m-%d')
    
    return bpi_df

def ma(start_date, end_date=None):
    """
    Calculates 50-day and 200-day Simple Moving Averages (SMA) for the VNINDEX.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Calculate the number of days to fetch, adding a buffer for the MA calculation
    days_to_fetch = (end_date_dt - start_date_dt).days + 250

    ma_df = get_stock_history('VNINDEX', 'day', end_date, days_to_fetch)
    
    if ma_df.empty:
        return pd.DataFrame(columns=['time', 'close', 'open', 'high', 'low', 'ma50', 'ma200'])

    required_cols = ['time', 'close', 'open', 'high', 'low']
    if not all(col in ma_df.columns for col in required_cols):
        return pd.DataFrame(columns=required_cols + ['ma50', 'ma200'])

    ma_df['ma50'] = ta.sma(ma_df['close'], length=50)
    ma_df['ma200'] = ta.sma(ma_df['close'], length=200)

    ma_df['time'] = pd.to_datetime(ma_df['time'])
    ma_df = ma_df[ma_df['time'] >= start_date_dt].copy()
    
    return ma_df
