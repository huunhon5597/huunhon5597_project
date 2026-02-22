import pandas as pd
import requests
import json
from datetime import datetime
import time

# Create a global session with connection pooling for better performance
# This allows connection reuse across multiple requests
_session = None

def _get_session():
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure connection pool with larger size for high concurrency
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,  # Number of connection pools to cache
            pool_maxsize=50,      # Maximum number of connections in pool (increased for high concurrency)
            max_retries=3,        # Retry failed requests
            pool_block=False
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session

def get_listing_date(symbol):
    """
    Lấy ngày niêm yết của cổ phiếu
    
    Args:
        symbol (str): Mã cổ phiếu (VD: VCB, VIC, HPG)
    
    Returns:
        str: Ngày niêm yết (format: YYYY-MM-DD)
    """
    profile_url = f"https://restv2.fireant.vn/symbols/{symbol}/profile"
    profile_headers = {
        'accept': 'application/json, text/plain, */*',
        'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoyMDY2MDUzNTc2LCJuYmYiOjE3NjYwNTM1NzYsImNsaWVudF9pZCI6ImZpcmVhbnQud2ViIiwic2NvcGUiOlsib3BlbmlkIiwicHJvZmlsZSIsInJvbGVzIiwiZW1haWwiLCJhY2NvdW50cy1yZWFkIiwiYWNjb3VudHMtd3JpdGUiLCJvcmRlcnMtcmVhZCIsIm9yZGVycy13cml0ZSIsImNvbXBhbmllcy1yZWFkIiwiaW5kaXZpZHVhbHMtcmVhZCIsImZpbmFuY2UtcmVhZCIsInBvc3RzLXdyaXRlIiwicG9zdHMtcmVhZCIsInN5bWJvbHMtcmVhZCIsInVzZXItZGF0YS1yZWFkIiwidXNlci1kYXRhLXdyaXRlIiwidXNlcnMtcmVhZCIsInNlYXJjaCIsImFjYWRlbXktcmVhZCIsImFjYWRlbXktd3JpdGUiLCJibG9nLXJlYWQiLCJpbnZlc3RvcGVkaWEtcmVhZCJdLCJzdWIiOiIxODAxZWMxMC0xOTlkLTQwZTItYjA2Zi05OTk1N2VjYTBhNTMiLCJhdXRoX3RpbWUiOjE3NjYwNTM1NzUsImlkcCI6Ikdvb2dsZSIsIm5hbWUiOiJodXVuaG9uNTU5N0BnbWFpbC5jb20iLCJzZWN1cml0eV9zdGFtcCI6IjBkMWNiYWM1LTJhY2ItNDM0YS04Y2RiLTkxYjJhMTQ0NDQwOSIsImp0aSI6IjRiZDY3NzFmOGI5NDA3MjdmODEwNzI4ZGU2ZTAxODMxIiwiYW1yIjpbImV4dGVybmFsIl19.Ha5wUehTJv7aXxJXkaOrCSTucp31SLaIW0EzS3O7t94YYYtUsfbpZIMbDaCx70J2By3-43RGeQ7SpYT9nr5U9KhKR5ohPsHIbjlr5XzW0q40OR807eMHFQtyGIl-apIFeCLLciMF7fLQm20EFOcV3UEfaL55SAb_amW4iEjn_8g_xLE4C66CCEvb2bktxgCWVPVVPauR4TAxxOMk5ofJ-IsKSh-LoyA37kenrxQFGY7DwtdbkAny5z7kTm-WWyr_4e6igeiy4hysiPkGtdBMANbRsZcSC_WtprkwAIx8-BD32Xs5IF0JfscJgaOUfAK3NXg693dSnnlkxIIZncbWPA'
    }
    
    session = _get_session()
    profile_response = session.get(profile_url, headers=profile_headers, timeout=10)
    listing_date = profile_response.json()['dateOfListing']
    
    return listing_date

# Cache for stock history - using a simple dict cache with TTL-like behavior
_stock_history_cache = {}
_cache_timestamps = {}
_CACHE_TTL = 300  # 5 minutes cache TTL

def _fetch_history_from_vietcap(symbol, period, end_date_epoch, count_back):
    """Fetch stock history from VietCap API."""
    url = f"https://api.vietcap.com.vn/ohlc-chart-service/v1/gap-chart?symbol={symbol}&to={end_date_epoch}&timeFrame=ONE_{period.upper()}&countBack={count_back}"
    headers = {
        'Accept': 'application/json',
        'Origin': 'https://trading.vietcap.com.vn',
        'Referer': 'https://trading.vietcap.com.vn/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    session = _get_session()
    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and response.text and response.text.strip() != '':
            json_data = response.json()
            if json_data and 'data' in json_data and json_data['data']:
                return json_data['data']
    except Exception:
        pass
    return None

def _fetch_history_from_fireant(symbol, period, end_date, count_back):
    """Fetch stock history from Fireant API as fallback."""
    try:
        # Calculate start date from end_date and count_back
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - timedelta(days=count_back * 2)  # Buffer for non-trading days
        
        url = f"https://restv2.fireant.vn/symbols/{symbol}/historical-prices"
        params = {
            'startDate': start_dt.strftime('%Y-%m-%d'),
            'endDate': end_dt if isinstance(end_date, str) else end_date,
            'offset': 0,
            'limit': count_back + 50
        }
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        session = _get_session()
        response = session.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200 and response.text:
            data = response.json()
            if data:
                # Convert Fireant format to VietCap format
                result = []
                for item in data:
                    result.append({
                        't': int(pd.to_datetime(item['tradingDate']).timestamp()),
                        'o': item.get('openPrice', 0),
                        'h': item.get('highestPrice', 0),
                        'l': item.get('lowestPrice', 0),
                        'c': item.get('closePrice', 0),
                        'v': item.get('totalVolume', 0)
                    })
                return result
    except Exception:
        pass
    return None

def get_stock_history(symbol, period="day", end_date=None, count_back=252):
    """
    Lấy dữ liệu giá lịch sử cổ phiếu với nhiều API sources.
    
    Args:
        symbol (str): Mã cổ phiếu (VD: VCB, SSI)
        period (str): Khung thời gian (day, week, month)
        end_date (str): Ngày kết thúc (format: YYYY-MM-DD), mặc định hôm nay
        count_back (int): Số nến lấy về, mặc định 252 (số ngày giao dịch 1 năm)
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu OHLCV
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create cache key
    cache_key = f"{symbol}_{period}_{end_date}_{count_back}"
    
    # Check cache
    current_time = time.time()
    if cache_key in _stock_history_cache:
        cache_age = current_time - _cache_timestamps.get(cache_key, 0)
        if cache_age < _CACHE_TTL:
            return _stock_history_cache[cache_key].copy()
    
    end_date_epoch = int(pd.to_datetime(end_date).timestamp())
    
    # Try VietCap API first
    data = _fetch_history_from_vietcap(symbol, period, end_date_epoch, count_back)
    
    # Try Fireant API as fallback
    if not data:
        data = _fetch_history_from_fireant(symbol, period, end_date, count_back)
    
    if not data:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(data)
        df.drop(['symbol', 'accumulatedVolume', 'accumulatedValue', 'minBatchTruncTime'], axis=1, inplace=True, errors='ignore')
        df = df.rename(columns={'t': 'time', 'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
        
        # Store in cache
        _stock_history_cache[cache_key] = df.copy()
        _cache_timestamps[cache_key] = time.time()
        
        return df
    except Exception:
        return pd.DataFrame()

# Cache for stock symbols - using a simple dict cache with TTL-like behavior
_symbols_cache = None
_symbols_cache_time = 0
_SYMBOLS_CACHE_TTL = 3600  # 1 hour cache TTL

def _fetch_symbols_from_vietcap(exchange="HOSE"):
    """Fetch stock symbols from VietCap API."""
    url = "https://trading.vietcap.com.vn/api/price/v1/w/priceboard/tickers/price/group"
    payload = json.dumps({"group": exchange})
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Origin': 'https://trading.vietcap.com.vn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    session = _get_session()
    try:
        response = session.post(url, headers=headers, data=payload, timeout=15)
        if response.status_code == 200 and response.text and response.text.strip() != '':
            json_data = response.json()
            if json_data:
                data = pd.DataFrame(json_data)
                if not data.empty and 's' in data.columns:
                    return data['s'].sort_values().tolist()
    except Exception as e:
        print(f"VietCap API error: {e}")
    return None

def _fetch_symbols_from_fireant(exchange="HOSE"):
    """Fetch stock symbols from Fireant API as fallback."""
    url = "https://restv2.fireant.vn/symbols"
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    session = _get_session()
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and response.text:
            data = response.json()
            if isinstance(data, list):
                # Filter by exchange (HOSE = "HOSE", HNX = "HNX", UPCOM = "UPCOM")
                symbols = [s['symbol'] for s in data if s.get('exchange', '') == exchange]
                if symbols:
                    return sorted(symbols)
    except Exception as e:
        print(f"Fireant API error: {e}")
    return None

def _fetch_symbols_from_ssi(exchange="HOSE"):
    """Fetch stock symbols from SSI API as another fallback."""
    # SSI API endpoint for stock list
    url = f"https://fc-data.ssi.com.vn/api/v2/Market/Securities?market={exchange}"
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    session = _get_session()
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and response.text:
            json_data = response.json()
            if json_data and 'data' in json_data:
                data = json_data['data']
                if isinstance(data, list):
                    # Get symbol from Ticker field
                    symbols = [s.get('Ticker', s.get('Symbol', '')) for s in data if s.get('Ticker') or s.get('Symbol')]
                    if symbols:
                        return sorted([s for s in symbols if s])
    except Exception as e:
        print(f"SSI API error: {e}")
    return None

def get_stock_symbols(exchange="HOSE"):
    """
    Get stock symbols list with caching and multiple API fallbacks.
    
    Tries APIs in order:
    1. VietCap API (primary)
    2. Fireant API (fallback 1)
    3. SSI API (fallback 2)
    
    Returns cached data if all APIs fail.
    """
    global _symbols_cache, _symbols_cache_time
    
    # Check cache first
    current_time = time.time()
    if _symbols_cache is not None and (current_time - _symbols_cache_time) < _SYMBOLS_CACHE_TTL:
        return _symbols_cache
    
    # Try VietCap API first (primary source)
    print(f"Fetching {exchange} symbols from VietCap API...")
    symbols = _fetch_symbols_from_vietcap(exchange)
    
    if not symbols:
        # Try Fireant API as fallback
        print(f"VietCap failed, trying Fireant API...")
        symbols = _fetch_symbols_from_fireant(exchange)
    
    if not symbols:
        # Try SSI API as last resort
        print(f"Fireant failed, trying SSI API...")
        symbols = _fetch_symbols_from_ssi(exchange)
    
    if symbols:
        # Cache successful result
        _symbols_cache = symbols
        _symbols_cache_time = current_time
        print(f"Successfully fetched {len(symbols)} {exchange} symbols")
        return symbols
    
    # Return cached data if available (even if expired)
    if _symbols_cache is not None:
        print(f"All APIs failed, using cached data ({len(_symbols_cache)} symbols)")
        return _symbols_cache
    
    # No data available
    print(f"All APIs failed and no cache available for {exchange}")
    return []

def investor_type(symbol='VN-Index', start_date=None, end_date=None):
    """
    Hàm lấy dữ liệu phân loại nhà đầu tư từ nguoiquansat.vn
    
    Parameters:
    - symbol (str): Mã chứng khoán, mặc định là 'VN-Index' (sàn khác: HNX-Index, UPCOM-Index)
    - start_date (str): Ngày bắt đầu (định dạng yyyy-mm-dd)
    - end_date (str): Ngày kết thúc (định dạng yyyy-mm-dd), mặc định là ngày hiện tại
    
    Returns:
    - DataFrame: DataFrame chứa dữ liệu phân loại nhà đầu tư đã được xử lý
    """
    import datetime
    
    # Xử lý ngày kết thúc
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    # Định dạng lại ngày cho URL
    start_date_formatted = datetime.datetime.strptime(start_date, '%Y-%m-%d').date().strftime('%d/%m/%Y')
    end_date_formatted = datetime.datetime.strptime(end_date, '%Y-%m-%d').date().strftime('%d/%m/%Y')
    
    # Khởi tạo dataframe rỗng
    all_data = pd.DataFrame()
    page = 1
    
    while True:
        # Tạo URL với số trang hiện tại
        url = f'https://dulieu.nguoiquansat.vn/History/PhanLoaiNDTHistory?page={page}&fromDate={start_date_formatted}&toDate={end_date_formatted}&exId=&code={symbol}&idNganh=&_=1769924091168'
        
        try:
            # Đọc dữ liệu từ URL
            table = pd.read_html(url, encoding='utf-8')
            df = pd.DataFrame(table[0])
            
            # Kiểm tra nếu dataframe rỗng thì thoát vòng lặp
            if df.empty:
                break
            
            # Ghép dataframe mới vào dataframe tổng
            all_data = pd.concat([all_data, df], ignore_index=True)
            
            # Tăng số trang lên
            page += 1
            
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu từ trang {page}: {e}")
            break
    
    # Xử lý dataframe
    # 1. Xóa các cột có level 1 là 'GTGD khớp lệnh'
    if isinstance(all_data.columns, pd.MultiIndex):
        cols_to_drop = [col for col in all_data.columns if col[1] == 'GTGD khớp lệnh']
        all_data = all_data.drop(columns=cols_to_drop)
        
        # 2. Xử lý multi-level index columns
        all_data.columns = [
            col[0] if col[0] == col[1] else f"{col[0]} {col[1]}"
            for col in all_data.columns
        ]
    
    # 3. Bỏ cột 'STT'
    if 'STT' in all_data.columns:
        all_data = all_data.drop(columns=['STT'])
    
    # 4. Xóa đoạn chữ 'Tổng GTGD' khỏi tên các cột
    all_data.columns = all_data.columns.str.replace('Tổng GTGD', '', regex=False)
    all_data.columns = all_data.columns.str.strip()
    
    # 5. Chuyển đổi cột 'Ngày' sang định dạng datetime và sắp xếp
    all_data['Ngày'] = pd.to_datetime(all_data['Ngày'], format='%d/%m/%Y')
    all_data = all_data.sort_values('Ngày')
    all_data = all_data.reset_index(drop=True)
    
    return all_data
