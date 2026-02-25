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
            pool_connections=30,  # Number of connection pools to cache
            pool_maxsize=100,     # Maximum number of connections in pool (increased for high concurrency)
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
_CACHE_TTL = 1800  # 30 minutes cache TTL

def get_stock_history(symbol, period="day", start_date=None, end_date=None, count_back=None):
    """
    Lấy dữ liệu giá lịch sử cổ phiếu từ 24hmoney API
    
    Args:
        symbol (str): Mã cổ phiếu (VD: VCB, SSI)
        period (str): Khung thời gian (day, week, month) - mapped to resolution (1D, 1W, 1M)
        start_date (str): Ngày bắt đầu (format: YYYY-MM-DD), mặc định 1 năm trước
        end_date (str): Ngày kết thúc (format: YYYY-MM-DD), mặc định hôm nay
        count_back (int): Số ngày lấy về từ end_date (legacy parameter, mặc định 252)
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu OHLCV
    """
    from datetime import timedelta
    
    # Default end_date to today
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Handle count_back for backward compatibility
    if count_back is not None:
        # If count_back is provided, calculate start_date from end_date
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=count_back)).strftime('%Y-%m-%d')
    elif start_date is None:
        # Default start_date to 1 year ago if not provided
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Create cache key
    cache_key = f"{symbol}_{period}_{start_date}_{end_date}"
    
    # Check cache
    current_time = time.time()
    if cache_key in _stock_history_cache:
        cache_age = current_time - _cache_timestamps.get(cache_key, 0)
        if cache_age < _CACHE_TTL:
            return _stock_history_cache[cache_key].copy()
    
    # Convert dates to Unix timestamps
    start_date_epoch = int(pd.to_datetime(start_date).timestamp())
    end_date_epoch = int(pd.to_datetime(end_date).timestamp())
    
    # Map period to resolution
    resolution_map = {
        'day': '1D',
        'week': '1W',
        'month': '1M'
    }
    resolution = resolution_map.get(period.lower(), '1D')
    
    url = f"https://24hmoney.vn/dchart/history?symbol={symbol}&resolution={resolution}&from={start_date_epoch}&to={end_date_epoch}"
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'priority': 'u=1, i',
        'referer': 'https://24hmoney.vn/stock/MWG/technical-analysis',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Opera GX";v="124"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 OPR/124.0.0.0 (Edition globalgames-sd)',
    }

    session = _get_session()
    response = session.get(url, headers=headers, timeout=15)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()
    
    try:
        data = response.json()
        if 's' in data and data['s'] != 'ok':
            print(f"API Error: {data.get('s', 'Unknown error')}")
            return pd.DataFrame()
            
        # Handle different response formats
        if 't' in data:
            df = pd.DataFrame(data)
        elif 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            print(f"Unexpected response format: {data.keys()}")
            return pd.DataFrame()
        
        df.drop(['s'], axis=1, inplace=True, errors='ignore')
        df = df.rename(columns={'t': 'time', 'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Scale OHLC values by 1000 (API returns values in different unit)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * 1000
        
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
        
        # Store in cache
        _stock_history_cache[cache_key] = df.copy()
        _cache_timestamps[cache_key] = time.time()
        
        return df
    except Exception as e:
        print(f"Error parsing response: {e}")
        return pd.DataFrame()

# Cache for stock symbols - using a simple dict cache with TTL-like behavior
# Now caches by exchange to support multiple exchanges
_symbols_cache_dict = {}
_SYMBOLS_CACHE_TTL = 300  # 5 minutes cache TTL

def get_stock_symbols(exchange='HOSE'):
    """
    Lấy danh sách mã cổ phiếu theo sàn giao dịch từ 24hmoney API.
    
    Parameters:
    -----------
    exchange : str, default='HOSE'
        Sàn giao dịch (HOSE, HNX, UPCOM)
    
    Returns:
    --------
    list
        Danh sách các mã cổ phiếu (symbol có 3 ký tự)
    """
    global _symbols_cache_dict
    
    # Check cache for this specific exchange
    current_time = time.time()
    cache_entry = _symbols_cache_dict.get(exchange)
    if cache_entry is not None and (current_time - cache_entry['time']) < _SYMBOLS_CACHE_TTL:
        return cache_entry['symbols']
    
    url = "https://api-finance-t19.24hmoney.vn/v1/ios/company/az?device_id=web1739193qgnab7r86yja22nscew0z4zvqnubixw0430303&device_name=INVALID&device_model=Windows+10&network_carrier=INVALID&connection_type=INVALID&os=Opera&os_version=127.0.0.0&access_token=INVALID&push_token=INVALID&locale=vi&browser_id=web1739193qgnab7r86yja22nscew0z4zvqnubixw0430303&industry_code=all&floor_code=all&com_type=all&letter=all&page=1&per_page=2000"
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'origin': 'https://24hmoney.vn',
        'referer': 'https://24hmoney.vn/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    session = _get_session()
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Error fetching stock symbols: HTTP {response.status_code}")
            cache_entry = _symbols_cache_dict.get(exchange)
            return cache_entry['symbols'] if cache_entry else []
        
        # Check if response has content
        if not response.text or response.text.strip() == '':
            print("Error: Empty response from API")
            cache_entry = _symbols_cache_dict.get(exchange)
            return cache_entry['symbols'] if cache_entry else []
        
        json_data = response.json()
        if not json_data or 'data' not in json_data:
            print("Error: Invalid JSON response")
            cache_entry = _symbols_cache_dict.get(exchange)
            return cache_entry['symbols'] if cache_entry else []
        
        temp_df = pd.DataFrame(json_data['data'])
        if temp_df.empty or 'data' not in temp_df.columns:
            print("Error: No symbol data found")
            cache_entry = _symbols_cache_dict.get(exchange)
            return cache_entry['symbols'] if cache_entry else []
        
        data = pd.DataFrame(temp_df['data'].tolist())
        
        # Filter symbols with 3 characters and by exchange
        data = data[data['symbol'].str.len() == 3]
        filtered_data = data[data['floor'] == exchange]
        result = filtered_data['symbol'].sort_values().tolist()
        
        # Cache the result for this exchange
        _symbols_cache_dict[exchange] = {
            'symbols': result,
            'time': current_time
        }
        
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        cache_entry = _symbols_cache_dict.get(exchange)
        return cache_entry['symbols'] if cache_entry else []
    except Exception as e:
        print(f"Error fetching stock symbols: {e}")
        cache_entry = _symbols_cache_dict.get(exchange)
        return cache_entry['symbols'] if cache_entry else []

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
    
    # Kiểm tra nếu không có dữ liệu
    if all_data.empty:
        return pd.DataFrame()
    
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
