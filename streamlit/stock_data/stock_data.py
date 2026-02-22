import pandas as pd
import requests
import json
from datetime import datetime
import time
import threading

# Create a global session with connection pooling for better performance
# This allows connection reuse across multiple requests
_session = None

# Rate Limiter - Giới hạn số requests per second để tránh bị chặn
class RateLimiter:
    """
    Rate limiter sử dụng token bucket algorithm.
    Giới hạn số requests có thể thực hiện trong một khoảng thời gian.
    """
    def __init__(self, max_requests_per_second=5, burst_size=10):
        """
        Args:
            max_requests_per_second: Số requests tối đa mỗi giây
            burst_size: Số requests có thể thực hiện cùng lúc (burst)
        """
        self.max_requests_per_second = max_requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens=1):
        """
        Acquire tokens from the bucket. Blocks if not enough tokens available.
        """
        with self.lock:
            now = time.time()
            # Replenish tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.max_requests_per_second)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0  # No wait needed
            
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.max_requests_per_second
            self.tokens = 0
            return wait_time
    
    def wait(self, tokens=1):
        """
        Wait if necessary to acquire tokens.
        """
        wait_time = self.acquire(tokens)
        if wait_time > 0:
            time.sleep(wait_time)

# Global rate limiter instance - 5 requests per second, burst up to 10
_rate_limiter = RateLimiter(max_requests_per_second=5, burst_size=10)

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

def get_stock_history(symbol, period="day", end_date=None, count_back=252):
    """
    Lấy dữ liệu giá lịch sử cổ phiếu
    
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
    
    url = f"https://api.vietcap.com.vn/ohlc-chart-service/v1/gap-chart?symbol={symbol}&to={end_date_epoch}&timeFrame=ONE_{period.upper()}&countBack={count_back}"
    
    headers = {
        'Accept': 'application/json',
        'Origin': 'https://trading.vietcap.com.vn',
        'Referer': 'https://trading.vietcap.com.vn/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    session = _get_session()
    
    # Apply rate limiting before making the request
    _rate_limiter.wait()
    
    try:
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        # Check if response has content
        if not response.text or response.text.strip() == '':
            return pd.DataFrame()
        
        json_data = response.json()
        if not json_data or 'data' not in json_data:
            return pd.DataFrame()
        
        data = json_data['data']
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.drop(['symbol', 'accumulatedVolume', 'accumulatedValue', 'minBatchTruncTime'], axis=1, inplace=True, errors='ignore')
        df = df.rename(columns={'t': 'time', 'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
        
        # Store in cache
        _stock_history_cache[cache_key] = df.copy()
        _cache_timestamps[cache_key] = time.time()
        
        return df
    except Exception as e:
        # Silent fail - return empty DataFrame
        return pd.DataFrame()

# Cache for stock symbols - using a simple dict cache with TTL-like behavior
_symbols_cache = None
_symbols_cache_time = 0
_SYMBOLS_CACHE_TTL = 300  # 5 minutes cache TTL

# Fallback list of popular HOSE stocks in case API fails
_HOSE_FALLBACK = [
    'ACB', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB',
    'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'REE', 'SAB', 'SSI', 'STB',
    'TCB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE', 'ANV',
    'APG', 'ASM', 'BCG', 'BMP', 'BSI', 'BWE', 'CII', 'CMG', 'CRE', 'CTS',
    'DBC', 'DCM', 'DGC', 'DHC', 'DIG', 'DPG', 'DPM', 'DRC', 'DXG', 'EIB',
    'FIT', 'FLC', 'FRT', 'FTS', 'GEG', 'GEX', 'GMD', 'HAG', 'HAH', 'HBC',
    'HCM', 'HDC', 'HHS', 'HNG', 'HPG', 'HPX', 'HSG', 'HT1', 'HVG', 'KDC',
    'KDH', 'KOS', 'KQB', 'L14', 'LDG', 'MCH', 'MCG', 'MSH', 'NBB', 'NKG',
    'NLG', 'NT2', 'NTL', 'OCB', 'PC1', 'PGB', 'PGC', 'PNJ', 'PPC', 'PVT',
    'PVD', 'PVS', 'QCG', 'RGC', 'SBT', 'SCS', 'SGC', 'SGN', 'SHI', 'SIP',
    'SKG', 'SMB', 'SMT', 'SRD', 'SSB', 'SSC', 'STK', 'SZC', 'TAC', 'TCI',
    'TDM', 'TNG', 'TPB', 'TTB', 'TV2', 'TVB', 'VCA', 'VCF', 'VCI', 'VDS',
    'VGC', 'VHC', 'VIB', 'VND', 'VNE', 'VNI', 'VNP', 'VOS', 'VPI', 'VSC',
    'VSH', 'VTO', 'YEG'
]

def get_stock_symbols(exchange="HOSE"):
    """Get stock symbols list with caching to avoid repeated API calls."""
    global _symbols_cache, _symbols_cache_time
    
    # Check cache
    current_time = time.time()
    if _symbols_cache is not None and (current_time - _symbols_cache_time) < _SYMBOLS_CACHE_TTL:
        return _symbols_cache
    
    url = "https://trading.vietcap.com.vn/api/price/v1/w/priceboard/tickers/price/group"
    
    payload = json.dumps({"group": exchange})
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Device-Id': '194d5c0250f11306',
        'Origin': 'https://trading.vietcap.com.vn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
    }
    
    session = _get_session()
    try:
        response = session.post(url, headers=headers, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"Error fetching stock symbols: HTTP {response.status_code}, using fallback list")
            return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK
        
        # Check if response has content
        if not response.text or response.text.strip() == '':
            print("Error: Empty response from API, using fallback list")
            return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK
        
        json_data = response.json()
        if not json_data:
            print("Error: Empty JSON response, using fallback list")
            return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK
        
        data = pd.DataFrame(json_data)
        if data.empty or 's' not in data.columns:
            print("Error: No symbol data found, using fallback list")
            return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK
        
        result = data['s'].sort_values().tolist()
        # Only cache successful results
        _symbols_cache = result
        _symbols_cache_time = current_time
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}, using fallback list")
        return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK
    except Exception as e:
        print(f"Error fetching stock symbols: {e}, using fallback list")
        return _symbols_cache if _symbols_cache is not None else _HOSE_FALLBACK

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
