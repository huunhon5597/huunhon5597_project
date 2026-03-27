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
        
        # Check if this is an index (VNINDEX, HNXINDEX, UPCOMINDEX)
        # Index values from API don't need scaling, but stocks do
        is_index = symbol.upper() in ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30', 'HNX30']
        
        if not is_index:
            # Scale OHLC values by 1000 for stocks (API returns values in different unit)
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
_SYMBOLS_CACHE_TTL = 1800  # 30 minutes cache TTL

# Cache for investor type data
_investor_type_cache = {}
_investor_type_cache_timestamps = {}
_INVESTOR_TYPE_CACHE_TTL = 1800  # 30 minutes cache TTL

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

def investor_type(symbol="VNINDEX", frequency="Daily"):
    """
    Lấy dữ liệu giao dịch theo loại nhà đầu tư từ FiinTrade.

    Parameters
    ----------
    symbol : str, default "VNINDEX"
        Mã chứng khoán (VNINDEX, HNXINDEX, UPCOMINDEX)
    frequency : str, default "Daily"
        Có thể là "Weekly" hoặc "Monthly"
    
    Returns:
    --------
    pd.DataFrame: DataFrame chứa dữ liệu giao dịch theo loại nhà đầu tư
    """
    # Create cache key
    cache_key = f"{symbol}_{frequency}"
    
    # Check cache
    current_time = time.time()
    if cache_key in _investor_type_cache:
        cache_age = current_time - _investor_type_cache_timestamps.get(cache_key, 0)
        if cache_age < _INVESTOR_TYPE_CACHE_TTL:
            return _investor_type_cache[cache_key].copy()
    
    url = f"https://wlgw-market.fiintrade.vn/MoneyFlow/GetStatisticInvestorChart?language=vi&Code={symbol}&Frequently={frequency}"
    headers = {
        'accept': 'application/json, text/plain, */*',
        'origin': 'https://app-kafi.fiintrade.vn',
        'referer': 'https://app-kafi.fiintrade.vn/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 OPR/127.0.0.0'
    }

    session = _get_session()
    response = session.get(url, headers=headers, timeout=15)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()
    
    try:
        json_data = response.json()
        if 'items' not in json_data:
            print("Error: Invalid response format")
            return pd.DataFrame()
            
        df = pd.DataFrame(json_data['items'])
        
        if df.empty:
            return pd.DataFrame()
        
        cols = ['code', 'tradingDate', 'closeValue', 'foreignBuyValue', 'foreignSellValue',
                'proprietaryTotalBuyTradeValue', 'proprietaryTotalSellTradeValue',
                'localIndividualBuyValue', 'localIndividualSellValue',
                'localInstitutionalBuyValue', 'localInstitutionalSellValue',
                'foreignIndividualBuyTradingValue', 'foreignIndividualSellTradingValue',
                'foreignInstitutionalBuyTradingValue', 'foreignInstitutionalSellTradingValue']
        df = df[cols]

        # Calculate NetValue for each investor type
        pairs = [
            ('foreignBuyValue', 'foreignSellValue', 'foreignNetValue'),
            ('proprietaryTotalBuyTradeValue', 'proprietaryTotalSellTradeValue', 'proprietaryNetValue'),
            ('localIndividualBuyValue', 'localIndividualSellValue', 'localIndividualNetValue'),
            ('localInstitutionalBuyValue', 'localInstitutionalSellValue', 'localInstitutionalNetValue'),
            ('foreignIndividualBuyTradingValue', 'foreignIndividualSellTradingValue', 'foreignIndividualNetValue'),
            ('foreignInstitutionalBuyTradingValue', 'foreignInstitutionalSellTradingValue', 'foreignInstitutionalNetValue'),
        ]
        for buy_col, sell_col, net_col in pairs:
            df[net_col] = df[buy_col] - df[sell_col]

        # Calculate Total Traded Value for each type
        df['foreignTotalTraded'] = df['foreignBuyValue'] + df['foreignSellValue']
        df['proprietaryTotalTraded'] = df['proprietaryTotalBuyTradeValue'] + df['proprietaryTotalSellTradeValue']
        df['domesticTotalTraded'] = (df['localIndividualBuyValue'] + df['localIndividualSellValue'] +
                                    df['localInstitutionalBuyValue'] + df['localInstitutionalSellValue'])
        df['foreignIndividualTotalTraded'] = df['foreignIndividualBuyTradingValue'] + df['foreignIndividualSellTradingValue']
        df['foreignInstitutionalTotalTraded'] = df['foreignInstitutionalBuyTradingValue'] + df['foreignInstitutionalSellTradingValue']

        # Keep buy/sell columns for now, don't drop them
        
        # Store in cache
        _investor_type_cache[cache_key] = df.copy()
        _investor_type_cache_timestamps[cache_key] = time.time()
        
        return df
    except Exception as e:
        print(f"Error parsing response: {e}")
        return pd.DataFrame()
