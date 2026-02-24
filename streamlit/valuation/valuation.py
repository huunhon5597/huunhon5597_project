import pandas as pd
import requests
import json
import sys
import os
import concurrent.futures
import importlib.util
import datetime
from datetime import datetime as dt, timedelta

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


def get_pb_pe(symbol, start_date=None, end_date=None):
    """
    Fetch historical pb/pe series for a symbol from FiinTrade API.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format (default: 1 year ago)
        end_date (str): End date in YYYY-MM-DD format (default: today)
    
    Returns:
        pd.DataFrame: DataFrame with columns ['symbol', 'date', 'price', 'pe', 'pb']
    """
    if end_date is None:
        end_date = dt.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (dt.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    url = f"https://wl-market.fiintrade.vn/MarketInDepth/GetValuationSeriesV2?language=vi&Code={symbol}&TimeRange=AllTime&FromDate={start_date}&ToDate={end_date}"
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'Connection': 'keep-alive',
        'Origin': 'https://app-kafi.fiintrade.vn',
        'Referer': 'https://app-kafi.fiintrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 OPR/127.0.0.0 (Edition globalgames-sd)',
        'sec-ch-ua': '"Opera GX";v="127", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    
    session = _get_session()
    response = session.get(url, headers=headers, timeout=15)
    
    if response.status_code != 200:
        raise Exception(f"API returned status {response.status_code}")
    
    data = pd.DataFrame(response.json()['items'])
    data = data.rename(columns={'code': 'symbol', 'tradingDate': 'date', 'value': 'price', 'r21': 'pe', 'r25': 'pb'})
    return data[['symbol', 'date', 'price', 'pe', 'pb']]


def _get_sector_symbols(symbol):
    """
    Get list of symbols in the same sector and exchange as the given symbol.
    Uses 24hmoney API which works on deployed servers.
    
    Args:
        symbol (str): Stock symbol to find peers for
        
    Returns:
        list: List of symbols in the same sector and exchange
    """
    url = "https://api-finance-t19.24hmoney.vn/v1/ios/company/az?device_id=web1739193qgnab7r86yja22nscew0z4zvqnubixw0430303&device_name=INVALID&device_model=Windows+10&network_carrier=INVALID&connection_type=INVALID&os=Opera&os_version=127.0.0.0&access_token=INVALID&push_token=INVALID&locale=vi&browser_id=web1739193qgnab7r86yja22nscew0z4zvqnubixw0430303&industry_code=all&floor_code=all&com_type=all&letter=all&page=1&per_page=2000"
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'origin': 'https://24hmoney.vn',
        'referer': 'https://24hmoney.vn/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    try:
        session = _get_session()
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"24hmoney API returned status {response.status_code}")
            return []
        
        temp_df = pd.DataFrame(response.json()['data'])
        sym_list = pd.DataFrame(temp_df['data'].tolist())
        
        # Filter for 3-letter symbols only (stocks)
        sym_list = sym_list[sym_list['symbol'].str.len() == 3]
        
        # Get the target symbol's sector code and floor
        symbol_data = sym_list[sym_list['symbol'] == symbol]
        if symbol_data.empty:
            print(f"Symbol {symbol} not found in 24hmoney data")
            return []
        
        target_icb_code = symbol_data['fiingroup_icb_code'].iloc[0]
        target_floor = symbol_data['floor'].iloc[0]
        
        # Filter for same sector and exchange
        sector_symbols = sym_list[
            (sym_list['fiingroup_icb_code'] == target_icb_code) & 
            (sym_list['floor'] == target_floor)
        ]
        
        return sector_symbols['symbol'].tolist()
        
    except Exception as e:
        print(f"Error getting sector symbols: {e}")
        return []


def ref_pb_pe(symbol, start_date=None, end_date=None):
    """
    Calculate P/B and P/E reference values for a symbol including sector comparison.
    Uses 24hmoney API for sector peers and FiinTrade API for valuation data.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format (default: 1 year ago)
        end_date (str): End date in YYYY-MM-DD format (default: today)
        
    Returns:
        tuple: (pb_result, pe_result) where each is a pd.Series with:
            - ttm_avg: TTM average
            - ttm_med: TTM median  
            - sec_avg: Sector average
            - sec_med: Sector median
    """
    # Get TTM values for the target symbol
    df = get_pb_pe(symbol, start_date, end_date)
    pb_ttm_avg = df['pb'].mean()
    pb_ttm_med = df['pb'].median()
    pe_ttm_avg = df['pe'].mean()
    pe_ttm_med = df['pe'].median()
    
    # Get latest P/B and P/E for the target symbol
    pb_latest = df['pb'].dropna().iloc[-1] if not df['pb'].dropna().empty else None
    pe_latest = df['pe'].dropna().iloc[-1] if not df['pe'].dropna().empty else None
    
    # Get sector symbols
    sector_symbols = _get_sector_symbols(symbol)
    
    pb_sec_values = []
    pe_sec_values = []
    
    if sector_symbols:
        def fetch_valuation(s):
            try:
                df_temp = get_pb_pe(s)
                if df_temp is not None and not df_temp.empty:
                    pb_val = df_temp['pb'].dropna().iloc[-1] if not df_temp['pb'].dropna().empty else None
                    pe_val = df_temp['pe'].dropna().iloc[-1] if not df_temp['pe'].dropna().empty else None
                    return {'symbol': s, 'pb': pb_val, 'pe': pe_val}
            except:
                pass
            return None
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(fetch_valuation, sector_symbols))
        
        # Collect valid values
        for r in results:
            if r is not None:
                if r['pb'] is not None and not pd.isna(r['pb']):
                    pb_sec_values.append(r['pb'])
                if r['pe'] is not None and not pd.isna(r['pe']):
                    pe_sec_values.append(r['pe'])
    
    # Calculate sector statistics
    pb_sec_avg = pd.Series(pb_sec_values).mean() if pb_sec_values else None
    pb_sec_med = pd.Series(pb_sec_values).median() if pb_sec_values else None
    pe_sec_avg = pd.Series(pe_sec_values).mean() if pe_sec_values else None
    pe_sec_med = pd.Series(pe_sec_values).median() if pe_sec_values else None
    
    pb_result = pd.Series({
        'pb_ttm_avg': pb_ttm_avg,
        'pb_ttm_med': pb_ttm_med,
        'pb_latest': pb_latest,
        'pb_sec_avg': pb_sec_avg,
        'pb_sec_med': pb_sec_med,
        'sector_count': len(pb_sec_values)
    })
    
    pe_result = pd.Series({
        'pe_ttm_avg': pe_ttm_avg,
        'pe_ttm_med': pe_ttm_med,
        'pe_latest': pe_latest,
        'pe_sec_avg': pe_sec_avg,
        'pe_sec_med': pe_sec_med,
        'sector_count': len(pe_sec_values)
    })
    
    return pb_result, pe_result


def get_peg(symbol):
    """
    Calculate PEG ratio (Price/Earnings to Growth) for a given stock symbol.
    Uses Vietcap API (iq.vietcap.com.vn) to get EPS growth for PEG calculation.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: PEG data with keys:
            - peg_ratio: PEG ratio (P/E / EPS growth)
            - pe_ratio: Current P/E (TTM)
            - eps_current: Current EPS (TTM)
            - eps_forward: Forward EPS
            - eps_growth: EPS growth rate (%)
            - note: Status message
    """
    print(f"Calculating PEG for {symbol}")
    
    try:
        # Get P/E from FiinTrade API
        df_pe = get_pb_pe(symbol)
        if df_pe is None or df_pe.empty:
            print(f"Dữ liệu P/E trống cho {symbol}")
            return None
            
        # Get the last valid P/E value
        pe_values = df_pe['pe'].dropna()
        if pe_values.empty:
            print(f"Không có giá trị P/E hợp lệ cho {symbol}")
            return None
            
        pe = pe_values.iloc[-1]
        if pd.isna(pe) or pe <= 0:
            print(f"Giá trị P/E không hợp lệ cho {symbol}")
            return None
        
        # Import and get token from vci_token
        try:
            from vci_token.token import get_token
            token = get_token()
        except Exception as e:
            print(f"Lỗi khi lấy token: {e}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': f'Lỗi lấy token: {e}'
            }
        
        # Get EPS growth from Vietcap API
        url = f"https://iq.vietcap.com.vn/api/iq-insight-service/v2/company/{symbol}/financial-data"
        
        headers = {
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Authorization': f'Bearer {token}',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://trading.vietcap.com.vn',
            'Referer': 'https://trading.vietcap.com.vn/iq/coverage?login-from=individual',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'client-id': 'a670914c-8964-4b2c-a289-6de4d5b9d2c4',
            'client-secret': '42IGbQ9oXZ1p2JK',
            'device-id': '194d5c0250f11306',
            'grant-type': 'password',
        }
        
        session = _get_session()
        response = session.get(url, headers=headers, timeout=15)
        
        print(f"Vietcap API status: {response.status_code}")
        
        if response.status_code != 200:
            # Store 403 error info for debugging
            error_info = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'text': response.text[:500] if response.text else ''
            }
            print(f"Vietcap API error info: {error_info}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': f'API returned status {response.status_code}',
                'error_info': error_info
            }
        
        response_data = response.json()
        
        if 'data' not in response_data:
            print(f"Không tìm thấy dữ liệu cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No data in API response'
            }
        
        data = pd.DataFrame(response_data['data'])
        
        if data.empty:
            print(f"Dữ liệu trống cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'Empty data from API'
            }
        
        # Set index to string to filter forecast data
        data.index = data.index.astype(str)
        data = data[data.index.str.contains('F', na=False)]
        
        # Filter by year - current year and next year (or just next year if after October)
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        years_allowed = [str(year), str(year + 1)] if month < 10 else [str(year + 1)]
        data = data[data.index.str[:4].isin(years_allowed)]
        
        if data.empty:
            print(f"Không có dữ liệu dự phóng (Forecast) phù hợp cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No forecast data available'
            }
        
        # Check for epsgrowth column
        if 'epsgrowth' not in data.columns:
            print(f"Không tìm thấy cột 'epsgrowth' trong dữ liệu cho {symbol}")
            print(f"Các cột có sẵn: {data.columns.tolist()}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No epsgrowth column in data'
            }
        
        # Get EPS growth value
        epsgrowth_series = data['epsgrowth'].dropna()
        if epsgrowth_series.empty:
            print(f"Không có giá trị epsgrowth hợp lệ cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No valid EPS growth value'
            }
        
        if len(epsgrowth_series) > 1:
            eps_growth = epsgrowth_series.mean()
        else:
            eps_growth = epsgrowth_series.iloc[0]
        
        # Check if eps_growth is valid (convert from percentage if needed)
        # EPS growth from API is usually in percentage form (e.g., 15 for 15%)
        if pd.isna(eps_growth) or eps_growth == 0:
            print(f"EPS growth không hợp lệ: {eps_growth} cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'Invalid EPS growth value'
            }
        
        # Convert EPS growth to percentage form
        # If eps_growth > 1, assume it's in percentage (e.g., 15 for 15%)
        # Multiply by 100 so dashboard can display directly without * 100
        if abs(eps_growth) > 1:
            eps_growth_pct = eps_growth
        else:
            eps_growth_pct = eps_growth * 100
        
        print(f"EPS growth: {eps_growth_pct}%")
        
        # Calculate PEG ratio: P/E / EPS Growth Rate (%)
        # PEG = P/E / (EPS Growth %) - only calculate if growth is positive
        if eps_growth_pct > 0:
            peg_ratio = pe / eps_growth_pct
        else:
            peg_ratio = None
        
        return {
            'peg_ratio': peg_ratio,
            'pe_ratio': pe,
            'eps_current': None,
            'eps_forward': None,
            'eps_growth': eps_growth_pct,
            'note': 'PEG calculated successfully' if peg_ratio else 'PEG calculation failed'
        }

    except Exception as e:
        print(f"Có lỗi xảy ra trong hàm get_peg: {e}")
        return None
