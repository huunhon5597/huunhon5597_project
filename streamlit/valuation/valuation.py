import pandas as pd
import requests
import json
import sys
import os
import concurrent.futures
import importlib.util
import datetime
from datetime import datetime as dt, timedelta
import yfinance as yf

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
        
        # Get EPS from FiinTrade API (no token required)
        url = f"https://wl-fundamental.fiintrade.vn/Snapshot/GetSnapshot?language=vi&OrganCode={symbol}"
        
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
        
        print(f"FiinTrade API status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"FiinTrade API returned status {response.status_code}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': f'API returned status {response.status_code}'
            }
        
        response_data = response.json()
        
        if 'items' not in response_data:
            print(f"Không tìm thấy dữ liệu cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No data in API response'
            }
        
        data = pd.DataFrame(response_data['items'])
        
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
        
        # Get summary column
        if 'summary' not in data.columns:
            print(f"Không tìm thấy cột 'summary' trong dữ liệu cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No summary column in data'
            }
        
        # Convert the summary data to DataFrame
        summary_data = pd.DataFrame(data['summary'].tolist())
        
        # Get EPS values: rtd14 = EPS hiện tại (TTM), rtd53 = EPS forward
        eps_current = summary_data['rtd14'].iloc[0] if 'rtd14' in summary_data.columns else None
        eps_forward = summary_data['rtd53'].iloc[0] if 'rtd53' in summary_data.columns else None
        
        print(f"EPS hiện tại: {eps_current}, EPS forward: {eps_forward}")
        
        if eps_current is None or pd.isna(eps_current):
            print(f"Không có EPS hiện tại cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': eps_forward,
                'eps_growth': None,
                'note': 'No current EPS available'
            }
        
        # Calculate EPS growth if we have both current and forward EPS
        eps_growth_pct = None
        note = ''
        
        if eps_forward is not None and not pd.isna(eps_forward) and eps_current > 0:
            eps_growth_pct = ((eps_forward - eps_current) / abs(eps_current)) * 100
            print(f"EPS growth: {eps_growth_pct:.2f}%")
            
            # Check if eps_growth is negative
            if eps_growth_pct < 0:
                note = f'EPS growth âm ({eps_growth_pct:.2f}%) - Không có ý nghĩa tính PEG'
                print(f"{note}")
        
        # Calculate PEG ratio: P/E / EPS Growth Rate (%)
        # PEG = P/E / (EPS Growth %) - only calculate if growth is positive
        peg_ratio = None
        if eps_growth_pct is not None and eps_growth_pct > 0:
            peg_ratio = pe / eps_growth_pct
            note = 'PEG calculated successfully'
        elif eps_growth_pct is not None and eps_growth_pct <= 0:
            # Note already set above
            pass
        else:
            note = 'Không đủ dữ liệu để tính EPS growth'
        
        return {
            'peg_ratio': peg_ratio,
            'pe_ratio': pe,
            'eps_current': eps_current,
            'eps_forward': eps_forward,
            'eps_growth': eps_growth_pct,
            'note': note
        }

    except Exception as e:
        print(f"Có lỗi xảy ra trong hàm get_peg: {e}")
        return None


def fireant_valuation(symbol):
    """
    Fetch stock valuation from Fireant API.
    
    Args:
        symbol (str): Stock symbol (e.g., 'SSI', 'VNM', 'FPT')
    
    Returns:
        float: The composed/estimated price from Fireant, or None if failed
    """
    url = f"https://restv2.fireant.vn/symbols/{symbol}/estimated-price"
    
    headers = {
        'sec-ch-ua-platform': '"Windows"',
        'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoyMDIwNDI4ODMwLCJuYmYiOjE3MjA0Mjg4MzAsImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2NvcGUiOlsib3BlbmlkIiwicHJvZmlsZSIsInJvbGVzIiwiZW1haWwiLCJhY2NvdW50cy1yZWFkIiwiYWNjb3VudHMtd3JpdGUiLCJvcmRlcnMtcmVhZCIsIm9yZGVycy13cml0ZSIsImNvbXBhbmllcy1yZWFkIiwiaW5kaXZpZHVhbHMtcmVhZCIsImZpbmFuY2UtcmVhZCIsInBvc3RzLXdyaXRlIiwicG9zdHMtcmVhZCIsInN5bWJvbHMtcmVhZCIsInVzZXItZGF0YS1yZWFkIiwidXNlci1kYXRhLXdyaXRlIiwidXNlcnMtcmVhZCIsInNlYXJjaCIsImFjYWRlbXktcmVhZCIsImFjYWRlbXktd3JpdGUiLCJibG9nLXJlYWQiLCJpbnZlc3RvcGVkaWEtcmVhZCJdLCJzdWIiOiIxODAxZWMxMC0xOTlkLTQwZTItYjA2Zi05OTk1N2VjYTBhNTMiLCJhdXRoX3RpbWUiOjE3MjA0Mjg4MjksImlkcCI6Ikdvb2dsZSIsIm5hbWUiOiJodXVuaG9uNTU5N0BnbWFpbC5jb20iLCJzZWN1cml0eV9zdGFtcCI6IjBkMWNiYWM1LTJhY2ItNDM0YS04Y2RiLTkxYjJhMTQ0NDQwOSIsImp0aSI6IjdkZTVjNWFlYmIyMzI1ZTgxOTc1ZGI0ZDZiOGVhODFkIiwiYW1yIjpbImV4dGVybmFsIl19.uo0_GkgcLPW3FcESTrF8y4Frx8Y6qkEGeCkAc_CBLzpfMaMiTjTEL2hqotwaYBpupg8dPGMFo_NF6SMoEkJezMTDIoOdO6JrOxA_ZiKtWo24wOTJ-2lKfJeKV-d7iE5JyioFfhBFGFiDx17TcCqaE7js6boXPrr2h5-HmfaljEHcADSohl-B1ceW6U-yJHLjB-97ZTl4ggG5Cr0lTTj5ipmAwYIW32ZZWt9z1NfHnd4d3ZTyfmXZ2SR3xJmKcaBt9spwtzLGZqo9HRje2reKNK73MWespQL3n7pvufelK2HihYG1MMbIy52Tmc0T0Vrj3Z5xxizWhG6fUzUYud_clQ',
        'Referer': '',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 OPR/122.0.0.0 (Edition globalgames-sd)',
        'Accept': 'application/json, text/plain, */*',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Opera GX";v="122"',
        'sec-ch-ua-mobile': '?0'
    }
    
    try:
        session = _get_session()
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"Fireant API returned status {response.status_code}")
            return None
        
        data = response.json()
        composed_price = data.get('composedPrice')
        
        return composed_price
        
    except Exception as e:
        print(f"Error fetching Fireant valuation for {symbol}: {e}")
        return None


def analyst_price_targets(symbol):
    """
    Fetch analyst price targets for a Vietnamese stock using valueinvesting.io API.
    
    Args:
        symbol (str): Stock symbol (e.g., 'HPG', 'VNM', 'FPT')
    
    Returns:
        dict: Dictionary containing:
            - high: Highest price target
            - low: Lowest price target  
            - mean: Mean price target
            - median: Median price target
            Or None if failed
    """
    url = f"https://valueinvesting.io/company/estimates?limit=12&symbol={symbol}.VN"
    
    payload = json.dumps({
        "conditions": "[]",
        "mode": "watchlist",
        "existing_columns": "[]",
        "screener_currency": "",
        "latest_tracking_code": "1",
        "metrics_to_queries": "[\"256\",\"258\",\"260\",\"262\",\"146\",\"146\",\"146\",\"146\",\"146\",\"146\",\"103\",\"103\",\"103\",\"103\",\"103\",\"103\",\"233\",\"233\",\"233\",\"233\",\"233\",\"232\",\"232\",\"232\",\"232\",\"232\",\"237\",\"237\",\"237\",\"237\",\"237\",\"234\",\"234\",\"234\",\"234\",\"234\",\"235\",\"235\",\"235\",\"235\",\"235\",\"238\",\"238\",\"238\",\"238\",\"238\",\"239\",\"239\",\"239\",\"239\",\"239\",\"12\",\"168\",\"299\",\"206\"]",
        "metrics_to_queries_period": "[\"-1\",\"-1\",\"-1\",\"-1\",\"0\",\"FY-1\",\"FY-2\",\"FY-3\",\"FY-4\",\"FY-5\",\"0\",\"FY-1\",\"FY-2\",\"FY-3\",\"FY-4\",\"FY-5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"FY+1\",\"FY+2\",\"FY+3\",\"FY+4\",\"FY+5\",\"-1\",\"-1\",\"-1\",\"-1\"]",
        "metrics_to_queries_currency_type": "[\"2\",\"2\",\"2\",\"2\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"1\",\"0\",\"0\",\"0\",\"2\"]",
        "filter_default_freq": "[]",
        "filter_metrics": "[]",
        "filter_currency_type": "[]",
        "metrics_financials_grouping": "[\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\",\"NF\"]",
        "metrics_which_latest_days": "[\"None\",\"None\",\"None\",\"None\",\"None\",\"FY\",\"FY\",\"FY\",\"FY\",\"FY\",\"None\",\"FY\",\"FY\",\"FY\",\"FY\",\"FY\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"analystEstimateAnnual\",\"None\",\"None\",\"None\",\"None\"]",
        "name_of_considered_screener": "",
        "ticker_str": "(('" + symbol + "', 'VN'))",
        "original_filters": "[]",
        "original_columns": "[{\"field_code\":\"256\"},{\"field_code\":\"258\"},{\"field_code\":\"260\"},{\"field_code\":\"262\"},{\"field_code\":\"146\",\"period\":\"FY@6\"},{\"field_code\":\"103\",\"period\":\"FY@6\"},{\"field_code\":\"233\",\"period\":\"FY+1\"},{\"field_code\":\"233\",\"period\":\"FY+2\"},{\"field_code\":\"233\",\"period\":\"FY+3\"},{\"field_code\":\"233\",\"period\":\"FY+4\"},{\"field_code\":\"233\",\"period\":\"FY+5\"},{\"field_code\":\"232\",\"period\":\"FY+1\"},{\"field_code\":\"232\",\"period\":\"FY+2\"},{\"field_code\":\"232\",\"period\":\"FY+3\"},{\"field_code\":\"232\",\"period\":\"FY+4\"},{\"field_code\":\"232\",\"period\":\"FY+5\"},{\"field_code\":\"237\",\"period\":\"FY+1\"},{\"field_code\":\"237\",\"period\":\"FY+2\"},{\"field_code\":\"237\",\"period\":\"FY+3\"},{\"field_code\":\"237\",\"period\":\"FY+4\"},{\"field_code\":\"237\",\"period\":\"FY+5\"},{\"field_code\":\"234\",\"period\":\"FY+1\"},{\"field_code\":\"234\",\"period\":\"FY+2\"},{\"field_code\":\"234\",\"period\":\"FY+3\"},{\"field_code\":\"234\",\"period\":\"FY+4\"},{\"field_code\":\"234\",\"period\":\"FY+5\"},{\"field_code\":\"235\",\"period\":\"FY+1\"},{\"field_code\":\"235\",\"period\":\"FY+2\"},{\"field_code\":\"235\",\"period\":\"FY+3\"},{\"field_code\":\"235\",\"period\":\"FY+4\"},{\"field_code\":\"235\",\"period\":\"FY+5\"},{\"field_code\":\"238\",\"period\":\"FY+1\"},{\"field_code\":\"238\",\"period\":\"FY+2\"},{\"field_code\":\"238\",\"period\":\"FY+3\"},{\"field_code\":\"238\",\"period\":\"FY+4\"},{\"field_code\":\"238\",\"period\":\"FY+5\"},{\"field_code\":\"239\",\"period\":\"FY+1\"},{\"field_code\":\"239\",\"period\":\"FY+2\"},{\"field_code\":\"239\",\"period\":\"FY+3\"},{\"field_code\":\"239\",\"period\":\"FY+4\"},{\"field_code\":\"239\",\"period\":\"FY+5\"},{\"field_code\":\"12\"},{\"field_code\":\"168\"},{\"field_code\":\"299\"},{\"field_code\":\"206\"}]",
        "original_currency": "[\"\",\"\"]",
        "count_arr": "[]"
    })
    
    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'content-type': 'application/json',
        'origin': 'https://valueinvesting.io',
        'priority': 'u=1, i',
        'sec-ch-ua': '"Opera GX";v="127", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-full-version': '"127.0.5778.75"',
        'sec-ch-ua-full-version-list': '"Opera GX";v="127.0.5778.75", "Chromium";v="143.0.7499.194", "Not A(Brand";v="24.0.0.0"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-platform-version': '"10.0.0"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 OPR/127.0.0.0 (Edition globalgames-sd)',
        'Cookie': 'beegosessionID=c8d83cd290a878b2ef8edd8948f9667c; cf_clearance=aHgdqJ9DYvPpXFxZJi0vdiZglkvWjN9ivXJt3ZV49ME-1772535795-1.2.1.1-UySGf.5p_WYS6107qYFn5PWfppro9DkJ8uIKHqIqVqjy7lbiJLdsJRK7o1ExAd4lJH8qVSXbPVgBwkcz_VjfSdDrpJIDHcNz6TgM9GuO2cC4xEwbBcaRbjLVH6hzJarEIVNQuHts54q6uK38uKZY949QcVpPJc0b4h__YILNRultG8c1sTAbZDPDf9ud3aG7NN7i_pCOuUZV5skMtbyzBxWvLl8.jB6caY5XhEDgtx0; essay=eEXPcXslkD; _ga=GA1.1.2012916349.1772535797; twk_idm_key=m3rv0sT3N-Z878nry1I_8; TawkConnectionTime=0; twk_uuid_611c4284d6e7610a49b0ad9d=%7B%22uuid%22%3A%221.92R2xxtoQpimRheYQeUvZHdi6ymGX7CtAyweqbdOBtfXIbFEGHMcDDCQcEJIrsQRAqLq77c2yZijQgeKfqUvOjUGtDt4A0gJC3sNCEqLTzJSyzc8EfC2swVrPv10%22%2C%22version%22%3A3%2C%22domain%22%3A%22valueinvesting.io%22%2C%22ts%22%3A1772535801947%7D; token=d6jc01jaiij57r13levg; email=huunhon5597@gmail.com; _ga_4KHY6KT2C0=GS2.1.s1772535796$o1$g1$t1772537684$j6$l0$h0; email=huunhon5597@gmail.com'
    }
    
    try:
        session = _get_session()
        response = session.post(url, headers=headers, data=payload, timeout=15)
        
        if response.status_code != 200:
            print(f"valueinvesting.io API returned status {response.status_code}")
            return None
        
        response_data = response.json()
        
        if 'from_ar' not in response_data or 'main' not in response_data['from_ar']:
            print(f"Invalid response structure for {symbol}")
            return None
        
        response = response_data['from_ar']['main']
        df = pd.DataFrame(response)
        df = df.transpose()
        
        # Extract price target values
        # 256 = high, 258 = low, 260 = mean, 262 = median
        price_target_high = df['256_-1'].iloc[0] if '256_-1' in df.columns and not df['256_-1'].empty else None
        price_target_low = df['258_-1'].iloc[0] if '258_-1' in df.columns and not df['258_-1'].empty else None
        price_target_mean = df['260_-1'].iloc[0] if '260_-1' in df.columns and not df['260_-1'].empty else None
        price_target_median = df['262_-1'].iloc[0] if '262_-1' in df.columns and not df['262_-1'].empty else None
        
        result = {
            'high': float(price_target_high) if price_target_high is not None and not pd.isna(price_target_high) else None,
            'low': float(price_target_low) if price_target_low is not None and not pd.isna(price_target_low) else None,
            'mean': float(price_target_mean) if price_target_mean is not None and not pd.isna(price_target_mean) else None,
            'median': float(price_target_median) if price_target_median is not None and not pd.isna(price_target_median) else None
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching analyst price targets for {symbol}: {e}")
        return None
