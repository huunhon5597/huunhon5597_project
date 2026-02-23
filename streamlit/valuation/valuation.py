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


def ref_pb_pe(symbol):
    """
    Calculate P/B and P/E reference values for a symbol including sector comparison.
    Uses 24hmoney API for sector peers and FiinTrade API for valuation data.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        tuple: (pb_result, pe_result) where each is a pd.Series with:
            - ttm_avg: TTM average
            - ttm_med: TTM median  
            - sec_avg: Sector average
            - sec_med: Sector median
    """
    # Get TTM values for the target symbol
    df = get_pb_pe(symbol)
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
    Uses FiinFundamental API to get EPS and forward EPS for growth calculation.
    
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
        
        # Get EPS and forward EPS from FiinFundamental API
        url = f"https://fiin-fundamental.ssi.com.vn/Snapshot/GetSnapshotNoneBank?OrganCode={symbol}"
        
        headers = {
            'accept': 'application/json',
            'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'origin': 'https://iboard.ssi.com.vn',
            'referer': 'https://iboard.ssi.com.vn/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 OPR/127.0.0.0 (Edition globalgames-sd)',
            'x-fiin-key': 'KEY',
            'x-fiin-seed': 'SEED',
            'x-fiin-user-id': 'ID',
            'x-fiin-user-token': '34,22,95,237,126,230,175,226,69,231,148,31,21,154,184,98,89,2,124,83,66,90,71,118,187,69,225,133,105,147,125,14,207,122,125,145,36,117,2,176,137,41,48,41,85,180,251,125,64,63,194,2,16,11,81,40,218,17,80,175,250,50,43,242,209,147,34,59,159,141,250,83,179,6,195,148,115,79,9,205,182,143,122,48,248,207,40,40,73,109,225,243,146,190,200,97,123,231,164,86,186,241,114,247,68,58,6,95,176,71,68,79,244,2,63,198,5,128,12,97,36,24,39,242,157,228,206,19,195,253,60,28,145,126,219,167,146,151,29,114,213,194,105,255,135,42,139,41,119,172,252,192,159,138,217,159,77,40,220,162,58,24,54,71,204,195,130,115,146,204,87,249,18,125,171,199,198,99,9,34,56,108,100,141,61,62,139,27,247,22,129,159,251,64,216,225,209,246,39,116,104,32,83,188,119,161,189,108,139,11,230,198,91,168,189,60,135,140,145,73,157,249,244,197,4,70,216,166,106,14,64,203,230,186,7,175,6,235,85,14,33,199,250,246,57,161,12,215,193,145,43,82,175,179,191,195'
        }
        
        session = _get_session()
        response = session.get(url, headers=headers, timeout=15)
        
        print(f"FiinFundamental API status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"FiinFundamental API returned status {response.status_code}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': f'API returned status {response.status_code}'
            }
        
        response_data = response.json()
        print(f"FiinFundamental API response keys: {response_data.keys()}")
        
        data = pd.DataFrame(response_data['items'])
        print(f"FiinFundamental items columns: {data.columns.tolist()}")
        
        if 'summary' not in data.columns:
            print(f"No 'summary' column in response for {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'No summary data in API response'
            }
        
        data = pd.DataFrame(data['summary'].tolist())
        print(f"FiinFundamental summary columns: {data.columns.tolist()}")
        
        eps_current = data['rtd14'].iloc[0] if 'rtd14' in data.columns else None
        eps_forward = data['rtd53'].iloc[0] if 'rtd53' in data.columns else None
        
        print(f"EPS current: {eps_current}, EPS forward: {eps_forward}")
        
        if eps_current is None or eps_forward is None:
            print(f"Không lấy được EPS data cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': None,
                'eps_forward': None,
                'eps_growth': None,
                'note': 'EPS data not available'
            }
        
        # Calculate EPS growth rate: ((Forward EPS - Current EPS) / Current EPS)
        # Note: Return as decimal (not multiplied by 100), dashboard will display with "%" symbol
        if eps_current > 0:
            eps_growth = (eps_forward - eps_current) / eps_current
        else:
            print(f"EPS current <= 0 cho {symbol}")
            return {
                'peg_ratio': None,
                'pe_ratio': pe,
                'eps_current': eps_current,
                'eps_forward': eps_forward,
                'eps_growth': None,
                'note': 'Invalid EPS current value'
            }
        
        # Calculate PEG ratio: P/E / EPS Growth Rate (%)
        # EPS growth is in decimal form, need to multiply by 100 for PEG calculation
        # PEG = P/E / (EPS Growth %)
        if eps_growth != 0:
            peg_ratio = pe / abs(eps_growth * 100)
        else:
            peg_ratio = None
        
        return {
            'peg_ratio': peg_ratio,
            'pe_ratio': pe,
            'eps_current': eps_current,
            'eps_forward': eps_forward,
            'eps_growth': eps_growth,
            'note': 'PEG calculated successfully' if peg_ratio else 'PEG calculation failed'
        }

    except Exception as e:
        print(f"Có lỗi xảy ra trong hàm get_peg: {e}")
        return None
