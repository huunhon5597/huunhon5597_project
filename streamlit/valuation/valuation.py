import pandas as pd
import requests
import json
import sys
import os
import concurrent.futures
import importlib.util
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vci_token import get_token

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


def get_pb_pe(symbol, countback=252, token=None):
  """Fetch historical pb/pe series for a symbol.

  Token is retrieved at call time so importing this module doesn't trigger network I/O.
  """
  if token is None:
      token = get_token()
  url = (
    f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/company-ratio-daily/{symbol}?lengthReport=10"
  )

  payload = {}
  headers = {
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
    'Authorization': f'Bearer {token}',
    'Connection': 'keep-alive',
    'Origin': 'https://trading.vietcap.com.vn',
    'Referer': 'https://trading.vietcap.com.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0 (Edition globalgames-sd)',
    'sec-ch-ua': '"Opera GX";v="125", "Not?A_Brand";v="8", "Chromium";v="141"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Cookie': (
      'FECW=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2; '
      'FECWS=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2'
    ),
  }

  session = _get_session()
  response = session.get(url, headers=headers, params=payload, timeout=15)
  response = response.json()
  data = pd.DataFrame(response['data'])
  data = data[['tradingDate', 'pb', 'pe']]
  data = data.tail(countback)

  return data


def ref_pb(symbol):
  df = get_pb_pe(symbol)
  pb_ttm_avg = df['pb'].mean()
  pb_ttm_med = df['pb'].median()

  url = (
    f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/valuation/{symbol}/peer-comparison?sectorType=ICB"
  )

  payload = {}
  # get a fresh token for this call
  token = get_token()

  headers = {
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
    'Authorization': f'Bearer {token}',
    'Connection': 'keep-alive',
    'Origin': 'https://trading.vietcap.com.vn',
    'Referer': 'https://trading.vietcap.com.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0 (Edition globalgames-sd)',
    'sec-ch-ua': '"Opera GX";v="125", "Not?A_Brand";v="8", "Chromium";v="141"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Cookie': (
      'FECW=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2; '
      'FECWS=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2'
    ),
  }

  session = _get_session()
  response = session.get(url, headers=headers, params=payload, timeout=15)
  response = response.json()
  data = pd.DataFrame(response['data'])
  sector_list = data['ticker'].tolist()
  sector_list = [s for s in sector_list if s not in [symbol, 'Median']]

  def fetch_pb(s):
      try:
          df_temp = get_pb_pe(s, token=token)
          pb_latest = df_temp['pb'].iloc[-1]
          return {'symbol': s, 'pb_latest': pb_latest}
      except:
          return None

  with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
      results = executor.map(fetch_pb, sector_list)

  pb_data = [r for r in results if r is not None]
    
  pb_df = pd.DataFrame(pb_data)
  pb_sec_avg = pb_df['pb_latest'].mean()
  pb_sec_med = pb_df['pb_latest'].median()
  result = pd.Series({
      'pb_ttm_avg': pb_ttm_avg,
      'pb_ttm_med': pb_ttm_med,
      'pb_sec_avg': pb_sec_avg,
      'pb_sec_med': pb_sec_med
  })
  return result


def get_peg(symbol):
    """Calculate PEG ratio (Price/Earnings to Growth) for a given stock symbol.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        float: PEG ratio value or None if calculation fails
    """
    # Lấy token (đảm bảo hàm này đã được định nghĩa)
    try:
        token = get_token()
    except NameError:
        print("Lỗi: Hàm 'get_token' chưa được định nghĩa.")
        return None

    url = f"https://iq.vietcap.com.vn/api/iq-insight-service/v2/company/{symbol}/financial-data"

    payload = {}
    # Lưu ý: Hardcode Cookie/Token thường không ổn định lâu dài,
    # nên có cơ chế lấy động nếu có thể.
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
        'Origin': 'https://trading.vietcap.com.vn',
        'Referer': 'https://trading.vietcap.com.vn/'
    }

    try:
        session = _get_session()
        response = session.get(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status() # Kiểm tra lỗi HTTP
        res_json = response.json()
        
        if 'data' not in res_json:
            print(f"Không tìm thấy dữ liệu cho {symbol}")
            return None
            
        data = pd.DataFrame(res_json['data'])
        
        # Kiểm tra xem index có phải là chuỗi chứa 'F' (Forecast) hay không
        # API thường trả về list dict, index mặc định là số (0,1,2).
        # Cần set index là cột năm/quý nếu dữ liệu trả về có cột đó, hoặc xử lý cột 'year'/'period'.
        # Giả định logic của bạn đúng với cấu trúc dữ liệu trả về:
        if not data.empty:
             # Nếu index chưa phải là string (ví dụ 2024F), dòng này có thể lỗi nếu index là số
             # Thường API trả về cột 'year' hoặc 'period', ta nên set index trước
             # Ví dụ: data.set_index('period', inplace=True) nếu có cột period
             pass

        # Lọc các hàng có index chứa chữ 'F' (Forecast)
        # Cần đảm bảo index là string trước khi dùng .str
        data.index = data.index.astype(str)
        data = data[data.index.str.contains('F', na=False)]

        # Lọc theo năm
        now = datetime.datetime.now()
        year = now.year
        month = now.month

        # Logic: Nếu chưa tới tháng 10 -> lấy năm hiện tại và năm kế. Nếu qua tháng 10 -> chỉ lấy năm kế
        years_allowed = [str(year), str(year + 1)] if month < 10 else [str(year + 1)]

        # Lọc data theo 4 ký tự đầu của index
        data = data[data.index.str[:4].isin(years_allowed)]

        if data.empty:
            print(f"Không có dữ liệu dự phóng (Forecast) phù hợp cho {symbol}")
            return None

        # Handle EPS growth calculation with better error handling
        try:
            if 'epsgrowth' not in data.columns:
                print(f"Không tìm thấy cột 'epsgrowth' trong dữ liệu cho {symbol}")
                print(f"Các cột có sẵn: {data.columns.tolist()}")
                return None
                
            # Check if data is empty after filtering
            if data.empty:
                print(f"Dữ liệu trống sau khi lọc cho {symbol}")
                return None
                
            # Check if epsgrowth column has valid data
            epsgrowth_series = data['epsgrowth']
            if epsgrowth_series.empty:
                print(f"Series epsgrowth trống cho {symbol}")
                return None
                
            # Remove NaN values before calculation
            valid_epsgrowth = epsgrowth_series.dropna()
            if valid_epsgrowth.empty:
                print(f"Không có giá trị epsgrowth hợp lệ cho {symbol}")
                return None
                
            if len(valid_epsgrowth) > 1:
                eps_growth = valid_epsgrowth.mean()
            else:
                eps_growth = valid_epsgrowth.iloc[0]
                
            # Check if eps_growth is valid
            if pd.isna(eps_growth) or eps_growth == 0:
                print(f"EPS growth không hợp lệ: {eps_growth} cho {symbol}")
                return None
                
        except Exception as e:
            print(f"Lỗi khi tính EPS growth: {e}")
            print(f"Chi tiết lỗi: {type(e).__name__}: {str(e)}")
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {data.columns.tolist()}")
            return None

        # Lấy P/E hiện tại với xử lý lỗi tốt hơn
        try:
            df_pe = get_pb_pe(symbol)
            if df_pe is None or df_pe.empty:
                print(f"Dữ liệu P/E trống cho {symbol}")
                return None
                
            # Check if 'pe' column exists and has valid data
            if 'pe' not in df_pe.columns:
                print(f"Không tìm thấy cột 'pe' trong dữ liệu cho {symbol}")
                print(f"Các cột có sẵn: {df_pe.columns.tolist()}")
                return None
                
            # Get the last valid P/E value
            pe_values = df_pe['pe'].dropna()
            if pe_values.empty:
                print(f"Không có giá trị P/E hợp lệ cho {symbol}")
                print(f"Dữ liệu P/E: {df_pe['pe'].head()}")
                return None
                
            pe = pe_values.iloc[-1]
            if pd.isna(pe):
                print(f"Giá trị P/E không hợp lệ cho {symbol}")
                return None
                
        except NameError:
            print("Lỗi: Hàm 'get_pb_pe' chưa được định nghĩa.")
            return None
        except Exception as e:
            print(f"Lỗi khi lấy P/E: {e}")
            print(f"Chi tiết lỗi: {type(e).__name__}: {str(e)}")
            print(f"PE data shape: {df_pe.shape if 'df_pe' in locals() else 'N/A'}")
            return None

        # Tính PEG
        try:
            peg_value = pe / (eps_growth * 100)
            
            # Return all data for dashboard display
            return {
                'peg_ratio': peg_value,
                'pe_ratio': pe,
                'eps_growth': eps_growth,
                'filtered_data': data
            }
        except Exception as e:
            print(f"Lỗi khi tính PEG: {e}")
            return None

    except Exception as e:
        print(f"Có lỗi xảy ra trong hàm get_peg: {e}")
        return None

def ref_pe(symbol):
  df = get_pb_pe(symbol)
  pe_ttm_avg = df['pe'].mean()
  pe_ttm_med = df['pe'].median()

  url = (
    f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/valuation/{symbol}/peer-comparison?sectorType=RA"
  )

  payload = {}
  # get a fresh token for this call
  token = get_token()

  headers = {
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
    'Authorization': f'Bearer {token}',
    'Connection': 'keep-alive',
    'Origin': 'https://trading.vietcap.com.vn',
    'Referer': 'https://trading.vietcap.com.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0 (Edition globalgames-sd)',
    'sec-ch-ua': '"Opera GX";v="125", "Not?A_Brand";v="8", "Chromium";v="141"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Cookie': (
      'FECW=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2; '
      'FECWS=0adc97d2e6659b7c3f81937668f755c57793eaefba70f8f2920cf54a9f332266029d2ec19a5404635a484ab0e1025015c76087b1dbb2791db555cc7713f41fab75abba96f89dbf7ab267c899c9e8e1d2c2'
    ),
  }

  session = _get_session()
  response = session.get(url, headers=headers, params=payload, timeout=15)
  response = response.json()
  data = pd.DataFrame(response['data'])
  sector_list = data['ticker'].tolist()
  sector_list = [s for s in sector_list if s not in [symbol, 'Median']]

  def fetch_pe(s):
      try:
          df_temp = get_pb_pe(s, token=token)
          pe_latest = df_temp['pe'].iloc[-1]
          return {'symbol': s, 'pe_latest': pe_latest}
      except:
          return None

  with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
      results = executor.map(fetch_pe, sector_list)

  pe_data = [r for r in results if r is not None]
    
  pe_df = pd.DataFrame(pe_data)
  pe_sec_avg = pe_df['pe_latest'].mean()
  pe_sec_med = pe_df['pe_latest'].median()
  result = pd.Series({
      'pe_ttm_avg': pe_ttm_avg,
      'pe_ttm_med': pe_ttm_med,
      'pe_sec_avg': pe_sec_avg,
      'pe_sec_med': pe_sec_med
  })
  return result
