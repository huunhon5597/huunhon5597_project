from stock_data import get_stock_history, get_listing_date, get_stock_symbols

# Ví dụ sử dụng thư viện
if __name__ == "__main__":
  
    # Lấy ngày niêm yết HPG
    hpg_listing = get_listing_date("HPG")
    print("Ngày niêm yết HPG:", hpg_listing)
    
    # Lấy dữ liệu giá 30 ngày gần nhất của SSI
    ssi_data = get_stock_history("SSI", count_back=30)
    print("Dữ liệu SSI 30 ngày:")
    print(ssi_data.head())

# Lấy mã HOSE (mặc định)
hose_symbols = get_stock_symbols()

# Lấy mã HNX
hnx_symbols = get_stock_symbols("HNX")

# Lấy mã UPCOM
upcom_symbols = get_stock_symbols("UPCOM")
