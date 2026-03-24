"""
Script tự động lấy cookies từ Opera GX
Chạy: python auth/fetch_cookies.py
"""

import os
import time
import subprocess
import sys

# Set UTF-8 encoding for Vietnamese characters
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def kill_opera_gx():
    """Đóng tất cả các process Opera GX."""
    print("Đang đóng Opera GX...")
    
    # Thử nhiều cách để đóng Opera
    processes = ['opera.exe', 'opera gx.exe', 'Opera GX.exe']
    
    for proc in processes:
        try:
            subprocess.run(['taskkill', '/F', '/IM', proc], 
                         capture_output=True, timeout=10)
        except Exception as e:
            pass
    
    # Đợi cho process kết thúc
    time.sleep(3)
    print("Đã đóng Opera GX")

def fetch_sstock_cookies():
    """Lấy cookies sstock bằng Selenium sau khi đóng Opera GX."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Sử dụng Opera GX profile
        opera_profile = os.path.expandvars(r"%APPDATA%\Opera Software\Opera GX Stable\Default")
        
        if os.path.exists(opera_profile):
            options.add_argument(f"--user-data-dir={opera_profile}")
            print(f"Sử dụng Opera GX profile: {opera_profile}")
        else:
            print(f"Profile không tìm thấy: {opera_profile}")
            return None
        
        print("Đang khởi động browser...")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(15)
        
        print("Đang truy cập sstock.vn...")
        driver.get("https://sstock.vn/")
        
        # Đợi trang load hoàn toàn
        time.sleep(5)
        
        # Truy cập trang công ty để lấy cookie sstock.current_company_full_info
        print("Đang truy cập trang công ty...")
        driver.get("https://sstock.vn/ho-chi-minh-city/")
        time.sleep(3)
        
        # Lấy cookies
        cookies = driver.get_cookies()
        driver.quit()
        
        if not cookies:
            print("Không lấy được cookies sstock")
            return None
        
        # Chuyển đổi thành cookie string
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
        
        print(f"Đã lấy {len(cookies)} cookies sstock")
        return cookie_str
        
    except Exception as e:
        print(f"Lỗi Selenium sstock: {e}")
        return None

def fetch_valueinvesting_cookies():
    """Lấy cookies valueinvesting.io bằng Selenium sau khi đóng Opera GX."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Sử dụng Opera GX profile
        opera_profile = os.path.expandvars(r"%APPDATA%\Opera Software\Opera GX Stable\Default")
        
        if os.path.exists(opera_profile):
            options.add_argument(f"--user-data-dir={opera_profile}")
            print(f"Sử dụng Opera GX profile: {opera_profile}")
        else:
            print(f"Profile không tìm thấy: {opera_profile}")
            return None
        
        print("Đang khởi động browser...")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(15)
        
        print("Đang truy cập valueinvesting.io...")
        driver.get("https://valueinvesting.io/")
        
        # Đợi trang load hoàn toàn
        time.sleep(5)
        
        # Lấy cookies
        cookies = driver.get_cookies()
        driver.quit()
        
        if not cookies:
            print("Không lấy được cookies valueinvesting.io")
            return None
        
        # Chuyển đổi thành cookie string
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
        
        print(f"Đã lấy {len(cookies)} cookies valueinvesting.io")
        return cookie_str
        
    except Exception as e:
        print(f"Lỗi Selenium valueinvesting.io: {e}")
        return None

def save_cookies(cookies, filename):
    """Lưu cookies vào file."""
    if cookies:
        filepath = os.path.join(os.path.dirname(__file__), "..", filename)
        with open(filepath, "w") as f:
            f.write(cookies)
        print(f"Đã lưu cookies vào {filepath}")
        return True
    return False

def main():
    print("=" * 50)
    print("AUTOMATIC COOKIE FETCHER FOR SSTOCK & VALUEINVESTING.IO")
    print("=" * 50)
    
    # Bước 1: Đóng Opera GX
    kill_opera_gx()
    
    # Bước 2: Lấy cookies sstock
    print("\n[1/2] Đang lấy cookies sstock...")
    sstock_cookies = fetch_sstock_cookies()
    
    # Bước 3: Lấy cookies valueinvesting.io
    print("\n[2/2] Đang lấy cookies valueinvesting.io...")
    valueinvesting_cookies = fetch_valueinvesting_cookies()
    
    # Bước 4: Lưu vào file
    saved_sstock = False
    saved_valueinvesting = False
    
    if sstock_cookies:
        if save_cookies(sstock_cookies, "sstock_cookie.txt"):
            saved_sstock = True
    
    if valueinvesting_cookies:
        if save_cookies(valueinvesting_cookies, "valueinvesting.txt"):
            saved_valueinvesting = True
    
    # Kết quả
    print("\n" + "=" * 50)
    if saved_sstock and saved_valueinvesting:
        print("✅ Thành công! Cả hai cookies đã được cập nhật.")
    elif saved_sstock:
        print("✅ Đã cập nhật cookie sstock")
        print("⚠️  Chưa cập nhật được cookie valueinvesting.io")
    elif saved_valueinvesting:
        print("✅ Đã cập nhật cookie valueinvesting.io")
        print("⚠️  Chưa cập nhật được cookie sstock")
    else:
        print("❌ Thất bại! Không lấy được cookies nào.")
        print("\nThử cách khác:")
        print("1. Đảm bảo đã đăng nhập sstock.vn và valueinvesting.io trên Opera GX")
        print("2. Chạy lại script này")
    print("=" * 50)

if __name__ == "__main__":
    main()
