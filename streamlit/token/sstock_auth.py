import os
import time
import json
from typing import Optional

# Cài đặt selenium trước: pip install selenium webdriver-manager
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

_CACHE: dict = {}
_COOKIES_FILE = "sstock_cookies.txt"


def _load_default_config():
    """Load configuration from environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    return {
        "email": os.environ.get("SSTOCK_EMAIL", ""),
        "password": os.environ.get("SSTOCK_PASSWORD", ""),
    }


def get_sstock_cookies(email: str = None, password: str = None, force_refresh: bool = False) -> str:
    """Lấy cookies từ sstock.vn bằng cách đăng nhập tự động.
    
    Args:
        email: Email đăng nhập sstock
        password: Mật khẩu đăng nhập sstock
        force_refresh: Nếu True, bỏ qua cache và lấy cookies mới
    
    Returns:
        Cookie string để sử dụng trong headers
    """
    global _CACHE
    
    now = time.time()
    
    # Kiểm tra cache
    if not force_refresh:
        cached = _CACHE.get("cookies")
        if cached and now - cached["ts"] < 86400:  # Cache 24 giờ
            return cached["value"]
    
    # Lấy credentials từ config nếu không được truyền vào
    if not email or not password:
        cfg = _load_default_config()
        email = email or cfg.get("email", "")
        password = password or cfg.get("password", "")
    
    if not email or not password:
        raise ValueError("Cần cung cấp email và password để đăng nhập sstock")
    
    # Thử đọc từ file trước
    cookies_str = _read_cookies_from_file()
    if cookies_str and not force_refresh:
        return cookies_str
    
    # Đăng nhập bằng Selenium
    cookies_str = _login_with_selenium(email, password)
    
    # Lưu vào cache và file
    if cookies_str:
        _CACHE["cookies"] = {"value": cookies_str, "ts": now}
        _save_cookies_to_file(cookies_str)
    
    return cookies_str


def _login_with_selenium(email: str, password: str) -> str:
    """Đăng nhập vào sstock.vn bằng Selenium."""
    if not SELENIUM_AVAILABLE:
        raise ImportError("Cần cài đặt selenium: pip install selenium webdriver-manager")
    
    driver = None
    try:
        # Khởi động trình duyệt
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.implicitly_wait(10)
        
        # Mở trang đăng nhập
        driver.get("https://sstock.vn/login")
        time.sleep(2)
        
        # Tìm và điền email - CẦN KIỂM TRA SELECTOR THỰC TẾ
        # Thử nhiều selector khác nhau
        try:
            email_input = driver.find_element(By.CSS_SELECTOR, "input[type='email']")
        except:
            try:
                email_input = driver.find_element(By.NAME, "email")
            except:
                email_input = driver.find_element(By.ID, "email")
        
        email_input.send_keys(email)
        
        # Tìm và điền password
        try:
            password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        except:
            try:
                password_input = driver.find_element(By.NAME, "password")
            except:
                password_input = driver.find_element(By.ID, "password")
        
        password_input.send_keys(password)
        
        # Click đăng nhập
        try:
            login_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        except:
            login_btn = driver.find_element(By.CSS_SELECTOR, "button.login-btn")
        
        login_btn.click()
        
        # Chờ đăng nhập thành công
        time.sleep(5)
        
        # Lấy cookies
        cookies = driver.get_cookies()
        
        # Chuyển đổi sang string
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
        
        return cookie_str
        
    except Exception as e:
        raise RuntimeError(f"Đăng nhập sstock thất bại: {e}")
    
    finally:
        if driver:
            driver.quit()


def _read_cookies_from_file() -> Optional[str]:
    """Đọc cookies từ file."""
    try:
        if os.path.exists(_COOKIES_FILE):
            with open(_COOKIES_FILE, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _save_cookies_to_file(cookies: str):
    """Lưu cookies vào file."""
    try:
        with open(_COOKIES_FILE, "w") as f:
            f.write(cookies)
    except Exception as e:
        print(f"Lưu cookies thất bại: {e}")


# Hàm tiện ích để lấy headers với cookies
def get_sstock_headers() -> dict:
    """Trả về headers đã có cookies cho API sstock."""
    cookies = get_sstock_cookies()
    
    return {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,vi;q=0.8,ko;q=0.7,fr;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'origin': 'https://sstock.vn',
        'priority': 'u=1, i',
        'referer': 'https://sstock.vn/',
        'sec-ch-ua': '"Opera GX";v="125", "Not?A_Brand";v="8", "Chromium";v="141"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0 (Edition globalgames-sd)',
        'Cookie': cookies
    }


if __name__ == "__main__":
    # Test - chạy script này để lấy cookies
    print("Đang đăng nhập sstock.vn...")
    
    # Đọc credentials từ environment hoặc sử dụng mặc định
    cfg = _load_default_config()
    
    if cfg.get("email") and cfg.get("password"):
        cookies = get_sstock_cookies(cfg["email"], cfg["password"], force_refresh=True)
        print("Đăng nhập thành công!")
        print(f"Cookies (50 ký tự đầu): {cookies[:50]}...")
        print(f"Đã lưu vào file: {_COOKIES_FILE}")
    else:
        print("Cần cung cấp SSTOCK_EMAIL và SSTOCK_PASSWORD trong environment variables")
        print("Hoặc chạy: python -c \"import os; os.environ['SSTOCK_EMAIL']='email'; os.environ['SSTOCK_PASSWORD']='pass'; from token import get_sstock_cookies; print(get_sstock_cookies())\"")
