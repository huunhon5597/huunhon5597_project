# 🚀 Deployment Guide

## Local Development

1. **Tạo file `.env`** trong thư mục gốc:
```env
# API Keys (lấy từ providers tương ứng)
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_hf_api_key

# Stock data cookie (tùy chọn)
STOCK_COOKIE=your_cookie_here
```

2. **Chạy locally:**
```bash
pip install -r requirements.txt
streamlit run dashboard-streamlit.py
```

---

## Streamlit Cloud Deployment

### Bước 1: Push code lên GitHub
```bash
git add .
git commit -m "Add dashboard"
git push origin main
```

### Bước 2: Cấu hình Secrets trên Streamlit Cloud

1. Vào [Streamlit Cloud](https://share.streamlit.io)
2. New app → chọn repo GitHub
3. Advanced settings → Secrets:

```toml
# Secrets for Streamlit Cloud
GROQ_API_KEY = "your_groq_api_key_here"
HF_API_KEY = "your_hf_api_key_here"
```

4. Deploy

### Bước 3: Kiểm tra

- ✅ App sẽ tự động đọc secrets từ Streamlit
- ✅ Không cần .env file trên Cloud
- ✅ API keys được bảo mật trong secrets

---

## Lưu ý

- **KHÔNG** commit `.env` lên GitHub (đã có trong .gitignore)
- Secrets chỉ hoạt động khi deploy lên Streamlit Cloud
- Local vẫn dùng `.env` file