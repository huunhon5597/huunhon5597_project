# Stock Analysis Dashboard

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run dashboard-streamlit.py
```

## Dừng ứng dụng

```bash
taskkill /f /im streamlit.exe
```

## Cấu hình

Tạo file `.env` với các biến môi trường:

```env
# Cookie cho sstock.vn (lấy từ browser)
SSTOCK_COOKIES=your_cookie_string
SSTOCK_EMAIL=your_email
SSTOCK_PASSWORD=your_password

# AI API Keys (chọn ít nhất 1)
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key
```

## Cấu trúc project

```
streamlit/
├── dashboard-streamlit.py     # Main app
├── ai_core/                   # AI chat module
│   ├── hybrid_client.py       # Hybrid AI (recommended)
│   ├── groq_client.py         # Groq LLM client
│   ├── hf_client.py           # HuggingFace client
│   ├── chat_ui.py             # AI chat UI
│   ├── tools.py               # Available tools
│   └── prompts.py             # System prompts
├── stock_data/                # Stock data module
├── valuation/                 # Valuation tools (P/E, P/B, PEG)
├── market_sentiment/          # Market sentiment analysis
└── auth/                      # Authentication & token management
```

## AI Models

- **Hybrid** (recommended): Gọi tools trực tiếp + LLM tổng hợp, tránh loop/repetition
- **Groq**: Llama 3.3 70B, miễn phí, context 128K
- **HuggingFace**: Llama 3.1 8B, miễn phí 500 requests/ngày