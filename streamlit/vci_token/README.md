VCI token helper
=================

Lightweight helper to fetch an authentication token from VCI trading API.

Usage
-----

from streamlit.vci_token import get_token

token = get_token()

Configuration
-------------
You can override defaults using environment variables:

- VCI_TOKEN_URL
- VCI_USERNAME
- VCI_PASSWORD
- VCI_CLIENT_ID
- VCI_CLIENT_SECRET
- VCI_DEVICE_ID
