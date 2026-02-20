"""VCI token helper package.

Expose get_token() for importing: from streamlit.vci_token import get_token
"""
from .token import get_token

__all__ = ["get_token"]
