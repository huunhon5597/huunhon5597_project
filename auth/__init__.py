"""Auth helper package.

Expose get_token() for importing: from streamlit.auth import get_token
"""
from .vci_token import get_token

__all__ = ["get_token"]
