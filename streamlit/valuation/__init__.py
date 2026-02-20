import os
import importlib.util

# re-export core functions from valuation.py
from .valuation import get_pb_pe, ref_pb, ref_pe, get_peg

# try to import get_token from the vci_token package if available;
# otherwise provide a lazy proxy that loads token.py on first call.
try:
    from ..vci_token import get_token as _imported_get_token  # type: ignore
    get_token = _imported_get_token
except Exception:
    def _make_lazy_get_token():
        def _lazy_get_token(*args, **kwargs):
            base = os.path.dirname(__file__)  # streamlit/valuation
            token_path = os.path.normpath(os.path.join(base, "..", "vci_token", "token.py"))
            spec = importlib.util.spec_from_file_location("vci_token_token", token_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            real_get_token = getattr(module, "get_token")
            return real_get_token(*args, **kwargs)
        return _lazy_get_token
    get_token = _make_lazy_get_token()

__all__ = ["get_pb_pe", "ref_pb", "ref_pe", "get_peg", "get_token"]