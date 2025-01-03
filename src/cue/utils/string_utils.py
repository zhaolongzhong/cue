from typing import Optional


def truncate_safely(text: Optional[str] = None, length: Optional[int] = 100):
    """Safely truncate text, handling None values."""
    if text is None:
        return None
    if length:
        return str(text)[:length]
    else:
        return str(text)
