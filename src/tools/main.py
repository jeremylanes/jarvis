from datetime import datetime

from langchain_core.tools import tool


@tool
def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().isoformat(timespec="seconds")
