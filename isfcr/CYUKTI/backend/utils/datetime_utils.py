from datetime import datetime, timezone

def normalize_datetime(value):
    if value is None:
        return None

    if hasattr(value, "to_native"):
        value = value.to_native()

    elif isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None

    elif not isinstance(value, datetime):
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value