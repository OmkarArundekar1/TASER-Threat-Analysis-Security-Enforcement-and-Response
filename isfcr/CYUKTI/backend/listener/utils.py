import hashlib

def generate_alert_id(alert):
    timestamp = alert.get("timestamp", "")
    rule_id = alert.get("rule", {}).get("id", "")
    agent_id = alert.get("agent", {}).get("id", "")
    data = alert.get("data", {})
    srcip = (
        data.get("srcip")
        or alert.get("srcip", "")
    )
    dstip = (
        data.get("dstip")
        or alert.get("dstip", "")
    )
    location = alert.get("location", "")
    fingerprint = (
        f"{timestamp}|"
        f"{rule_id}|"
        f"{agent_id}|"
        f"{srcip}|"
        f"{dstip}|"
        f"{location}"
    )
    return hashlib.sha256(
        fingerprint.encode()
    ).hexdigest()