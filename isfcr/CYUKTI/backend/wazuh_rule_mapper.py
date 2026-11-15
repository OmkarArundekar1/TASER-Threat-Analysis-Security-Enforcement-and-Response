"""
wazuh_rule_mapper.py
====================
Maps Wazuh rule IDs to MITRE ATT&CK technique IDs.

This addresses the critical "labeling starvation" issue where 82% of alerts
are marked as UNKNOWN because Wazuh doesn't provide ATT&CK labels for most
of its rules. This mapper provides explicit mappings based on what the rules
actually detect.

The mappings below are structured by Wazuh rule ID ranges and common patterns:
- Authentication/SSH rules (5500-5799) -> Credential Access, Brute Force
- Privilege escalation (2500-2599) -> Privilege Escalation
- File modification (550-554) -> Defense Evasion, Persistence
- Network scanning (40100-40199) -> Reconnaissance, Discovery
- Web attacks (31100-31199) -> Initial Access, Exploitation

This is a starting template. The security team should:
1. Review actual Wazuh rules that fire in their environment
2. Map each rule to the appropriate ATT&CK technique(s)
3. Test and validate mappings against known attacks
4. Maintain this mapping as rules are added/updated

Reference:
- Wazuh ruleset: /var/ossec/ruleset/rules/
- ATT&CK Enterprise: https://attack.mitre.org/
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WazuhRuleMapper:
    """
    Maps Wazuh rule IDs to MITRE ATT&CK techniques.
    """
    
    # Core rule ID -> technique ID mappings
    # Format: rule_id -> (technique_id, description)
    RULE_MAPPINGS = {
        # SSH Authentication Rules (5500-5799)
        5551: ("T1110", "SSH Brute Force - Multiple authentication failures"),
        5503: ("T1110.001", "SSH Password Guessing - Failed password"),
        5504: ("T1110.001", "SSH Password Guessing - Failed password"),
        5710: ("T1078", "Valid Accounts - Successful SSH authentication"),
        5712: ("T1021.004", "Remote Services: SSH - SSH login"),
        5720: ("T1078", "Valid Accounts - User authentication"),
        
        # Privilege Escalation (2500-2599)
        2501: ("T1548", "Abuse Elevation Control Mechanism - Sudo usage"),
        2502: ("T1548.003", "Sudo and Sudo Caching - Successful sudo"),
        2830: ("T1548", "Abuse Elevation Control Mechanism - User elevated privileges"),
        2831: ("T1548", "Abuse Elevation Control Mechanism - User elevated privileges"),
        
        # Account/User Management (5400-5499)
        5401: ("T1136", "Create Account - New user added"),
        5402: ("T1136.001", "Create Account: Local Account - User added to system"),
        5403: ("T1531", "Account Access Removal - User deleted"),
        
        # File Integrity Monitoring (550-554)
        550: ("T1565.001", "Data Manipulation: Stored Data - File modified"),
        551: ("T1070.004", "Indicator Removal: File Deletion - File deleted"),
        552: ("T1105", "Ingress Tool Transfer - File added"),
        553: ("T1222", "File and Directory Permissions Modification"),
        554: ("T1222", "File and Directory Permissions Modification"),
        
        # Network Scanning (40000-40199)
        40100: ("T1046", "Network Service Discovery - Port scanning detected"),
        40101: ("T1046", "Network Service Discovery - Port scanning"),
        40102: ("T1046", "Network Service Discovery - Network scan"),
        
        # Web Attacks (31000-31199)
        31100: ("T1190", "Exploit Public-Facing Application - Web attack"),
        31101: ("T1190", "Exploit Public-Facing Application - SQL injection attempt"),
        31102: ("T1059.007", "Command Injection - Command injection attempt"),
        31103: ("T1190", "Exploit Public-Facing Application - Web vulnerability scan"),
        31106: ("T1083", "File and Directory Discovery - Directory traversal"),
        31108: ("T1110", "Brute Force - Web brute force attack"),
        
        # Process Monitoring (2550-2650)
        2550: ("T1059.004", "Command and Scripting Interpreter: Unix Shell"),
        2551: ("T1059.004", "Command and Scripting Interpreter: Unix Shell"),
        2650: ("T1059.001", "Command and Scripting Interpreter: PowerShell"),
        
        # Service/Daemon Control (2900-2999)
        2900: ("T1543.002", "Create or Modify System Process: Systemd Service"),
        2901: ("T1569.002", "System Services: Service Execution"),
        2902: ("T1489", "Service Stop - Service stopped"),
        
        # Lateral Movement (18100-18199)
        18100: ("T1021", "Remote Services - Remote connection"),
        18101: ("T1021.002", "Remote Services: SMB/Windows Admin Shares"),
        18102: ("T1021.001", "Remote Services: Remote Desktop Protocol"),
        
        # Exfiltration (31500-31599)
        31500: ("T1041", "Exfiltration Over C2 Channel"),
        31501: ("T1048", "Exfiltration Over Alternative Protocol"),
        
        # Discovery (2800-2850)
        2800: ("T1057", "Process Discovery"),
        2801: ("T1082", "System Information Discovery"),
        2802: ("T1016", "System Network Configuration Discovery"),
        2803: ("T1049", "System Network Connections Discovery"),
        
        # Persistence (5100-5199)
        5100: ("T1053.003", "Scheduled Task/Job: Cron"),
        5101: ("T1543", "Create or Modify System Process"),
        5102: ("T1547", "Boot or Logon Autostart Execution"),
    }
    
    # Pattern-based mappings for rule ranges
    # Format: (start_id, end_id) -> technique_id
    RANGE_MAPPINGS = {
        # SSH authentication failures -> Brute Force
        (5500, 5550): "T1110.001",
        # Authentication success -> Valid Accounts
        (5700, 5750): "T1078",
        # Web attacks -> Exploit Public-Facing Application
        (31000, 31100): "T1190",
        # File changes -> varies, default to Data Manipulation
        (500, 600): "T1565",
    }
    
    def __init__(self, custom_mappings: dict = None):
        """
        Initialize mapper with optional custom mappings.
        
        Args:
            custom_mappings: Additional rule_id -> technique_id mappings
        """
        self.mappings = self.RULE_MAPPINGS.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)
        
        logger.info(
            f"Initialized WazuhRuleMapper with {len(self.mappings)} "
            f"explicit mappings"
        )
    
    def get_technique(self, rule_id: int) -> Optional[str]:
        """
        Get ATT&CK technique ID for a Wazuh rule.
        
        Args:
            rule_id: Wazuh rule ID
        
        Returns:
            ATT&CK technique ID (e.g., "T1110") or None if unmapped
        """
        # Try exact mapping first
        if rule_id in self.mappings:
            mapping = self.mappings[rule_id]
            # Handle tuple format (technique, description)
            if isinstance(mapping, tuple):
                return mapping[0]
            return mapping
        
        # Try range mappings
        for (start, end), technique in self.RANGE_MAPPINGS.items():
            if start <= rule_id <= end:
                logger.debug(
                    f"Rule {rule_id} mapped via range [{start}-{end}] -> {technique}"
                )
                return technique
        
        return None
    
    def get_description(self, rule_id: int) -> Optional[str]:
        """
        Get human-readable description for a mapped rule.
        
        Args:
            rule_id: Wazuh rule ID
        
        Returns:
            Description string or None
        """
        if rule_id in self.mappings:
            mapping = self.mappings[rule_id]
            if isinstance(mapping, tuple) and len(mapping) > 1:
                return mapping[1]
        return None
    
    def map_alert(self, alert: dict) -> Optional[str]:
        """
        Extract and map technique from a Wazuh alert.
        
        Args:
            alert: Wazuh alert dictionary
        
        Returns:
            ATT&CK technique ID or None
        """
        # First check if Wazuh already provided a technique
        if "rule" in alert and "mitre" in alert["rule"]:
            mitre_info = alert["rule"]["mitre"]
            if mitre_info and "id" in mitre_info:
                # Wazuh provides it, use that
                techniques = mitre_info["id"]
                if isinstance(techniques, list) and len(techniques) > 0:
                    return techniques[0]
                elif isinstance(techniques, str):
                    return techniques
        
        # Otherwise, try our mapping
        if "rule" in alert and "id" in alert["rule"]:
            rule_id = alert["rule"]["id"]
            try:
                rule_id_int = int(rule_id)
                technique = self.get_technique(rule_id_int)
                if technique:
                    logger.debug(
                        f"Mapped rule {rule_id} -> {technique} via custom mapping"
                    )
                    return technique
            except (ValueError, TypeError):
                logger.warning(f"Invalid rule ID format: {rule_id}")
        
        return None
    
    def add_mapping(self, rule_id: int, technique_id: str, description: str = None):
        """
        Add or update a rule mapping at runtime.
        
        Args:
            rule_id: Wazuh rule ID
            technique_id: ATT&CK technique ID
            description: Optional description
        """
        if description:
            self.mappings[rule_id] = (technique_id, description)
        else:
            self.mappings[rule_id] = technique_id
        
        logger.info(f"Added mapping: Rule {rule_id} -> {technique_id}")
    
    def export_mappings(self) -> dict:
        """Export all mappings for persistence/review."""
        return self.mappings.copy()
    
    def get_coverage_stats(self) -> dict:
        """
        Get statistics about mapping coverage.
        
        Returns:
            Dict with coverage statistics
        """
        return {
            "explicit_mappings": len(self.mappings),
            "range_mappings": len(self.RANGE_MAPPINGS),
            "total_mapped_rules": len(self.mappings),
        }


# Global mapper instance
mapper = WazuhRuleMapper()


# Convenience function for direct use
def get_technique_for_rule(rule_id: int) -> Optional[str]:
    """
    Get ATT&CK technique for a Wazuh rule ID.
    
    Args:
        rule_id: Wazuh rule ID
    
    Returns:
        ATT&CK technique ID or None
    """
    return mapper.get_technique(rule_id)


def get_technique_for_alert(alert: dict) -> Optional[str]:
    """
    Get ATT&CK technique from a Wazuh alert.
    
    Args:
        alert: Wazuh alert dictionary
    
    Returns:
        ATT&CK technique ID or None
    """
    return mapper.map_alert(alert)
