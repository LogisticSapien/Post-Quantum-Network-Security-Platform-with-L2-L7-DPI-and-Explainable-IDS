"""
Alert Correlation Engine
=========================
XDR-style alert correlation:
  - Groups alerts by source IP within time window
  - Attack chain detection (scan -> exploit -> C2)
  - Incident severity escalation:
    * 1 alert  = original severity
    * 2 alerts = MEDIUM (minimum)
    * 3+ alerts = HIGH/CRITICAL
  - Mirrors Cisco Talos correlation methodology
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ids import ThreatEvent, Severity, SEVERITY_LABELS


# Attack chain definitions: category sequences that indicate escalation
ATTACK_CHAINS = {
    "recon_to_exploit": {
        "stages": ["PORT_SCAN", "BRUTE_FORCE"],
        "description": "Reconnaissance followed by credential attack",
    },
    "scan_flood_tunnel": {
        "stages": ["PORT_SCAN", "SYN_FLOOD", "DNS_TUNNEL"],
        "description": "Full kill chain: recon -> DoS -> C2 tunnel",
    },
    "recon_to_c2": {
        "stages": ["PORT_SCAN", "DNS_TUNNEL"],
        "description": "Reconnaissance followed by C2 channel establishment",
    },
    "brute_to_exfil": {
        "stages": ["BRUTE_FORCE", "DNS_EXFIL"],
        "description": "Credential attack followed by data exfiltration",
    },
    "flood_tunnel": {
        "stages": ["SYN_FLOOD", "ICMP_TUNNEL"],
        "description": "DoS distraction followed by covert channel",
    },
}


@dataclass
class Incident:
    """A correlated incident grouping multiple related alerts."""
    incident_id: str
    source_ip: str
    alerts: List[ThreatEvent] = field(default_factory=list)
    escalated_severity: Severity = Severity.LOW
    attack_chain: Optional[str] = None
    chain_description: Optional[str] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_escalated: bool = False

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    @property
    def categories(self) -> List[str]:
        return list(dict.fromkeys(a.category for a in self.alerts))

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen if self.first_seen > 0 else 0.0

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS.get(self.escalated_severity, "UNKNOWN")

    @property
    def summary(self) -> str:
        cats = ", ".join(self.categories)
        chain = f" [{self.chain_description}]" if self.chain_description else ""
        return (
            f"Incident {self.incident_id}: {self.alert_count} alerts from {self.source_ip} "
            f"({cats}){chain} [{self.severity_label}]"
        )


class AlertCorrelator:
    """
    Correlate alerts by source IP within a time window.

    Groups related alerts into Incidents and escalates severity:
      - 1 alert = keep original severity
      - 2 correlated alerts = at least MEDIUM
      - 3+ correlated alerts = at least HIGH
      - Matching attack chain = CRITICAL
    """

    def __init__(
        self,
        time_window: float = 60.0,
        escalation_enabled: bool = True,
    ):
        self.time_window = time_window
        self.escalation_enabled = escalation_enabled

        # Active incidents by source IP
        self._incidents: Dict[str, Incident] = {}
        self._incident_counter: int = 0

        # Completed incidents
        self.completed: List[Incident] = []
        self._max_completed: int = 500

    def correlate(self, alert: ThreatEvent) -> Optional[Incident]:
        """
        Process an alert and attempt correlation.

        Returns an Incident if correlation threshold is met
        (i.e., the incident was created or escalated).
        """
        src = alert.source_ip or "unknown"
        now = alert.timestamp or time.time()

        # Check if we have an active incident for this source
        incident = self._incidents.get(src)

        if incident is not None:
            # Check if within time window
            if now - incident.last_seen <= self.time_window:
                # Add to existing incident
                incident.alerts.append(alert)
                incident.last_seen = now

                # Re-escalate
                if self.escalation_enabled:
                    self._escalate(incident)

                return incident
            else:
                # Window expired — close old incident, start new
                self._close_incident(src)

        # Create new incident
        self._incident_counter += 1
        incident = Incident(
            incident_id=f"INC-{self._incident_counter:05d}",
            source_ip=src,
            alerts=[alert],
            escalated_severity=alert.severity,
            first_seen=now,
            last_seen=now,
        )
        self._incidents[src] = incident

        return incident

    def _escalate(self, incident: Incident):
        """Apply severity escalation rules."""
        count = incident.alert_count

        # Base: max severity of constituent alerts
        max_sev = max(a.severity for a in incident.alerts)

        if count >= 3:
            escalated = max(max_sev, Severity.HIGH)
        elif count >= 2:
            escalated = max(max_sev, Severity.MEDIUM)
        else:
            escalated = max_sev

        # Check for attack chain match
        categories = set(a.category for a in incident.alerts)
        for chain_name, chain_def in ATTACK_CHAINS.items():
            stages = set(chain_def["stages"])
            if stages.issubset(categories):
                escalated = Severity.CRITICAL
                incident.attack_chain = chain_name
                incident.chain_description = chain_def["description"]
                break

        if escalated > incident.escalated_severity:
            incident.is_escalated = True
        incident.escalated_severity = escalated

    def _close_incident(self, src: str):
        """Move an incident to completed."""
        incident = self._incidents.pop(src, None)
        if incident:
            self.completed.append(incident)
            if len(self.completed) > self._max_completed:
                self.completed = self.completed[-self._max_completed:]

    def cleanup_expired(self):
        """Close incidents that have exceeded the time window."""
        now = time.time()
        expired = [
            src for src, inc in self._incidents.items()
            if now - inc.last_seen > self.time_window
        ]
        for src in expired:
            self._close_incident(src)

    @property
    def active_incidents(self) -> List[Incident]:
        return list(self._incidents.values())

    @property
    def active_count(self) -> int:
        return len(self._incidents)

    @property
    def total_incidents(self) -> int:
        return self._incident_counter

    @property
    def stats(self) -> dict:
        return {
            "active_incidents": self.active_count,
            "total_incidents": self.total_incidents,
            "completed_incidents": len(self.completed),
            "escalated": sum(
                1 for inc in self.completed + self.active_incidents
                if inc.is_escalated
            ),
        }
