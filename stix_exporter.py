"""
STIX 2.1 Threat Intelligence Exporter
=======================================
Converts IDS alerts to STIX 2.1 JSON bundles for SIEM ingestion:
  - Cisco SecureX / XDR
  - Splunk
  - Any STIX-compatible platform

Objects produced:
  - Indicator (SDO) — per alert, with STIX pattern
  - Attack-Pattern (SDO) — from MITRE ATT&CK references
  - IPv4-Addr (SCO) — observed source/destination IPs
  - Relationship — links indicators to attack patterns
  - Bundle — wraps all objects
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ids import ThreatEvent, Severity, SEVERITY_LABELS


# STIX 2.1 constants
STIX_SPEC_VERSION = "2.1"
STIX_NAMESPACE = "quantum-sniffer"


def _deterministic_id(stix_type: str, *args: str) -> str:
    """Generate a deterministic STIX ID from type + seed values."""
    seed = "|".join(str(a) for a in args)
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return f"{stix_type}--{h[:8]}-{h[8:12]}-4{h[12:15]}-{h[0]}{h[15:18]}-{h[18:30]}00"


def _timestamp_to_stix(ts: float) -> str:
    """Convert Unix timestamp to STIX datetime string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now_stix() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _severity_to_stix_confidence(severity: Severity) -> int:
    """Map IDS severity to STIX confidence (0-100)."""
    return {
        Severity.INFO: 10,
        Severity.LOW: 25,
        Severity.MEDIUM: 50,
        Severity.HIGH: 75,
        Severity.CRITICAL: 95,
    }.get(severity, 50)


def _category_to_pattern(alert: ThreatEvent) -> str:
    """Generate a STIX indicator pattern from alert category."""
    patterns = {
        "PORT_SCAN": f"[network-traffic:src_ref.type = 'ipv4-addr' AND network-traffic:src_ref.value = '{alert.source_ip}']",
        "SYN_FLOOD": f"[network-traffic:dst_ref.type = 'ipv4-addr' AND network-traffic:dst_ref.value = '{alert.dest_ip}']",
        "DNS_TUNNEL": f"[domain-name:value LIKE '%' AND network-traffic:src_ref.value = '{alert.source_ip}']",
        "DNS_EXFIL": f"[domain-name:value LIKE '%' AND network-traffic:src_ref.value = '{alert.source_ip}']",
        "ARP_SPOOF": f"[mac-addr:value = '{alert.raw_evidence}']" if alert.raw_evidence else "[mac-addr:value LIKE '%']",
        "BRUTE_FORCE": f"[network-traffic:dst_port = {alert.dest_port} AND network-traffic:src_ref.value = '{alert.source_ip}']",
        "ICMP_TUNNEL": f"[network-traffic:protocols[0] = 'icmp' AND network-traffic:src_ref.value = '{alert.source_ip}']",
        "PROTO_ANOMALY": f"[network-traffic:src_ref.value = '{alert.source_ip}']",
        "ANOMALY": f"[network-traffic:src_ref.type = 'ipv4-addr']",
    }
    return patterns.get(alert.category,
                       f"[network-traffic:src_ref.value = '{alert.source_ip or '0.0.0.0'}']")


class STIXExporter:
    """Export IDS alerts as STIX 2.1 bundles."""

    def __init__(self, identity_name: str = "Quantum Sniffer IDS"):
        self.identity_name = identity_name
        self._identity_id = _deterministic_id("identity", identity_name)

    def _make_identity(self) -> Dict[str, Any]:
        """Create the STIX Identity object for this system."""
        return {
            "type": "identity",
            "spec_version": STIX_SPEC_VERSION,
            "id": self._identity_id,
            "created": _now_stix(),
            "modified": _now_stix(),
            "name": self.identity_name,
            "identity_class": "system",
            "description": "Quantum-Resistant Network Analyzer with IDS capabilities",
        }

    def _alert_to_indicator(self, alert: ThreatEvent) -> Dict[str, Any]:
        """Convert a ThreatEvent to a STIX Indicator SDO."""
        indicator_id = _deterministic_id(
            "indicator", alert.category, str(alert.timestamp),
            alert.source_ip or "", alert.description[:50]
        )

        return {
            "type": "indicator",
            "spec_version": STIX_SPEC_VERSION,
            "id": indicator_id,
            "created": _timestamp_to_stix(alert.timestamp),
            "modified": _timestamp_to_stix(alert.timestamp),
            "name": f"{alert.category}: {alert.description[:80]}",
            "description": alert.explanation or alert.description,
            "pattern": _category_to_pattern(alert),
            "pattern_type": "stix",
            "valid_from": _timestamp_to_stix(alert.timestamp),
            "indicator_types": ["malicious-activity"],
            "confidence": _severity_to_stix_confidence(alert.severity),
            "created_by_ref": self._identity_id,
            "labels": [
                f"severity:{SEVERITY_LABELS.get(alert.severity, 'UNKNOWN')}",
                f"category:{alert.category}",
            ],
            "external_references": (
                [{"source_name": "mitre-attack", "description": alert.mitre_ref}]
                if alert.mitre_ref else []
            ),
        }

    def _alert_to_attack_pattern(self, alert: ThreatEvent) -> Optional[Dict[str, Any]]:
        """Create a STIX Attack-Pattern from MITRE reference."""
        if not alert.mitre_ref:
            return None

        ap_id = _deterministic_id("attack-pattern", alert.mitre_ref)
        technique_id = alert.mitre_ref.split(" - ")[0].strip()

        return {
            "type": "attack-pattern",
            "spec_version": STIX_SPEC_VERSION,
            "id": ap_id,
            "created": _now_stix(),
            "modified": _now_stix(),
            "name": alert.mitre_ref,
            "external_references": [{
                "source_name": "mitre-attack",
                "external_id": technique_id,
                "url": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/",
            }],
        }

    def _make_ipv4_sco(self, ip: str) -> Dict[str, Any]:
        """Create a STIX IPv4 Address SCO."""
        return {
            "type": "ipv4-addr",
            "spec_version": STIX_SPEC_VERSION,
            "id": _deterministic_id("ipv4-addr", ip),
            "value": ip,
        }

    def _make_relationship(self, source_id: str, target_id: str,
                           relationship_type: str = "indicates") -> Dict[str, Any]:
        """Create a STIX Relationship SRO."""
        return {
            "type": "relationship",
            "spec_version": STIX_SPEC_VERSION,
            "id": _deterministic_id("relationship", source_id, target_id),
            "created": _now_stix(),
            "modified": _now_stix(),
            "relationship_type": relationship_type,
            "source_ref": source_id,
            "target_ref": target_id,
        }

    def export_bundle(self, alerts: List[ThreatEvent],
                      minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Export alerts as a STIX 2.1 Bundle.

        Args:
            alerts: List of ThreatEvent objects to export.
            minutes: If set, only include alerts from the last N minutes.

        Returns:
            STIX 2.1 Bundle as a dict (JSON-serializable).
        """
        # Time filter
        if minutes:
            cutoff = time.time() - (minutes * 60)
            alerts = [a for a in alerts if a.timestamp >= cutoff]

        objects: List[Dict[str, Any]] = []
        seen_ips = set()
        seen_attack_patterns = set()

        # Identity
        objects.append(self._make_identity())

        for alert in alerts:
            # Indicator
            indicator = self._alert_to_indicator(alert)
            objects.append(indicator)

            # Attack Pattern (deduplicated by MITRE ref)
            if alert.mitre_ref and alert.mitre_ref not in seen_attack_patterns:
                ap = self._alert_to_attack_pattern(alert)
                if ap:
                    objects.append(ap)
                    seen_attack_patterns.add(alert.mitre_ref)
                    # Relationship: indicator → attack-pattern
                    objects.append(self._make_relationship(
                        indicator["id"], ap["id"], "indicates"
                    ))

            # IPv4 SCOs (deduplicated)
            for ip in [alert.source_ip, alert.dest_ip]:
                if ip and ip not in seen_ips:
                    objects.append(self._make_ipv4_sco(ip))
                    seen_ips.add(ip)

        bundle_id = f"bundle--{uuid.uuid4()}"
        return {
            "type": "bundle",
            "id": bundle_id,
            "objects": objects,
        }

    def export_json(self, alerts: List[ThreatEvent],
                    minutes: Optional[int] = None,
                    indent: int = 2) -> str:
        """Export as formatted JSON string."""
        bundle = self.export_bundle(alerts, minutes)
        return json.dumps(bundle, indent=indent, default=str)
