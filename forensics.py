"""
Forensic PCAP Export
=====================
Automatic evidence capture for incident response:
  - Ring buffer of last N raw packets
  - On alert trigger: dump surrounding packets to .pcap
  - Named by timestamp + attack type + source IP
  - Uses Scapy's wrpcap() for standard PCAP format
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

try:
    from scapy.all import Ether, wrpcap
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

from ids import ThreatEvent, Severity


class ForensicCapture:
    """
    Ring-buffer packet capture for forensic evidence.

    Maintains a rolling window of raw packets.
    When a HIGH/CRITICAL alert fires, dumps the buffer to a .pcap file.
    """

    def __init__(
        self,
        buffer_size: int = 200,
        output_dir: str = "./forensics",
        min_severity: Severity = Severity.HIGH,
    ):
        self.buffer_size = buffer_size
        self.output_dir = Path(output_dir)
        self.min_severity = min_severity
        self._buffer: deque = deque(maxlen=buffer_size)
        self._capture_count = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_packet(self, raw_bytes: bytes):
        """Add a raw packet to the ring buffer."""
        self._buffer.append({
            "raw": raw_bytes,
            "time": time.time(),
        })

    def on_alert(self, alert: ThreatEvent) -> Optional[str]:
        """
        Called when an IDS alert fires.
        If severity >= min_severity, dump buffer to pcap.

        Returns the pcap filename if dumped, None otherwise.
        """
        if alert.severity < self.min_severity:
            return None

        if not self._buffer:
            return None

        if not HAS_SCAPY:
            return None

        return self._dump_pcap(alert)

    def _dump_pcap(self, alert: ThreatEvent) -> Optional[str]:
        """Dump current buffer to a PCAP file."""
        self._capture_count += 1

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(alert.timestamp))
        src_ip = (alert.source_ip or "unknown").replace(".", "-")
        category = alert.category.lower()

        filename = self.output_dir / f"{timestamp}_{category}_{src_ip}.pcap"

        try:
            packets = []
            for entry in self._buffer:
                try:
                    pkt = Ether(entry["raw"])
                    pkt.time = entry["time"]
                    packets.append(pkt)
                except Exception:
                    continue

            if packets:
                wrpcap(str(filename), packets)
                return str(filename)
        except Exception:
            pass

        return None

    @property
    def buffer_count(self) -> int:
        """Number of packets currently in the ring buffer."""
        return len(self._buffer)

    @property
    def capture_count(self) -> int:
        """Total PCAP files created."""
        return self._capture_count

    @property
    def stats(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "packets_buffered": self.buffer_count,
            "captures_created": self.capture_count,
            "output_dir": str(self.output_dir),
            "min_severity": self.min_severity.name,
        }
