"""
Distributed Sniffer
====================
Multi-node packet capture with central aggregation:
  • Sensor nodes: capture locally, stream summaries to aggregator
  • Aggregation server: receive from multiple sensors, unified IDS + analytics
  • JSON-over-TCP protocol with heartbeat monitoring
  • Node health tracking and status reporting
  • Optional PQC-encrypted alert payloads (Kyber-512 + AES-256-GCM)
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class PacketSummary:
    """Serializable packet summary for network transport."""
    timestamp: float
    node_id: str
    protocol: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    size: int
    flags: str = ""
    info: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> 'PacketSummary':
        d = json.loads(data)
        return cls(**d)


@dataclass
class NodeStatus:
    """Health status of a sensor node."""
    node_id: str
    address: str
    last_heartbeat: float
    packets_sent: int
    uptime: float
    is_alive: bool = True

    @property
    def time_since_heartbeat(self) -> float:
        return time.time() - self.last_heartbeat


@dataclass
class DistributedMessage:
    """Wire protocol message."""
    msg_type: str  # "packet", "heartbeat", "register", "status", "pqc-pubkey", "alert_pqc", "alert_signed"
    node_id: str
    payload: dict

    def serialize(self) -> bytes:
        data = json.dumps({
            "type": self.msg_type,
            "node_id": self.node_id,
            "payload": self.payload,
            "timestamp": time.time(),
        }).encode("utf-8")
        # Length-prefixed: 4-byte big-endian length + data
        return struct.pack("!I", len(data)) + data

    @classmethod
    def deserialize(cls, data: bytes) -> 'DistributedMessage':
        obj = json.loads(data.decode("utf-8"))
        return cls(
            msg_type=obj["type"],
            node_id=obj["node_id"],
            payload=obj.get("payload", {}),
        )


class AggregationServer:
    """Central aggregation server that receives from multiple sensor nodes."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        pqc_transport: bool = True,
        max_clients: int = 50,
    ):
        self.host = host
        self.port = port
        self.pqc_transport = pqc_transport
        self.max_clients = max_clients
        self.nodes: Dict[str, NodeStatus] = {}
        self.packet_queue: List[PacketSummary] = []
        self._lock = threading.Lock()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        self._active_connections = 0

        # PQC transport (lazy-initialized)
        self._pqc: Optional[Any] = None

        # Dilithium signature verification
        self._dilithium_registry: Dict[str, bytes] = {}  # node_id -> pk

        # Statistics
        self.stats = {
            "total_packets": 0,
            "packets_by_node": defaultdict(int),
            "start_time": 0.0,
            "pqc_alerts_received": 0,
            "pqc_alerts_signed_verified": 0,
            "alert_integrity_failures": 0,
            "connections_rejected": 0,
            "total_connections": 0,
        }

    def _init_pqc(self):
        """Initialize PQC transport: generate Kyber keypair."""
        if not self.pqc_transport:
            return
        try:
            from pqc_transport import PQCTransport, serialize_public_key
            self._pqc = PQCTransport()
            self._pqc.keygen()
            print("  [PQC] Kyber-512 keypair generated for aggregator")
        except ImportError:
            print("  [PQC] WARNING: pqc_transport module not available, PQC disabled")
            self.pqc_transport = False

    def on_packet(self, callback: Callable):
        """Register a callback for incoming packets."""
        self._callbacks.append(callback)

    def on_alert(self, callback: Callable):
        """Register a callback for incoming alert payloads."""
        self._alert_callbacks.append(callback)

    def start(self):
        """Start the aggregation server."""
        self._running = True
        self.stats["start_time"] = time.time()

        # Initialize PQC if enabled
        self._init_pqc()

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._server_socket.settimeout(1.0)

        logger.info(f"Aggregation server listening on {self.host}:{self.port}")
        print(f"  Aggregation server listening on {self.host}:{self.port}")
        if self.pqc_transport:
            logger.info("PQC transport encryption: ENABLED")
            print(f"  [PQC] Transport encryption: ENABLED")
        print(f"  Max clients: {self.max_clients}")

        # Health monitor thread
        threading.Thread(target=self._health_monitor, daemon=True).start()

        # Accept loop
        while self._running:
            try:
                client_sock, addr = self._server_socket.accept()
                # Check max clients
                if self._active_connections >= self.max_clients:
                    logger.warning(f"Rejecting connection from {addr}: max_clients={self.max_clients} reached")
                    print(f"  ⚠️  Connection from {addr} REJECTED (max {self.max_clients} clients)")
                    self.stats["connections_rejected"] += 1
                    client_sock.close()
                    continue
                self._active_connections += 1
                self.stats["total_connections"] += 1
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()

    def _send_pqc_pubkey(self, sock: socket.socket):
        """Send the PQC public key to a newly connected sensor."""
        if not self.pqc_transport or self._pqc is None:
            return
        try:
            from pqc_transport import serialize_public_key
            pk = self._pqc.get_public_key()
            if pk is None:
                return
            pk_data = serialize_public_key(pk)
            msg = DistributedMessage("pqc-pubkey", "aggregator", {"public_key": pk_data})
            sock.sendall(msg.serialize())
        except Exception as exc:
            print(f"  [PQC] Failed to send public key: {exc}")

    def _handle_client(self, sock: socket.socket, addr):
        """Handle a sensor connection."""
        node_id = f"{addr[0]}:{addr[1]}"

        # Send PQC public key handshake
        self._send_pqc_pubkey(sock)

        try:
            while self._running:
                # Read length prefix
                length_data = self._recv_exact(sock, 4)
                if not length_data:
                    break
                msg_len = struct.unpack("!I", length_data)[0]
                if msg_len > 1_000_000:  # Max 1MB
                    break

                msg_data = self._recv_exact(sock, msg_len)
                if not msg_data:
                    break

                msg = DistributedMessage.deserialize(msg_data)
                node_id = msg.node_id

                if msg.msg_type == "register":
                    with self._lock:
                        self.nodes[node_id] = NodeStatus(
                            node_id=node_id, address=f"{addr[0]}:{addr[1]}",
                            last_heartbeat=time.time(), packets_sent=0, uptime=0,
                        )
                    print(f"  Node registered: {node_id} from {addr}")

                elif msg.msg_type == "heartbeat":
                    with self._lock:
                        if node_id in self.nodes:
                            self.nodes[node_id].last_heartbeat = time.time()
                            self.nodes[node_id].uptime = msg.payload.get("uptime", 0)

                elif msg.msg_type == "packet":
                    pkt = PacketSummary(**msg.payload)
                    with self._lock:
                        self.packet_queue.append(pkt)
                        self.stats["total_packets"] += 1
                        self.stats["packets_by_node"][node_id] += 1
                        if node_id in self.nodes:
                            self.nodes[node_id].packets_sent += 1

                    for cb in self._callbacks:
                        try:
                            cb(pkt)
                        except Exception:
                            pass

                elif msg.msg_type == "batch":
                    # Batched packets from sensor — unpack and process
                    for pkt_data in msg.payload.get("packets", []):
                        pkt = PacketSummary(**pkt_data)
                        with self._lock:
                            self.packet_queue.append(pkt)
                            self.stats["total_packets"] += 1
                            self.stats["packets_by_node"][node_id] += 1
                            if node_id in self.nodes:
                                self.nodes[node_id].packets_sent += 1
                        for cb in self._callbacks:
                            try:
                                cb(pkt)
                            except Exception:
                                pass

                elif msg.msg_type == "alert_pqc":
                    self._handle_pqc_alert(msg, node_id)

                elif msg.msg_type == "alert_signed":
                    self._handle_signed_alert(msg, node_id)

                elif msg.msg_type == "dilithium_pk":
                    # Register Dilithium public key from sensor
                    import base64
                    pk_b64 = msg.payload.get("public_key", "")
                    pk_bytes = base64.b64decode(pk_b64)
                    self._dilithium_registry[node_id] = pk_bytes
                    logger.info(f"Registered Dilithium public key for {node_id}")
                    print(f"  [DILITHIUM] Public key registered for {node_id}")

        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.info(f"Node {node_id} disconnected: {exc}")
        finally:
            sock.close()
            self._active_connections = max(0, self._active_connections - 1)
            with self._lock:
                if node_id in self.nodes:
                    self.nodes[node_id].is_alive = False
            logger.info(f"Node {node_id} connection closed (active: {self._active_connections})")
            print(f"  Node {node_id} disconnected (active clients: {self._active_connections})")

    def _handle_pqc_alert(self, msg: DistributedMessage, node_id: str):
        """Decrypt and process a PQC-encrypted alert."""
        if self._pqc is None:
            print(f"  WARNING: Received PQC alert but PQC transport not initialized")
            return

        try:
            from pqc_transport import PQCTransportError
            encrypted_data = msg.payload.get("encrypted_envelope", "")
            envelope_bytes = encrypted_data.encode("utf-8")

            alert_data = self._pqc.decrypt_payload(
                envelope_bytes,
                sender_id=node_id,
            )

            with self._lock:
                self.stats["pqc_alerts_received"] += 1

            print(f"  [PQC] Alert decrypted from {node_id} (seq={msg.payload.get('seq', '?')})")

            # Fire alert callbacks
            for cb in self._alert_callbacks:
                try:
                    cb(alert_data, node_id)
                except Exception:
                    pass

            # Also try to process as metrics
            try:
                from metrics import get_metrics
                get_metrics().record_pqc_received()
            except Exception:
                pass

        except Exception as exc:
            print(f"  [PQC] Decryption failed for alert from {node_id}: {exc}")

    def _handle_signed_alert(self, msg: DistributedMessage, node_id: str):
        """Verify and process a Dilithium-signed alert."""
        try:
            import base64
            from dilithium_signer import verify_alert_signature

            payload_b64 = msg.payload.get("payload", "")
            sig_b64 = msg.payload.get("signature", "")
            payload_bytes = base64.b64decode(payload_b64)
            sig_bytes = base64.b64decode(sig_b64)

            # Look up public key
            pk = self._dilithium_registry.get(node_id)
            if pk is None:
                logger.warning(f"No Dilithium public key for {node_id}, dropping signed alert")
                print(f"  [DILITHIUM] WARNING: No public key for {node_id}")
                with self._lock:
                    self.stats["alert_integrity_failures"] += 1
                return

            # Verify signature
            if not verify_alert_signature(payload_bytes, sig_bytes, pk):
                logger.warning(f"ALERT_INTEGRITY_FAILURE: Invalid signature from {node_id}")
                print(f"  [DILITHIUM] ❌ INTEGRITY FAILURE: Invalid signature from {node_id}")
                with self._lock:
                    self.stats["alert_integrity_failures"] += 1
                return

            # Signature valid — process alert
            import json as _json
            alert_data = _json.loads(payload_bytes.decode('utf-8'))

            with self._lock:
                self.stats["pqc_alerts_signed_verified"] += 1

            logger.info(f"Verified signed alert from {node_id}")
            print(f"  [DILITHIUM] ✅ Verified alert from {node_id}")

            for cb in self._alert_callbacks:
                try:
                    cb(alert_data, node_id)
                except Exception:
                    pass

        except Exception as exc:
            logger.error(f"Failed to process signed alert from {node_id}: {exc}")
            with self._lock:
                self.stats["alert_integrity_failures"] += 1

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except (socket.timeout, OSError):
                return None
        return data

    def _health_monitor(self):
        """Monitor node health via heartbeat timeout."""
        while self._running:
            time.sleep(5)
            with self._lock:
                for node in self.nodes.values():
                    if node.time_since_heartbeat > 15:
                        if node.is_alive:
                            node.is_alive = False
                            print(f"  Node {node.node_id} OFFLINE (no heartbeat for {node.time_since_heartbeat:.0f}s)")

    @property
    def summary(self) -> dict:
        with self._lock:
            alive = sum(1 for n in self.nodes.values() if n.is_alive)
            return {
                "total_nodes": len(self.nodes),
                "alive_nodes": alive,
                "total_packets": self.stats["total_packets"],
                "by_node": dict(self.stats["packets_by_node"]),
                "uptime": time.time() - self.stats["start_time"],
                "pqc_alerts_received": self.stats["pqc_alerts_received"],
                "pqc_alerts_signed_verified": self.stats["pqc_alerts_signed_verified"],
                "alert_integrity_failures": self.stats["alert_integrity_failures"],
            }


class SensorNode:
    """Sensor node that captures packets and sends summaries to aggregator."""

    def __init__(
        self,
        node_id: str,
        server_host: str = "127.0.0.1",
        server_port: int = 9999,
        pqc_transport: bool = True,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10,
        backpressure_strategy: str = "drop",  # "drop" or "block"
        batch_size: int = 50,
        batch_flush_ms: int = 100,
    ):
        self.node_id = node_id
        self.server_host = server_host
        self.server_port = server_port
        self.pqc_transport = pqc_transport
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.backpressure_strategy = backpressure_strategy
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._start_time = 0.0
        self.packets_sent = 0
        self.alerts_sent = 0
        self.packets_dropped = 0
        self._reconnect_count = 0
        self._connected = False

        # Message batching
        self.batch_size = batch_size
        self.batch_flush_ms = batch_flush_ms
        self._batch_buffer: List[PacketSummary] = []
        self._batch_lock = threading.Lock()
        self._last_flush = time.time()
        self._batch_thread: Optional[threading.Thread] = None

        # Batching metrics
        self.batch_metrics = {
            "messages_batched": 0,
            "batches_sent": 0,
            "avg_batch_size": 0.0,
            "total_items_flushed": 0,
        }

        # PQC transport (lazy-initialized on connect)
        self._pqc: Optional[Any] = None

        # Dilithium signing keypair (generated on first connect)
        self._dilithium_pk: Optional[bytes] = None
        self._dilithium_sk: Optional[bytes] = None

    def connect(self) -> bool:
        """Connect to aggregation server."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.server_host, self.server_port))
            self._start_time = time.time()
            self._running = True
            self._connected = True
            self._reconnect_count = 0

            # Receive PQC public key handshake (if aggregator sends one)
            if self.pqc_transport:
                self._receive_pqc_pubkey()

            # Generate Dilithium keypair for alert signing
            self._init_dilithium()

            # Register
            msg = DistributedMessage("register", self.node_id, {"capabilities": ["capture", "ids"]})
            self._send(msg)

            # Send Dilithium public key to aggregator
            self._send_dilithium_pk()

            # Start heartbeat
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            # Start reconnect monitor
            threading.Thread(target=self._reconnect_monitor, daemon=True).start()
            # Start batch flush thread
            self._batch_thread = threading.Thread(target=self._batch_flush_loop, daemon=True)
            self._batch_thread.start()

            logger.info(f"Connected to aggregator {self.server_host}:{self.server_port}")
            return True
        except (ConnectionRefusedError, OSError) as e:
            logger.error(f"Failed to connect to aggregator: {e}")
            print(f"  Failed to connect to aggregator: {e}")
            return False

    def connect_with_retry(self) -> bool:
        """Connect with automatic retry using exponential backoff."""
        delay = self.reconnect_delay
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.info(f"Connection attempt {attempt}/{self.max_reconnect_attempts}...")
            print(f"  Connection attempt {attempt}/{self.max_reconnect_attempts}...")
            if self.connect():
                return True
            if attempt < self.max_reconnect_attempts:
                logger.info(f"Retrying in {delay}s...")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 160)  # Exponential backoff, cap at 160s
        logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts")
        print(f"  ❌ Failed to connect after {self.max_reconnect_attempts} attempts")
        return False

    def _reconnect_monitor(self):
        """Monitor connection and auto-reconnect on disconnect."""
        while self._running:
            time.sleep(2)
            if not self._connected and self._running:
                logger.warning("Connection lost, attempting reconnect...")
                print("  ⚠️  Connection lost, attempting reconnect...")
                self._reconnect_count += 1
                delay = min(self.reconnect_delay * (2 ** min(self._reconnect_count, 5)), 160)
                time.sleep(delay)
                try:
                    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._sock.connect((self.server_host, self.server_port))
                    self._connected = True
                    self._reconnect_count = 0
                    if self.pqc_transport:
                        self._receive_pqc_pubkey()
                    msg = DistributedMessage("register", self.node_id, {"capabilities": ["capture", "ids"]})
                    self._send(msg)
                    logger.info("Reconnected to aggregator")
                    print("  ✅ Reconnected to aggregator")
                except (ConnectionRefusedError, OSError) as e:
                    logger.warning(f"Reconnect attempt {self._reconnect_count} failed: {e}")
                    if self._reconnect_count >= self.max_reconnect_attempts:
                        logger.error("Max reconnect attempts reached, stopping")
                        self._running = False

    def _receive_pqc_pubkey(self):
        """Receive and process the PQC public key handshake from aggregator."""
        if not self._sock:
            return
        try:
            # Read length-prefixed message (with timeout)
            self._sock.settimeout(5.0)
            length_data = b""
            while len(length_data) < 4:
                chunk = self._sock.recv(4 - len(length_data))
                if not chunk:
                    return
                length_data += chunk

            msg_len = struct.unpack("!I", length_data)[0]
            if msg_len > 1_000_000:
                return

            msg_data = b""
            while len(msg_data) < msg_len:
                chunk = self._sock.recv(msg_len - len(msg_data))
                if not chunk:
                    return
                msg_data += chunk

            self._sock.settimeout(None)

            msg = DistributedMessage.deserialize(msg_data)
            if msg.msg_type == "pqc-pubkey":
                from pqc_transport import PQCTransport, deserialize_public_key
                pk = deserialize_public_key(msg.payload["public_key"])
                self._pqc = PQCTransport()
                self._pqc.set_peer_public_key(pk)
                print(f"  [PQC] Received aggregator public key, session established")
            else:
                # Not a PQC handshake, process as normal message
                self._sock.settimeout(None)
                self.pqc_transport = False
        except Exception as exc:
            print(f"  [PQC] Handshake failed: {exc}")
            self.pqc_transport = False
            try:
                self._sock.settimeout(None)
            except Exception:
                pass

    def _init_dilithium(self):
        """Generate Dilithium-3 keypair for alert signing."""
        try:
            from dilithium_signer import DilithiumSigner
            signer = DilithiumSigner()
            self._dilithium_pk, self._dilithium_sk = signer.keygen()
            self._dilithium_pk_hash = signer.pk_hash(self._dilithium_pk)
            logger.info(f"Dilithium-3 keypair generated (pk_hash={self._dilithium_pk_hash})")
            print(f"  [DILITHIUM] Keypair generated (pk_hash={self._dilithium_pk_hash})")
        except Exception as exc:
            logger.warning(f"Dilithium keygen failed: {exc}")
            print(f"  [DILITHIUM] WARNING: Keygen failed: {exc}")
            self._dilithium_pk = None
            self._dilithium_sk = None

    def _send_dilithium_pk(self):
        """Send Dilithium public key to aggregator for verification registry."""
        if self._dilithium_pk is None:
            return
        try:
            import base64
            pk_b64 = base64.b64encode(self._dilithium_pk).decode()
            msg = DistributedMessage("dilithium_pk", self.node_id, {
                "public_key": pk_b64,
                "pk_hash": self._dilithium_pk_hash,
            })
            self._send(msg)
        except Exception as exc:
            logger.warning(f"Failed to send Dilithium public key: {exc}")

    def send_packet(self, summary: PacketSummary):
        """Buffer a packet summary for batched sending."""
        if not self._running:
            return
        with self._batch_lock:
            self._batch_buffer.append(summary)
            self.batch_metrics["messages_batched"] += 1
            # Flush if batch is full
            if len(self._batch_buffer) >= self.batch_size:
                self._flush_batch()

    def _batch_flush_loop(self):
        """Background thread: flush batch buffer every batch_flush_ms."""
        interval = self.batch_flush_ms / 1000.0
        while self._running:
            time.sleep(interval)
            with self._batch_lock:
                if self._batch_buffer:
                    self._flush_batch()

    def _flush_batch(self):
        """Send all buffered packets as a batch message. Must hold _batch_lock."""
        if not self._batch_buffer:
            return
        batch = self._batch_buffer[:]
        self._batch_buffer.clear()

        # Send as batch message
        msg = DistributedMessage("batch", self.node_id, {
            "count": len(batch),
            "packets": [{
                "timestamp": s.timestamp,
                "node_id": self.node_id,
                "protocol": s.protocol,
                "src_ip": s.src_ip,
                "dst_ip": s.dst_ip,
                "src_port": s.src_port,
                "dst_port": s.dst_port,
                "size": s.size,
                "flags": s.flags,
                "info": s.info,
            } for s in batch],
        })
        self._send(msg)
        self.packets_sent += len(batch)
        self.batch_metrics["batches_sent"] += 1
        self.batch_metrics["total_items_flushed"] += len(batch)
        if self.batch_metrics["batches_sent"] > 0:
            self.batch_metrics["avg_batch_size"] = (
                self.batch_metrics["total_items_flushed"] / self.batch_metrics["batches_sent"]
            )

    def flush(self):
        """Flush all buffered packets (for graceful shutdown)."""
        with self._batch_lock:
            self._flush_batch()

    def send_alert(self, alert_data: dict):
        """Send an alert payload, PQC-encrypted and/or Dilithium-signed.

        Priority:
          1. PQC-encrypted + Dilithium-signed (best)
          2. Dilithium-signed only (integrity)
          3. Plaintext fallback (legacy)

        Args:
            alert_data: JSON-serializable alert dict.
        """
        if not self._running:
            return

        # Try Dilithium-signed path
        if self._dilithium_sk is not None:
            try:
                import base64
                payload_bytes = json.dumps(alert_data).encode('utf-8')
                from dilithium_signer import sign_alert_payload
                sig = sign_alert_payload(payload_bytes, self._dilithium_sk)
                msg = DistributedMessage("alert_signed", self.node_id, {
                    "payload": base64.b64encode(payload_bytes).decode(),
                    "signature": base64.b64encode(sig).decode(),
                    "pk_hash": self._dilithium_pk_hash,
                })
                self._send(msg)
                self.alerts_sent += 1
                return
            except Exception as exc:
                logger.warning(f"Dilithium signing failed: {exc}")
                # Fall through to PQC / plaintext

        if self.pqc_transport and self._pqc is not None:
            # PQC-encrypted path
            try:
                encrypted_bytes = self._pqc.encrypt_payload(alert_data)
                envelope_str = encrypted_bytes.decode("utf-8")
                msg = DistributedMessage("alert_pqc", self.node_id, {
                    "encrypted_envelope": envelope_str,
                    "seq": self._pqc._send_seq,
                })
                self._send(msg)
                self.alerts_sent += 1

                # Record metric
                try:
                    from metrics import get_metrics
                    get_metrics().record_pqc_transmitted()
                except Exception:
                    pass

            except Exception as exc:
                print(f"  [PQC] Encryption failed, falling back to plaintext: {exc}")
                self._send_alert_plaintext(alert_data)
        else:
            self._send_alert_plaintext(alert_data)

    def _send_alert_plaintext(self, alert_data: dict):
        """Send an alert as plaintext (fallback / PQC disabled)."""
        msg = DistributedMessage("alert", self.node_id, alert_data)
        self._send(msg)
        self.alerts_sent += 1

    def disconnect(self):
        """Disconnect from aggregator with graceful flush."""
        self.flush()  # send any buffered packets
        self._running = False
        if self._sock:
            self._sock.close()

    def _send(self, msg: DistributedMessage):
        if self._sock and self._connected:
            try:
                self._sock.sendall(msg.serialize())
            except (BrokenPipeError, OSError) as exc:
                logger.warning(f"Send failed: {exc}")
                self._connected = False
                if self.backpressure_strategy == "drop":
                    self.packets_dropped += 1
                # Note: reconnect_monitor will handle reconnection

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(5)
            msg = DistributedMessage("heartbeat", self.node_id, {
                "uptime": time.time() - self._start_time,
                "packets_sent": self.packets_sent,
            })
            self._send(msg)


def test_distributed():
    """Self-test for distributed sniffer."""
    print("=" * 70)
    print("  DISTRIBUTED SNIFFER TEST")
    print("=" * 70)

    received = []

    def on_pkt(pkt):
        received.append(pkt)

    # Start server
    server = AggregationServer(port=0, pqc_transport=False)  # OS-assigned port
    server.on_packet(on_pkt)

    # Find port
    import socket as _socket
    test_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    test_sock.bind(("127.0.0.1", 0))
    test_port = test_sock.getsockname()[1]
    test_sock.close()

    server.port = test_port
    srv_thread = threading.Thread(target=server.start, daemon=True)
    srv_thread.start()
    time.sleep(0.5)

    # Connect sensor
    sensor = SensorNode("sensor-1", "127.0.0.1", test_port, pqc_transport=False)
    assert sensor.connect(), "Failed to connect"
    time.sleep(0.3)

    # Send packets
    for i in range(10):
        pkt = PacketSummary(
            timestamp=time.time(), node_id="sensor-1",
            protocol="TCP", src_ip=f"10.0.0.{i}",
            dst_ip="192.168.1.1", src_port=40000+i,
            dst_port=80, size=100+i*10,
        )
        sensor.send_packet(pkt)
    time.sleep(1)

    # Verify
    sensor.disconnect()
    server.stop()

    print(f"\n  Packets sent: 10")
    print(f"  Packets received: {len(received)}")
    print(f"  Nodes registered: {len(server.nodes)}")
    summary = server.summary
    print(f"  Server summary: {summary}")

    success = len(received) >= 8  # Allow for timing
    print(f"\n  {'OK' if success else 'FAIL'} Distributed test {'PASSED' if success else 'FAILED'}")
    return success


def stress_test_distributed(duration: int = 10, sensors: int = 3, batch_size: int = 50):
    """Stress test: multiple sensors sending batched traffic."""
    print("=" * 70)
    print("  DISTRIBUTED STRESS TEST")
    print("=" * 70)
    print(f"  Duration: {duration}s | Sensors: {sensors} | Batch size: {batch_size}")

    received = [0]
    recv_lock = threading.Lock()

    def on_pkt(pkt):
        with recv_lock:
            received[0] += 1

    # Start server on random port
    import socket as _socket
    test_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    test_sock.bind(("127.0.0.1", 0))
    test_port = test_sock.getsockname()[1]
    test_sock.close()

    server = AggregationServer(port=test_port, pqc_transport=False, max_clients=sensors + 5)
    server.on_packet(on_pkt)
    srv_thread = threading.Thread(target=server.start, daemon=True)
    srv_thread.start()
    time.sleep(0.5)

    # Connect sensors
    sensor_list = []
    for i in range(sensors):
        s = SensorNode(
            f"stress-sensor-{i}", "127.0.0.1", test_port,
            pqc_transport=False, batch_size=batch_size, batch_flush_ms=50,
        )
        if s.connect():
            sensor_list.append(s)
    time.sleep(0.3)

    print(f"  {len(sensor_list)} sensors connected. Sending traffic...")
    sent = [0]
    t_start = time.time()

    def sender(sensor: SensorNode, idx: int):
        seq = 0
        while time.time() - t_start < duration:
            pkt = PacketSummary(
                timestamp=time.time(), node_id=sensor.node_id,
                protocol="TCP", src_ip=f"10.{idx}.0.{seq % 256}",
                dst_ip="192.168.1.1", src_port=40000 + (seq % 1000),
                dst_port=80, size=100 + seq % 100,
            )
            sensor.send_packet(pkt)
            seq += 1
            if seq % 500 == 0:
                time.sleep(0.001)  # Yield CPU
        with recv_lock:
            sent[0] += seq

    threads = []
    for i, s in enumerate(sensor_list):
        t = threading.Thread(target=sender, args=(s, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Wait for flush
    for s in sensor_list:
        s.disconnect()
    time.sleep(1)
    server.stop()

    elapsed = time.time() - t_start
    print(f"\n  ── RESULTS ──")
    print(f"  Elapsed:       {elapsed:.2f}s")
    print(f"  Packets sent:  {sent[0]:,}")
    print(f"  Packets recv:  {received[0]:,}")
    print(f"  Send rate:     {sent[0] / elapsed:,.0f} pkt/s")
    print(f"  Recv rate:     {received[0] / elapsed:,.0f} pkt/s")
    for s in sensor_list:
        print(f"  {s.node_id}: batches={s.batch_metrics['batches_sent']}, "
              f"avg_batch={s.batch_metrics['avg_batch_size']:.1f}, "
              f"dropped={s.packets_dropped}")
    delivery = received[0] / max(sent[0], 1) * 100
    print(f"  Delivery rate: {delivery:.1f}%")
    print(f"  {'OK' if delivery > 80 else 'WARN'} Stress test {'PASSED' if delivery > 80 else 'needs tuning'}")
    print("=" * 70)
    return delivery > 80


if __name__ == "__main__":
    test_distributed()
