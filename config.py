"""
Configuration Management
=========================
YAML/JSON configuration file support:
  • Loads config from config.yaml or config.json
  • Validates and provides defaults for all settings
  • CLI flags override config file values
  • Environment variable overrides (QS_ prefix)

Usage:
  Place config.yaml in the project root or pass --config path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────
# Configuration Schema
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CaptureConfig:
    """Packet capture settings."""
    interface: Optional[str] = None
    bpf_filter: Optional[str] = None
    queue_size: int = 10000
    backpressure_strategy: str = "drop"  # "drop" | "block"

@dataclass
class IDSConfig_:
    """IDS detection settings."""
    sensitivity: str = "medium"  # "low" | "medium" | "high"
    port_scan_threshold: int = 15
    syn_flood_threshold: int = 100
    brute_force_threshold: int = 10
    dns_entropy_threshold: float = 3.5

@dataclass
class PQCConfig:
    """Post-quantum cryptography settings."""
    enabled: bool = True
    log_dir: str = "./pqc_logs"
    transport_enabled: bool = True
    key_rotation_interval: int = 1000
    replay_window_sec: float = 30.0

@dataclass
class DistributedConfig:
    """Distributed sensor/aggregator settings."""
    mode: Optional[str] = None  # "sensor" | "aggregator"
    aggregator_host: str = "localhost"
    aggregator_port: int = 9100
    max_clients: int = 50
    heartbeat_interval: int = 10
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10

@dataclass
class DashboardConfig:
    """Dashboard settings."""
    terminal_enabled: bool = True
    web_enabled: bool = False
    web_port: int = 5000
    web_host: str = "0.0.0.0"

@dataclass
class MetricsConfig:
    """Prometheus metrics settings."""
    enabled: bool = True
    port: int = 9090

@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    file: Optional[str] = None
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    max_file_size_mb: int = 50
    backup_count: int = 3

@dataclass
class ExportConfig:
    """Data export settings."""
    analytics_path: Optional[str] = None
    stix_path: Optional[str] = None
    geo_enabled: bool = False

@dataclass
class QuantumSnifferConfig:
    """Top-level configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    ids: IDSConfig_ = field(default_factory=IDSConfig_)
    pqc: PQCConfig = field(default_factory=PQCConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config_dict: dict, prefix: str = "QS") -> dict:
    """Apply environment variable overrides with QS_ prefix.

    Mapping: QS_CAPTURE_INTERFACE → config['capture']['interface']
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue
        parts = key[len(prefix) + 1:].lower().split("_", 1)
        if len(parts) == 2:
            section, param = parts
            if section in config_dict and isinstance(config_dict[section], dict):
                # Type coercion
                existing = config_dict[section].get(param)
                if isinstance(existing, bool):
                    config_dict[section][param] = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, int):
                    try:
                        config_dict[section][param] = int(value)
                    except ValueError:
                        pass
                elif isinstance(existing, float):
                    try:
                        config_dict[section][param] = float(value)
                    except ValueError:
                        pass
                else:
                    config_dict[section][param] = value
    return config_dict


def _dict_to_config(d: dict) -> QuantumSnifferConfig:
    """Convert a flat dict to QuantumSnifferConfig."""
    return QuantumSnifferConfig(
        capture=CaptureConfig(**{k: v for k, v in d.get("capture", {}).items()
                                  if k in CaptureConfig.__dataclass_fields__}),
        ids=IDSConfig_(**{k: v for k, v in d.get("ids", {}).items()
                          if k in IDSConfig_.__dataclass_fields__}),
        pqc=PQCConfig(**{k: v for k, v in d.get("pqc", {}).items()
                          if k in PQCConfig.__dataclass_fields__}),
        distributed=DistributedConfig(**{k: v for k, v in d.get("distributed", {}).items()
                                          if k in DistributedConfig.__dataclass_fields__}),
        dashboard=DashboardConfig(**{k: v for k, v in d.get("dashboard", {}).items()
                                      if k in DashboardConfig.__dataclass_fields__}),
        metrics=MetricsConfig(**{k: v for k, v in d.get("metrics", {}).items()
                                  if k in MetricsConfig.__dataclass_fields__}),
        logging=LoggingConfig(**{k: v for k, v in d.get("logging", {}).items()
                                  if k in LoggingConfig.__dataclass_fields__}),
        export=ExportConfig(**{k: v for k, v in d.get("export", {}).items()
                                if k in ExportConfig.__dataclass_fields__}),
    )


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> QuantumSnifferConfig:
    """
    Load configuration with priority: CLI flags > env vars > config file > defaults.
    """
    # Start with defaults
    config_dict = asdict(QuantumSnifferConfig())

    # Load config file
    if config_path:
        path = Path(config_path)
    else:
        # Auto-discover
        for candidate in ["config.yaml", "config.yml", "config.json"]:
            path = Path(candidate)
            if path.exists():
                break
        else:
            path = None

    if path and path.exists():
        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                if HAS_YAML:
                    file_config = yaml.safe_load(f) or {}
                else:
                    print("  ⚠️  PyYAML not installed, skipping YAML config file")
                    file_config = {}
            else:
                file_config = json.load(f)
        config_dict = _deep_merge(config_dict, file_config)

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Apply CLI overrides
    if cli_overrides:
        config_dict = _deep_merge(config_dict, cli_overrides)

    return _dict_to_config(config_dict)


def save_default_config(path: str = "config.yaml"):
    """Save default config to a YAML file."""
    config = asdict(QuantumSnifferConfig())
    if HAS_YAML:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback to JSON
        path = path.replace(".yaml", ".json").replace(".yml", ".json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    print(f"  Default config saved to: {path}")
